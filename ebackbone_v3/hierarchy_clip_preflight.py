"""Real-data two-GPU gates for Mini training, with no full-training launch."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from . import hierarchy_clip_train as train
from .hierarchy_clip_checkpoint import (capture_rng_state, save_epoch, load_checkpoint,
    inspect_training_checkpoint, load_transfer_bundle, file_sha256)
from .hierarchy_clip_data import HierarchySequenceDataset, sequence_collate
from .hierarchy_clip_runtime import build_model
from .hierarchy_clip_staging import set_staged_train
from .hierarchy_ddp import ExactBatchShard
from .v1_training import atomic_json


def assert_training_gate(config, config_path=None):
    del config_path
    path = Path(config["training_gate_path"])
    gate = json.loads(path.read_text())
    if (gate.get("status") != "PASS" or gate.get("all_variants_passed") is not True
            or gate.get("config_sha256") != train.config_fingerprint(config)
            or set(gate.get("variants", [])) != set(train.VARIANTS)):
        raise ValueError("training requires matching successful preflights for all three variants")
    for name, digest in gate["source_sha256"].items():
        if file_sha256(Path(__file__).parent / name) != digest:
            raise ValueError(f"preflighted source changed: {name}")
    if set(gate.get("reports", {})) != set(train.VARIANTS):
        raise ValueError("missing per-variant preflight evidence")
    for name, record in gate["reports"].items():
        report_path = Path(record["path"])
        if file_sha256(report_path) != record["sha256"]:
            raise ValueError(f"preflight evidence changed: {name}")
        report = json.loads(report_path.read_text())
        if report.get("status") != "PASS" or report.get("config_sha256") != train.config_fingerprint(config):
            raise ValueError(f"failed or mismatched preflight: {name}")
    return gate


def state_fingerprint(value):
    """Fingerprint nested training state by values, not storage/serialization IDs."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous()
            digest.update(f"tensor:{array.dtype}:{tuple(array.shape)}:".encode())
            digest.update(array.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(f"numpy:{item.dtype}:{item.shape}:".encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for entry in item:
                visit(entry)
        else:
            digest.update((type(item).__name__ + ":" + repr(item) + ";").encode())
    visit(value)
    return digest.hexdigest()


def _flat_grad(model):
    values = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"missing/nonfinite trainable gradient: {name}")
            values.append(parameter.grad.detach().flatten())
        elif parameter.grad is not None:
            raise AssertionError(f"frozen gradient: {name}")
    return torch.cat(values)


def verify_gradient_average(ddp, samples, device):
    """Unequal local 2/1 samples: DDP must match explicit summed local gradients."""
    batch = sequence_collate(samples)
    inputs = train.move_inputs(batch["inputs"], device)
    labels = batch["labels"].to(device)
    set_staged_train(ddp.module)
    ddp.zero_grad(set_to_none=True)
    with ddp.no_sync():
        with train.amp_context(device):
            loss = torch.nn.functional.cross_entropy(ddp(inputs).float(), labels, reduction="sum")
        # Match the production trainer: AMP wraps forward/loss, never backward.
        # Keeping AMP enabled here changes checkpoint-recomputed gradients.
        (loss * 2 / 3).backward()
    expected = _flat_grad(ddp.module).clone()
    dist.all_reduce(expected)
    expected /= 2
    ddp.zero_grad(set_to_none=True)
    with train.amp_context(device):
        loss = torch.nn.functional.cross_entropy(ddp(inputs).float(), labels, reduction="sum")
    (loss * 2 / 3).backward()
    actual = _flat_grad(ddp.module)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-4)
    report = {"explicit_average_matches": True, "max_abs_error": float((actual - expected).abs().max()),
              "unequal_local_sample_counts": [2, 1]}
    ddp.zero_grad(set_to_none=True)
    return report


def verify_validation_reference(model, data, settings, device, distributed_metrics):
    """Compare reduction with a serial replay of both exact shard layouts."""
    report = None
    if dist.get_rank() == 0:
        values = torch.zeros(4, dtype=torch.float64, device=device)
        for rank in (0, 1):
            sampler = ExactBatchShard(len(data), rank, 2, settings["batch_size"], settings["seed"], 1, False)
            for indices in sampler:
                values += train.evaluation_batch(model, [data[i] for i in indices], device,
                                                microbatch_size=settings["microbatch_size"])
        expected = torch.tensor([distributed_metrics["loss"], distributed_metrics["top1"],
                                 distributed_metrics["top5"]], dtype=torch.float64, device=device)
        actual = values[:3] / values[3]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        if int(values[3]) != len(data):
            raise AssertionError("serial reference lost validation samples")
        report = {"serial_shard_replay_matches": True, "samples": len(data),
                  "rank_sample_counts": distributed_metrics["rank_sample_counts"],
                  "max_abs_metric_error": float((actual - expected).abs().max())}
    gathered = [report]
    dist.broadcast_object_list(gathered, src=0)
    return gathered[0]


def _resume_values(model, optimizer, update):
    return {"model": state_fingerprint(model.state_dict()),
            "optimizer": state_fingerprint(optimizer.state_dict()),
            "rng": state_fingerprint(capture_rng_state()),
            "update_metrics": state_fingerprint(update["values"])}


def exact_resume_check(config, variant, output, model, optimizer, identity, samples,
                       device, *, saved_epoch, next_epoch):
    """Reconstruct model, DDP, Adam and rank RNG; compare the exact next update.

    The saved epochs simulate a boundary; they are not completed training epochs.
    One test crosses connectors->selective and another resumes within selective.
    """
    settings = config["training"]
    saved_stage = train.stage_for_epoch(saved_epoch, settings["connector_epochs"])
    next_stage = train.stage_for_epoch(next_epoch, settings["connector_epochs"])
    scheduler = {"schedule": "absolute_cosine50", "completed_epoch": saved_epoch, "total_epochs": 50}
    train.set_epoch_learning_rate(optimizer, saved_epoch)
    saved = save_epoch(output, model, optimizer, epoch=saved_epoch, stage=saved_stage,
        identity=identity, history=[], best={"diagnostic": True}, improved=True, device=device,
        scheduler_state=scheduler, global_step=2,
        sampler_state={"seed": settings["seed"], "epoch": next_epoch, "offset": 0, "world_size": 2},
        extra={"bounded_preflight_simulated_epoch_boundary": True})
    # Caller released the old reducer before registering newly trainable parameters.
    ddp, optimizer, transition = train.setup_stage(model, next_stage, settings, device, optimizer)
    train.set_epoch_learning_rate(optimizer, next_epoch)
    uninterrupted = train.optimizer_update(ddp, samples, optimizer, device, 3,
                                          microbatch_size=settings["microbatch_size"], probe=True)
    expected = _resume_values(model, optimizer, uninterrupted)
    del uninterrupted, ddp, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    restored, provenance = build_model(variant, config)
    restored.to(device)
    ddp, optimizer, _ = train.setup_stage(restored, saved_stage, settings, device)
    state = load_checkpoint(Path(output) / "checkpoint_last.pt", restored, optimizer,
                            identity=identity, device=device)
    if state["scheduler"] != scheduler or state["sampler"]["epoch"] != next_epoch:
        raise AssertionError("resume scheduler/sampler differs")
    del state
    del ddp
    ddp, optimizer, restored_transition = train.setup_stage(restored, next_stage, settings, device, optimizer)
    train.set_epoch_learning_rate(optimizer, next_epoch)
    resumed = train.optimizer_update(ddp, samples, optimizer, device, 3,
                                    microbatch_size=settings["microbatch_size"], probe=True)
    actual = _resume_values(restored, optimizer, resumed)
    if actual != expected:
        raise AssertionError({"exact_resume_mismatch": {k: [expected[k], actual[k]] for k in expected if expected[k] != actual[k]}})
    report = {"exact_next_update": True, "model_optimizer_rng_metrics_bit_exact": True,
              "saved_epoch_boundary_simulation": saved_epoch, "next_epoch_simulation": next_epoch,
              "stage_from": saved_stage, "stage_to": next_stage,
              "preserved_adam_parameters": len(restored_transition["preserved_adam_parameters"]),
              "checkpoint_sha256": saved["checkpoints"]["last"]["sha256"]}
    return restored, ddp, optimizer, report


def run_preflight(config, variant, output_dir):
    output = Path(output_dir).resolve()
    device = train.initialize(config)
    rank = dist.get_rank()
    settings = config["training"]
    try:
        error = None
        if rank == 0:
            if output.exists() and any(output.iterdir()):
                error = "preflight output must be new or empty"
            else:
                output.mkdir(parents=True, exist_ok=True)
        if any(train.gather(error)):
            raise FileExistsError("preflight output must be new or empty")
        locations = {key: config[key] for key in ("manifest_dir", "dataset_root", "raw_cache")}
        training = HierarchySequenceDataset(**locations, split="train", num_windows=4)
        # Stress by largest stored byte size, independent of labels. This is a
        # bounded stress proxy, not a claim of maximum decoded event count.
        training.rows = tuple(sorted(training.rows, key=lambda row: (-row.raw_content_size_bytes, row.sample_id))[:67])
        validation = HierarchySequenceDataset(**locations, split="validation", num_windows=4,
                                              limit=65, seed=settings["seed"])
        model, provenance = build_model(variant, config)
        model.to(device)
        identity = train.build_identity(config, variant, provenance, training, validation)
        identity["bounded_preflight"] = {"train_samples": 67, "validation_samples": 65,
            "training_selection": "largest raw_content_size_bytes, label-independent",
            "train_ids_sha256": state_fingerprint([row.sample_id for row in training.rows]),
            "validation_ids_sha256": state_fingerprint([row.sample_id for row in validation.rows])}
        small_samples = [training[i] for i in range(rank, 3, 2)]
        local_counts = [sample["source"]["event_count"] for sample in small_samples]
        stages = {}
        optimizer = None
        ddp = None
        for stage, epoch in (("connectors", 1), ("selective", 6)):
            if ddp is not None:
                del ddp
            ddp, optimizer, transition = train.setup_stage(model, stage, settings, device, optimizer)
            train.set_epoch_learning_rate(optimizer, epoch)
            gradient = verify_gradient_average(ddp, small_samples, device)
            frozen = {name: state_fingerprint(parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad}
            torch.cuda.reset_peak_memory_stats(device)
            training_metrics = train.run_epoch(ddp, training, settings, device, epoch, optimizer, probe=True)
            validation_metrics = train.run_epoch(ddp, validation, settings, device, epoch)
            for name, parameter in model.named_parameters():
                if name in frozen and state_fingerprint(parameter) != frozen[name]:
                    raise AssertionError(f"frozen weight changed: {name}")
            reference = verify_validation_reference(model, validation, settings, device, validation_metrics)
            capacity = torch.cuda.get_device_properties(device).total_memory
            memory = train.gather({"rank": rank, "allocated_bytes": torch.cuda.max_memory_allocated(device),
                "reserved_bytes": torch.cuda.max_memory_reserved(device), "capacity_bytes": capacity})
            if any(item["reserved_bytes"] > .92 * item["capacity_bytes"] for item in memory):
                raise RuntimeError("preflight memory lacks 8 percent reserved-memory headroom")
            stages[stage] = {"gradient_average": gradient, "train": training_metrics,
                "validation": validation_metrics, "validation_reference": reference,
                "memory_by_rank": memory, "frozen_weights_unchanged": True,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
            if rank == 0:
                print(json.dumps({"event": "stage_preflight_pass", "variant": variant, "stage": stage,
                                  "memory_by_rank": memory}), flush=True)
            if stage == "connectors":
                del ddp
                model, ddp, optimizer, resume_boundary = exact_resume_check(config, variant, output / "resume_boundary",
                    model, optimizer, identity, small_samples, device, saved_epoch=5, next_epoch=6)
        del ddp
        model, ddp, optimizer, resume_selective = exact_resume_check(config, variant, output / "resume_selective",
            model, optimizer, identity, small_samples, device, saved_epoch=6, next_epoch=7)
        # Transfer artifact reconstructs all used model components independently
        # of a task head; the source-task option verifies identical source logits.
        del ddp
        reference = train.evaluation_batch(model, small_samples, device, microbatch_size=32)
        if rank == 0:
            from .hierarchy_clip_checkpoint import export_transfer_bundle
            export_transfer_bundle(output / "transfer_probe.pt", model, identity=identity, epoch=7, stage="selective")
        dist.barrier()
        for parameter in model.parameters():
            if parameter.requires_grad:
                with torch.no_grad():
                    parameter.add_(1)
        load_transfer_bundle(output / "transfer_probe.pt", model, expected_identity=identity)
        reloaded = train.evaluation_batch(model, small_samples, device, microbatch_size=32)
        torch.testing.assert_close(reference, reloaded, rtol=0, atol=0)
        reports = train.gather({"rank": rank, "resume_boundary": resume_boundary,
                              "resume_selective": resume_selective, "source_event_counts": local_counts})
        report = {"status": "PASS", "variant": variant, "config_sha256": train.config_fingerprint(config),
            "identity": identity, "stages": stages, "rank_checks": reports,
            "transfer_reload_bit_exact": True, "actual_local_batch": settings["batch_size"],
            "global_batch": 64, "microbatch_size": settings["microbatch_size"],
            "full_training_launched": False, "HARDVS_accessed": False, "paired_RGB_used": False,
            "final_test_accessed": False}
        if rank == 0:
            atomic_json(output / "report.json", report)
        return report
    finally:
        dist.destroy_process_group()
