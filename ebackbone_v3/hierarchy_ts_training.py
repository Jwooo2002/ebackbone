"""Separate TS-gated hierarchy runner, retaining the parallel V1 optimization protocol.

Epoch/accumulation/checkpoint logic is intentionally isolated so the running
parallel baseline and its resume source fingerprints remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import math

from .hierarchy_ts_data import HierarchyTSDataset as HierarchyDataset, TS_CONTRACT as POINT_CONTRACT, TS_CONTRACT_SHA256 as POINT_CONTRACT_SHA256, collate
from .hierarchy_ts_models import HierarchyTSV1 as HierarchyV1, MODEL_VERSION, profile_macs
from .v1_training import (TrainConfig, atomic_json, atomic_checkpoint, sha256_file,
                          seed_everything, seed_worker, autocast, synchronize,
                          rng_state, restore_rng, source_hashes as baseline_source_hashes)

MODEL_NAMES = ("hierarchy_ts",)


def source_hashes():
    root = Path(__file__).parent
    return {**baseline_source_hashes(), **{name: sha256_file(root / name) for name in
            ("hierarchy_models.py", "hierarchy_data.py", "hierarchy_training.py", "hierarchy.py", "v1.py", "hierarchy_ts_models.py", "hierarchy_ts_data.py", "hierarchy_ts_training.py", "hierarchy_ts.py")}}


def loader(dataset, config, epoch, train):
    # Separate generator: model initialization and branch count cannot affect order.
    generator = torch.Generator().manual_seed(config.seed + epoch)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=train, drop_last=False,
                      generator=generator, num_workers=config.num_workers, collate_fn=collate,
                      worker_init_fn=seed_worker, pin_memory=config.device.startswith("cuda"))


def run_epoch(model, data, config, epoch, *, optimizer=None):
    train = optimizer is not None
    model.train(train)
    batches = loader(data, config, epoch, train)
    loss_sum = correct1 = correct5 = count = 0
    rendering = decoding = forward_seconds = 0.0
    event_total, event_min, event_max = 0, None, 0
    first_update = None
    order = hashlib.sha256()
    synchronize(config)
    start = time.perf_counter()
    if train:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(batches):
        if set(batch["splits"]) != {data.split} or set(batch["source_splits"]) != {"train"}:
            raise ValueError("training/selection loader crossed the source-train boundary")
        inputs = {k: v.to(config.device, non_blocking=True) for k, v in batch["inputs"].items()}
        labels = batch["labels"].to(config.device, non_blocking=True)
        size = len(labels)
        event_counts = batch["inputs"]["event_counts"]
        event_total += int(event_counts.sum())
        event_min = min(event_min, int(event_counts.min())) if event_min is not None else int(event_counts.min())
        event_max = max(event_max, int(event_counts.max()))
        model.capture_gate_stats = batch_index == 0
        with torch.set_grad_enabled(train), autocast(config):
            synchronize(config)
            tick = time.perf_counter()
            logits = model(inputs)
            synchronize(config)
            forward_seconds += time.perf_counter() - tick
            loss = nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite cross-entropy")
        if train:
            group_start = (batch_index // config.accumulation_steps) * config.accumulation_steps * config.batch_size
            group_size = min(config.accumulation_steps * config.batch_size, len(data) - group_start)
            (loss / group_size).backward()
            if (batch_index + 1) % config.accumulation_steps == 0 or batch_index + 1 == len(batches):
                nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
                if first_update is None:
                    first_update = {"gradient_l1_by_stage": {
                        name: sum(float(p.grad.detach().abs().sum()) for p in module.parameters() if p.grad is not None)
                        for name, module in model.named_children() if any(True for _ in module.parameters())}}
                if batch_index == 1:
                    first_update["second_backward_gradient_l1_by_stage"] = {
                        name: sum(float(p.grad.detach().abs().sum()) for p in module.parameters() if p.grad is not None)
                        for name, module in model.named_children() if any(True for _ in module.parameters())}
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update = math.ceil((batch_index + 1) / config.accumulation_steps)
                if update == 1 or update % 100 == 0 or batch_index + 1 == len(batches):
                    print(json.dumps({"event": "optimizer_step", "model": model.name,
                                      "epoch": epoch, "update_in_epoch": update,
                                      "samples_seen": count + size,
                                      "batch_mean_loss": float(loss.detach()) / size,
                                      "device": config.device, "unix_time": time.time()}), flush=True)
        predictions = logits.detach().topk(min(5, logits.shape[1]), dim=1).indices
        correct1 += int((predictions[:, 0] == labels).sum())
        correct5 += int((predictions == labels[:, None]).any(1).sum())
        loss_sum += float(loss.detach())
        count += size
        rendering += batch["render_seconds"]
        decoding += batch["decode_seconds"]
        for sample_id in batch["sample_ids"]:
            order.update(sample_id.encode() + b"\n")
    synchronize(config)
    elapsed = time.perf_counter() - start
    if count != len(data) or count == 0:
        raise ValueError("epoch did not cover the complete selected split")
    return {"loss": loss_sum / count, "top1": correct1 / count, "top5": correct5 / count,
            "samples": count, "sample_order_sha256": order.hexdigest(), "seconds": elapsed,
            "samples_per_second": count / elapsed, "render_ms_per_sample": 1000 * rendering / count,
            "decode_ms_per_sample": 1000 * decoding / count,
            "events": {"total": event_total, "min": event_min, "max": event_max, "mean": event_total / count},
            "first_update": first_update, "gate_first_batch": model.last_gate_stats,
            "forward_ms_per_sample": 1000 * forward_seconds / count}


def verify_checkpoint(path, model, inputs, config):
    state = torch.load(path, map_location="cpu", weights_only=False)
    devices = [1] if config.device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        restored = HierarchyV1(model.num_classes, model.height, model.width).to(config.device).eval()
    restored.load_state_dict(state["model"], strict=True)
    for key, value in model.state_dict().items():
        if not torch.equal(value.cpu(), state["model"][key]):
            raise ValueError(f"saved model tensor differs: {key}")
    model.eval()
    with torch.no_grad(), autocast(config):
        expected, actual = model(inputs), restored(inputs)
    if not torch.equal(expected, actual):
        raise ValueError("strict checkpoint reload changed evaluation logits")
    return {"strict_load": True, "all_state_tensors_equal": True,
            "logits_bit_exact": True, "sha256": sha256_file(path)}


def run_model(name, config: TrainConfig, *, manifest_dir, dataset_root, output_dir, raw_cache=None,
              diagnostic_samples=None, resume=False, stop_after_epoch=None):
    config.validate()
    if name not in MODEL_NAMES:
        raise ValueError("unknown hierarchy model")
    if diagnostic_samples is not None and not 1 <= diagnostic_samples <= 16:
        raise ValueError("diagnostics are limited to 1-16 train samples")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= config.epochs:
        raise ValueError("stop_after_epoch outside configured budget")
    seed_everything(config)
    model = HierarchyV1().to(config.device)
    train = HierarchyDataset(manifest_dir, dataset_root, "train",
                      raw_cache=raw_cache, limit=diagnostic_samples, seed=config.seed)
    validation = (train if diagnostic_samples is not None else
                  HierarchyDataset(manifest_dir, dataset_root, "validation", raw_cache=raw_cache))
    if diagnostic_samples is None:
        train_ids = {r.sample_id for r in train.rows}
        train_hashes = {r.raw_content_sha256 for r in train.rows}
        if train_ids.intersection(r.sample_id for r in validation.rows):
            raise ValueError("train/validation sample identity overlap")
        # Exact duplicate bytes are reported, not used for relabeling/resplitting.
        duplicate_payloads = len(train_hashes.intersection(r.raw_content_sha256 for r in validation.rows))
    else:
        duplicate_payloads = None
    identity = {"schema": 1, "architecture": name, "model_version": MODEL_VERSION,
                "settings": asdict(config), "point_contract": POINT_CONTRACT, "point_contract_sha256": POINT_CONTRACT_SHA256,
                "train_manifest_sha256": train.manifest_sha256,
                "validation_manifest_sha256": validation.manifest_sha256,
                "diagnostic_samples": diagnostic_samples, "source_sha256": source_hashes(),
                "dataset": "N-ImageNet Mini, 100 classes", "torch_version": torch.__version__,
                "numpy_version": np.__version__, "raw_cache": str(Path(raw_cache).resolve()) if raw_cache else None,
                "optimizer": "SGD", "scheduler": "CosineAnnealingLR, eta_min=0, step after each epoch",
                "objective": "single final classifier cross-entropy only", "augmentation": "none",
                "gate": {"location": "after temporal collapse, before frame_stage", "formula": "F*(1+0.5*tanh(A))",
                         "initialization": "zero final projection; original backbone seed and RNG preserved"}}
    output = Path(output_dir).resolve()
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite existing run: {output}")
    if resume and not (output / "checkpoint_last.pt").is_file():
        raise ValueError("resume requires checkpoint_last.pt")
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum,
                                weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    history, best, first_epoch = [], None, 1
    if resume:
        checkpoint = torch.load(output / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        if checkpoint["identity"] != identity:
            raise ValueError("resume identity mismatch: model/settings/manifests/renderer/source/environment")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint["rng"])
        history, best = checkpoint["history"], checkpoint["best"]
        first_epoch = checkpoint["epoch"] + 1
    atomic_json(output / "config.json", identity)
    if config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(config.device)
    last_epoch = stop_after_epoch or config.epochs
    for epoch in range(first_epoch, last_epoch + 1):
        lr = optimizer.param_groups[0]["lr"]
        training = run_epoch(model, train, config, epoch, optimizer=optimizer)
        evaluation = run_epoch(model, validation, config, epoch)
        scheduler.step()
        record = {"epoch": epoch, "learning_rate": lr, "train": training,
                  "train_recheck" if diagnostic_samples is not None else "validation": evaluation}
        history.append(record)
        improved = best is None or (evaluation["top1"], -evaluation["loss"]) > (best["top1"], -best["loss"])
        if improved:
            best = {"epoch": epoch, **evaluation}
        checkpoint = {"identity": identity, "epoch": epoch, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "rng": rng_state(), "history": history, "best": best}
        # Best first: last.pt becomes the authoritative committed epoch. A
        # completed last checkpoint always has its corresponding best artifact.
        if improved:
            atomic_checkpoint(output / "checkpoint_best.pt", checkpoint)
        atomic_checkpoint(output / "checkpoint_last.pt", checkpoint)
        atomic_json(output / "history.json", history)
        print(json.dumps({"model": name, **record}), flush=True)
    if not history:
        raise ValueError("no completed epochs")
    reference = train[0]
    inputs = {k: v.to(config.device) for k, v in collate([reference])["inputs"].items()}
    verification = verify_checkpoint(output / "checkpoint_last.pt", model, inputs, config)
    report = {"status": "complete" if history[-1]["epoch"] == config.epochs else "partial",
              "evidence_kind": "train-only engineering diagnostic" if diagnostic_samples is not None else "full internal-validation comparison",
              "identity": identity, "completed_epochs": history[-1]["epoch"], "best": best,
              "history": history, "profile": profile_macs(model, event_count=reference["inputs"]["points"].shape[0]), "checkpoint_verification": verification,
              "final_test_accessed": False, "train_validation_identical_payload_count": duplicate_payloads,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(config.device) if config.device.startswith("cuda") else None,
              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "process_peak_rss_scope": "process lifetime; not an isolated per-model measurement",
              "device_name": torch.cuda.get_device_name(config.device) if config.device.startswith("cuda") else "CPU",
              "reference_source": reference["source"]}
    atomic_json(output / "report.json", report)
    return report
