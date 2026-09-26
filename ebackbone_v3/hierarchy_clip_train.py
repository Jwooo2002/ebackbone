"""Isolated staged Mini pretraining with exact two-rank global batches.

The public epoch/update helpers also support CPU Gloo verification. Production
launches require an explicit configuration and the completed preflight gate.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .hierarchy_clip_data import HierarchySequenceDataset, sequence_collate
from .hierarchy_clip_checkpoint import inspect_training_checkpoint, load_checkpoint, save_epoch
from .hierarchy_clip_staging import configure_stage, set_staged_train
from .hierarchy_ddp import ExactBatchShard
from .v1_training import atomic_json, seed_worker


VARIANTS = ("hierarchy_temporal", "hierarchy_text", "hierarchy_clip_vit")
STAGE_TRANSITION_POLICY = "preserve connector Adam moments and step; initialize newly unfrozen upper parameters"


def distributed():
    return dist.is_available() and dist.is_initialized()


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if distributed() else (0, 1)


def gather(value):
    if not distributed():
        return [value]
    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def amp_context(device):
    device = torch.device(device)
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def move_inputs(inputs, device):
    return {"packed": {key: value.to(device, non_blocking=True) for key, value in inputs["packed"].items()},
            "window_mask": inputs["window_mask"].to(device, non_blocking=True)}


def seed_everything(settings, device):
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(settings.get("cpu_threads", 2))
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is required by this training protocol")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def config_fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def initialize(config):
    """Initialize the production NCCL group and deterministic per-rank runtime."""
    settings = validate_training_config(config)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    if not distributed():
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    if dist.get_world_size() != settings["world_size"]:
        raise ValueError("production execution requires exactly two ranks")
    seed_everything(settings, device)
    return device


def build_identity(config, variant, provenance, train, validation):
    source = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in Path(__file__).parent.glob("hierarchy_clip*.py")}
    return {"schema": "hierarchy-clip-mini-pretrain-1", "variant": variant,
            "config": config, "world_size": rank_world()[1], "global_batch_size": 64,
            "model_provenance": provenance, "source_sha256": source,
            "manifest_sha256": {"train": train.manifest_sha256, "validation": validation.manifest_sha256},
            "stage_transition_policy": STAGE_TRANSITION_POLICY,
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}


def stage_for_epoch(epoch, connector_epochs=5):
    if epoch < 1:
        raise ValueError("epoch must be one-based")
    return "connectors" if epoch <= connector_epochs else "selective"


def set_epoch_learning_rate(optimizer, epoch, total_epochs=50):
    """One continuous cosine: epoch1 uses base LR; epoch50 uses step49/50.

    This matches stepping CosineAnnealingLR after each completed epoch. Stage
    changes do not restart the schedule or the connector optimizer moments.
    """
    if not 1 <= epoch <= total_epochs:
        raise ValueError("epoch is outside the configured cosine schedule")
    factor = .5 * (1 + math.cos(math.pi * (epoch - 1) / total_epochs))
    rates = {}
    for group in optimizer.param_groups:
        if "base_lr" not in group:
            raise ValueError("optimizer groups must retain their protocol base_lr")
        group["lr"] = group["base_lr"] * factor
        rates[group["name"]] = group["lr"]
    return rates


def setup_stage(model, stage, settings, device, previous_optimizer=None, *, wrap_ddp=True):
    """Configure trainability, retain connector Adam state, then rebuild DDP.

    The caller must release its old DDP wrapper before constructing the new one;
    the underlying module and Parameter identities stay unchanged.
    """
    if isinstance(model, DDP):
        raise ValueError("pass the unwrapped model after releasing old DDP")
    groups = configure_stage(model, stage, learning_rate=settings["learning_rate"],
                             backbone_lr_ratio=settings.get("backbone_lr_ratio", .1),
                             visual_blocks=settings.get("visual_blocks", 2))
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    for group in groups:
        group["base_lr"] = group["lr"]
        group["param_names"] = [names[id(parameter)] for parameter in group["params"]]
    optimizer = torch.optim.AdamW(groups, weight_decay=settings["weight_decay"])
    preserved = []
    if previous_optimizer is not None:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter in previous_optimizer.state:
                    # Reusing Parameter keys is exact and avoids remapping Adam
                    # state by an unstable numeric optimizer parameter index.
                    optimizer.state[parameter] = copy.deepcopy(previous_optimizer.state[parameter])
                    preserved.append(names[id(parameter)])
    device = torch.device(device)
    wrapped = model
    if wrap_ddp:
        if not distributed():
            raise RuntimeError("initialize the process group before constructing DDP")
        wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                      broadcast_buffers=False, find_unused_parameters=False)
    report = {"stage": stage, "policy": STAGE_TRANSITION_POLICY,
              "preserved_adam_parameters": preserved,
              "trainable_names": [name for name, parameter in model.named_parameters() if parameter.requires_grad],
              "optimizer_groups": [{"name": group["name"], "base_lr": group["base_lr"],
                                     "parameters": sum(p.numel() for p in group["params"])}
                                    for group in optimizer.param_groups]}
    return wrapped, optimizer, report


def _raw_samples(samples):
    # Keep B raw sequences separate so dense hierarchy activation microbatches
    # do not require packing/materializing an effective batch of 32 at once.
    return samples


def _stats(logits, labels, loss):
    predicted = logits.detach().topk(min(5, logits.shape[1]), dim=1).indices
    return torch.stack((loss.detach().double(), (predicted[:, 0] == labels).sum().double(),
                        (predicted == labels[:, None]).any(dim=1).sum().double(),
                        loss.new_tensor(len(labels), dtype=torch.float64)))


def _gradient_report(model):
    groups = {}
    flat = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError(f"frozen parameter acquired a gradient: {name}")
            continue
        if parameter.grad is None:
            raise RuntimeError(f"disconnected trainable parameter: {name}")
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"nonfinite gradient: {name}")
        prefix = name.split(".")[0]
        groups[prefix] = groups.get(prefix, 0.) + float(parameter.grad.detach().abs().sum())
        flat.append(parameter.grad.detach().reshape(-1))
    gradient = torch.cat(flat)
    if not bool(gradient.abs().sum() > 0):
        raise RuntimeError("all trainable gradients are zero")
    digest = hashlib.sha256(gradient.float().cpu().numpy().tobytes()).hexdigest()
    digests = gather(digest)
    if len(set(digests)) != 1:
        raise RuntimeError("DDP gradients differ between ranks")
    return {"stage_l1": groups, "gradient_sha256": digest,
            "frozen_parameters_have_no_gradient": True}


def optimizer_update(ddp, samples, optimizer, device, global_size, *, microbatch_size,
                     height=480, width=640, probe=False):
    """One optimizer step at the actual global mean, including uneven tails.

    Every rank performs exactly one synchronized backward. Earlier local
    microbatches use no_sync; summed CE is scaled by world_size/global_size so
    DDP's averaging gives the true per-example global mean.
    """
    if not samples or microbatch_size < 1 or global_size < len(samples):
        raise ValueError("invalid local/global optimizer batch")
    _, world = rank_world()
    model = unwrap(ddp)
    set_staged_train(model)
    optimizer.zero_grad(set_to_none=True)
    values = torch.zeros(4, dtype=torch.float64, device=device)
    slices = [samples[start:start + microbatch_size] for start in range(0, len(samples), microbatch_size)]
    for index, chunk in enumerate(slices):
        batch = sequence_collate(chunk, height=height, width=width)
        inputs = move_inputs(batch["inputs"], device)
        labels = batch["labels"].to(device, non_blocking=True)
        sync = ddp.no_sync() if isinstance(ddp, DDP) and index + 1 < len(slices) else nullcontext()
        with sync:
            with amp_context(device):
                logits = ddp(inputs)
                loss = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite training CE")
            (loss * world / global_size).backward()
        values += _stats(logits, labels, loss)
    trainable = [p for p in model.parameters() if p.requires_grad]
    # No gradient clipping is part of the protocol; infinity only checks finite.
    norm = torch.nn.utils.clip_grad_norm_(trainable, float("inf"), error_if_nonfinite=True)
    gradients = _gradient_report(model) if probe else None
    optimizer.step()
    return {"values": values, "loss_sum": float(values[0]), "top1_count": int(values[1]),
            "top5_count": int(values[2]), "samples": int(values[3]),
            "gradient_norm": float(norm), "gradient_report": gradients,
            "microbatches": len(slices)}


def evaluation_batch(model, samples, device, *, microbatch_size, height=480, width=640):
    """Evaluate the unwrapped model without any DDP forward collectives."""
    model = unwrap(model).eval()
    values = torch.zeros(4, dtype=torch.float64, device=device)
    with torch.no_grad():
        for start in range(0, len(samples), microbatch_size):
            batch = sequence_collate(samples[start:start + microbatch_size], height=height, width=width)
            inputs = move_inputs(batch["inputs"], device)
            labels = batch["labels"].to(device, non_blocking=True)
            with amp_context(device):
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite validation CE")
            values += _stats(logits, labels, loss)
    return values


def run_epoch(ddp, data, settings, device, epoch, optimizer=None, *, probe=False,
              height=480, width=640):
    """One exact-coverage epoch, usable by the bounded production preflight."""
    rank, world = rank_world()
    train = optimizer is not None
    model = unwrap(ddp)
    set_staged_train(model) if train else model.eval()
    sampler = ExactBatchShard(len(data), rank, world, settings["batch_size"], settings["seed"], epoch, train)
    loader = DataLoader(data, batch_sampler=sampler, collate_fn=_raw_samples,
                        num_workers=settings.get("num_workers", 0), worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(settings["seed"] + epoch),
                        pin_memory=torch.device(device).type == "cuda")
    values = torch.zeros(4, dtype=torch.float64, device=device)
    ids, checks = [], []
    if distributed():
        dist.barrier()
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for index, samples in enumerate(loader):
        if any(s["split"] != data.split or s["source_split"] != "train" for s in samples):
            raise ValueError("epoch crossed immutable project/source split")
        if train:
            result = optimizer_update(ddp, samples, optimizer, device, sampler.global_sizes[index],
                                      microbatch_size=settings.get("microbatch_size", settings["batch_size"]),
                                      height=height, width=width, probe=probe)
            values += result["values"]
            if probe:
                checks.append({"step": index + 1, "global_size": sampler.global_sizes[index],
                               "local_size": len(samples), "microbatches": result["microbatches"],
                               **result["gradient_report"]})
            if rank == 0 and (index == 0 or (index + 1) % 100 == 0):
                print(json.dumps({"event": "optimizer_step", "epoch": epoch, "step": index + 1,
                                  "global_batch": sampler.global_sizes[index],
                                  "loss_rank0": result["loss_sum"] / result["samples"]}), flush=True)
        else:
            values += evaluation_batch(model, samples, device,
                                       microbatch_size=settings.get("microbatch_size", settings["batch_size"]),
                                       height=height, width=width)
        ids.extend(sample["sample_id"] for sample in samples)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    seconds = torch.tensor(time.perf_counter() - started, dtype=torch.float64, device=device)
    if distributed():
        dist.all_reduce(values)
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
    shards = gather(ids)
    observed = [sample_id for shard in shards for sample_id in shard]
    expected = [row.sample_id for row in data.rows]
    if len(observed) != len(expected) or len(set(observed)) != len(observed) or set(observed) != set(expected):
        raise RuntimeError("distributed epoch must cover the immutable split exactly once")
    if int(values[3]) != len(expected):
        raise RuntimeError("metric denominator disagrees with exact sample coverage")
    result = {"loss": float(values[0] / values[3]), "top1": float(values[1] / values[3]),
              "top5": float(values[2] / values[3]), "samples": len(expected),
              "seconds": float(seconds), "samples_per_second": len(expected) / float(seconds),
              "rank_sample_counts": [len(shard) for shard in shards], "duplicates": 0, "missing": 0,
              "sample_order_sha256": hashlib.sha256("".join(s + "\n" for s in expected).encode()).hexdigest(),
              "rank_order_sha256": [hashlib.sha256("".join(s + "\n" for s in shard).encode()).hexdigest() for shard in shards],
              "optimizer_steps": len(sampler) if train else 0}
    if probe:
        result["gradient_checks"] = checks
    return result


def validate_training_config(config):
    settings = config["training"]
    required = {"epochs": 50, "batch_size": 32, "world_size": 2, "connector_epochs": 5,
                "learning_rate": 1e-4, "backbone_lr_ratio": .1, "weight_decay": .01,
                "precision": "bfloat16"}
    for key, expected in required.items():
        if settings.get(key) != expected:
            raise ValueError(f"training protocol requires {key}={expected!r}")
    if config.get("num_windows") != 4:
        raise ValueError("training protocol requires four ordered windows")
    if not 1 <= settings.get("microbatch_size", 32) <= 32:
        raise ValueError("microbatch_size must preserve effective local batch32")
    if not config.get("full_training_authorized", False):
        raise ValueError("configuration does not authorize full training")
    return settings


def _memory(device):
    rank, _ = rank_world()
    if torch.device(device).type != "cuda":
        return {"rank": rank, "allocated_bytes": 0, "reserved_bytes": 0}
    return {"rank": rank, "allocated_bytes": torch.cuda.max_memory_allocated(device),
            "reserved_bytes": torch.cuda.max_memory_reserved(device)}


def train(config, variant, output_dir, *, resume=False):
    """Run the authorized 50-epoch protocol inside an initialized two-rank group."""
    from .hierarchy_clip_runtime import build_model

    settings = validate_training_config(config)
    rank, world = rank_world()
    if world != 2 or variant not in VARIANTS:
        raise ValueError("training requires two ranks and a registered comparison")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    seed_everything(settings, device)
    locations = {key: config[key] for key in ("manifest_dir", "dataset_root", "raw_cache")}
    datasets = {split: HierarchySequenceDataset(**locations, split=split, num_windows=4)
                for split in ("train", "validation")}
    if {r.sample_id for r in datasets["train"].rows}.intersection(r.sample_id for r in datasets["validation"].rows):
        raise ValueError("immutable train and validation manifests overlap")
    model, provenance = build_model(variant, config)
    model.to(device)
    identity = build_identity(config, variant, provenance, datasets["train"], datasets["validation"])
    output = Path(output_dir).resolve()
    error = None
    if rank == 0:
        if resume and not (output / "checkpoint_last.pt").is_file():
            error = "resume checkpoint does not exist"
        elif not resume and output.exists() and any(output.iterdir()):
            error = "refusing to overwrite an existing run"
        else:
            output.mkdir(parents=True, exist_ok=True)
    errors = gather(error)
    if any(errors):
        raise FileExistsError(next(item for item in errors if item))
    first, history, best, global_step = 1, [], None, 0
    saved = inspect_training_checkpoint(output / "checkpoint_last.pt", identity=identity) if resume else None
    stage = saved["stage"] if saved is not None else "connectors"
    ddp, optimizer, transition = setup_stage(model, stage, settings, device)
    transitions = [transition]
    if resume:
        saved = load_checkpoint(saved, model, optimizer, identity=identity, device=device)
        first, history, best = saved["epoch"] + 1, saved["history"], saved["best"]
        global_step = saved["global_step"]
        if saved["scheduler"] != {"schedule": "absolute_cosine50", "completed_epoch": first - 1, "total_epochs": 50}:
            raise ValueError("checkpoint cosine schedule identity mismatch")
        if saved["sampler"] != {"seed": settings["seed"], "epoch": first, "offset": 0, "world_size": world}:
            raise ValueError("checkpoint is not an exact epoch-boundary resume")
        transitions = saved.get("extra", {}).get("stage_transitions", transitions)
        del saved
    if rank == 0:
        atomic_json(output / "config.json", identity)
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(first, settings["epochs"] + 1):
        requested_stage = stage_for_epoch(epoch, settings["connector_epochs"])
        if requested_stage != stage:
            del ddp
            ddp, optimizer, transition = setup_stage(model, requested_stage, settings, device, optimizer)
            stage = requested_stage
            transitions.append(transition)
        rates = set_epoch_learning_rate(optimizer, epoch, settings["epochs"])
        training = run_epoch(ddp, datasets["train"], settings, device, epoch, optimizer)
        validation = run_epoch(ddp, datasets["validation"], settings, device, epoch)
        global_step += training["optimizer_steps"]
        improved = best is None or (validation["top1"], -validation["loss"]) > (best["top1"], -best["loss"])
        if improved:
            best = {"epoch": epoch, **validation}
        history.append({"epoch": epoch, "stage": stage, "learning_rates": rates,
                        "train": training, "validation": validation,
                        "gpu_memory_by_rank": gather(_memory(device))})
        artifacts = save_epoch(output, model, optimizer, epoch=epoch, stage=stage, identity=identity,
                   history=history, best=best, improved=improved, device=device,
                   scheduler_state={"schedule": "absolute_cosine50", "completed_epoch": epoch, "total_epochs": 50},
                   global_step=global_step,
                   sampler_state={"seed": settings["seed"], "epoch": epoch + 1, "offset": 0, "world_size": world},
                   extra={"stage_transitions": transitions})
        if rank == 0:
            atomic_json(output / "history.json", history)
            atomic_json(output / "report.json", {"status": "complete" if epoch == 50 else "running",
                        "variant": variant, "completed_epochs": epoch, "best": best,
                        "final": validation, "global_step": global_step,
                        "checkpoint_artifacts": artifacts, "backbone_exported": True,
                        "final_test_accessed": False, "rgb_used": False})
            print(json.dumps({"event": "epoch_complete", "variant": variant, **history[-1]}), flush=True)
    return {"completed_epochs": history[-1]["epoch"], "best": best, "global_step": global_step}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "preflight"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.mode == "preflight":
        if args.resume:
            raise ValueError("preflight does not accept --resume")
        from .hierarchy_clip_preflight import run_preflight
        return run_preflight(config, args.variant, args.output_dir)
    validate_training_config(config)
    from .hierarchy_clip_preflight import assert_training_gate
    assert_training_gate(config, args.config)
    initialize(config)
    try:
        return train(config, args.variant, args.output_dir, resume=args.resume)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
