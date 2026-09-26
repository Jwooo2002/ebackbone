"""One supervised runner for all four V1 architectures, with exact epoch resume."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import resource
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .v1_data import RENDERER, RENDERER_SHA256, V1Dataset
from .v1_models import MODEL_NAMES, MODEL_VERSION, V1Backbone, match_frame_width, profile_macs


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 8
    accumulation_steps: int = 4
    learning_rate: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 1e-4
    seed: int = 20260908
    num_workers: int = 4
    device: str = "cuda:1"
    precision: str = "bfloat16"
    cpu_threads: int = 4

    def validate(self):
        for name in ("epochs", "batch_size", "accumulation_steps", "cpu_threads"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.num_workers) is not int or self.num_workers < 0 or type(self.seed) is not int:
            raise ValueError("invalid worker count or seed")
        if not all(math.isfinite(x) for x in (self.learning_rate, self.momentum, self.weight_decay)):
            raise ValueError("optimizer settings must be finite")
        if self.learning_rate <= 0 or not 0 <= self.momentum < 1 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if self.device not in {"cpu", "cuda:1"} or self.precision not in {"float32", "bfloat16"}:
            raise ValueError("use cpu/cuda:1 and float32/bfloat16")
        if self.device == "cpu" and self.precision != "float32":
            raise ValueError("CPU execution requires float32")


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    root = Path(__file__).parent
    return {name: sha256_file(root / name) for name in
            ("v1_models.py", "v1_data.py", "v1_training.py", "n_imagenet_mini_dataset.py",
             "representations.py", "splits.py", "n_imagenet_mini_index.py")}


def seed_everything(config):
    random.seed(config.seed)
    np.random.seed(config.seed % (2 ** 32))
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    if config.device.startswith("cuda"):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("cuda:1 is unavailable; full comparison has not started")
        torch.cuda.set_device(1)
        if config.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("cuda:1 does not support bfloat16")
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def collate(samples):
    keys = set(samples[0]["inputs"])
    if any(set(s["inputs"]) != keys for s in samples):
        raise ValueError("mixed representation availability within a batch")
    return {"inputs": {k: torch.stack([s["inputs"][k] for s in samples]) for k in sorted(keys)},
            "labels": torch.tensor([s["label"] for s in samples], dtype=torch.long),
            "sample_ids": [s["sample_id"] for s in samples],
            "splits": [s["split"] for s in samples],
            "source_splits": [s["source_split"] for s in samples],
            "render_seconds": sum(s["render_seconds"] for s in samples),
            "decode_seconds": sum(s["decode_seconds"] for s in samples)}


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def loader(dataset, config, epoch, train):
    # Separate generator: model initialization and branch count cannot affect order.
    generator = torch.Generator().manual_seed(config.seed + epoch)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=train, drop_last=False,
                      generator=generator, num_workers=config.num_workers, collate_fn=collate,
                      worker_init_fn=seed_worker, pin_memory=config.device.startswith("cuda"))


def autocast(config):
    return torch.autocast("cuda", dtype=torch.bfloat16) if config.precision == "bfloat16" else nullcontext()


def synchronize(config):
    if config.device.startswith("cuda"):
        torch.cuda.synchronize(config.device)


def run_epoch(model, data, config, epoch, *, optimizer=None):
    train = optimizer is not None
    model.train(train)
    batches = loader(data, config, epoch, train)
    loss_sum = correct1 = correct5 = count = 0
    rendering = decoding = forward_seconds = 0.0
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
            "forward_ms_per_sample": 1000 * forward_seconds / count}


def verify_checkpoint(path, model, inputs, config):
    state = torch.load(path, map_location="cpu", weights_only=False)
    devices = [1] if config.device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        restored = V1Backbone(model.name, frame_width=model.frame_width).to(config.device).eval()
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


def run_model(name, config, *, manifest_dir, dataset_root, output_dir, raw_cache=None,
              diagnostic_samples=None, resume=False, stop_after_epoch=None, matched_width=None):
    config.validate()
    if name not in MODEL_NAMES:
        raise ValueError("unknown V1 model")
    if diagnostic_samples is not None and not 1 <= diagnostic_samples <= 16:
        raise ValueError("diagnostics are limited to 1-16 train samples")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= config.epochs:
        raise ValueError("stop_after_epoch outside configured budget")
    seed_everything(config)
    width = (matched_width or match_frame_width()["frame_width"]) if name == "frame_matched" else 32
    model = V1Backbone(name, frame_width=width).to(config.device)
    train = V1Dataset(manifest_dir, dataset_root, "train", multiview=model.multiview,
                      raw_cache=raw_cache, limit=diagnostic_samples, seed=config.seed)
    validation = (train if diagnostic_samples is not None else
                  V1Dataset(manifest_dir, dataset_root, "validation", multiview=model.multiview, raw_cache=raw_cache))
    if diagnostic_samples is None:
        train_ids = {r.sample_id for r in train.rows}
        train_hashes = {r.raw_content_sha256 for r in train.rows}
        if train_ids.intersection(r.sample_id for r in validation.rows):
            raise ValueError("train/validation sample identity overlap")
        # Exact duplicate bytes are reported, not used for relabeling/resplitting.
        duplicate_payloads = len(train_hashes.intersection(r.raw_content_sha256 for r in validation.rows))
    else:
        duplicate_payloads = None
    identity = {"schema": 1, "architecture": name, "model_version": MODEL_VERSION, "frame_width": width,
                "settings": asdict(config), "renderer": RENDERER, "renderer_sha256": RENDERER_SHA256,
                "train_manifest_sha256": train.manifest_sha256,
                "validation_manifest_sha256": validation.manifest_sha256,
                "diagnostic_samples": diagnostic_samples, "source_sha256": source_hashes(),
                "dataset": "N-ImageNet Mini, 100 classes", "torch_version": torch.__version__,
                "numpy_version": np.__version__, "raw_cache": str(Path(raw_cache).resolve()) if raw_cache else None,
                "optimizer": "SGD", "scheduler": "CosineAnnealingLR, eta_min=0, step after each epoch",
                "objective": "fused classifier cross-entropy only", "augmentation": "none"}
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
    inputs = {k: v.unsqueeze(0).to(config.device) for k, v in reference["inputs"].items()}
    verification = verify_checkpoint(output / "checkpoint_last.pt", model, inputs, config)
    report = {"status": "complete" if history[-1]["epoch"] == config.epochs else "partial",
              "evidence_kind": "train-only engineering diagnostic" if diagnostic_samples is not None else "full internal-validation comparison",
              "identity": identity, "completed_epochs": history[-1]["epoch"], "best": best,
              "history": history, "profile": profile_macs(model), "checkpoint_verification": verification,
              "final_test_accessed": False, "train_validation_identical_payload_count": duplicate_payloads,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(config.device) if config.device.startswith("cuda") else None,
              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "process_peak_rss_scope": "process lifetime; not an isolated per-model measurement",
              "device_name": torch.cuda.get_device_name(config.device) if config.device.startswith("cuda") else "CPU",
              "reference_source": reference["source"]}
    atomic_json(output / "report.json", report)
    return report


def compare(config, *, manifest_dir, dataset_root, output_dir, raw_cache=None,
            diagnostic_samples=None, resume=False, stop_after_epoch=None):
    config.validate()
    seed_everything(config)  # Fail unavailable CUDA before opening data or creating run artifacts.
    output = Path(output_dir).resolve()
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite comparison: {output}")
    output.mkdir(parents=True, exist_ok=True)
    matched = match_frame_width()
    reports = {}
    for name in MODEL_NAMES:
        reports[name] = run_model(name, config, manifest_dir=manifest_dir, dataset_root=dataset_root,
                                 output_dir=output / name, raw_cache=raw_cache,
                                 diagnostic_samples=diagnostic_samples, stop_after_epoch=stop_after_epoch,
                                 resume=resume and (output / name / "checkpoint_last.pt").exists(),
                                 matched_width=matched["frame_width"])
        atomic_json(output / "progress.json", {"finished_models": list(reports), "matched": matched})
    reference = reports["frame"]
    for name, report in reports.items():
        for key in ("settings", "train_manifest_sha256", "validation_manifest_sha256", "renderer_sha256"):
            if report["identity"][key] != reference["identity"][key]:
                raise ValueError(f"comparison identity mismatch: {name}, {key}")
        if [r["train"]["sample_order_sha256"] for r in report["history"]] != [r["train"]["sample_order_sha256"] for r in reference["history"]]:
            raise ValueError("models did not see identical ordered training samples")
    result = {"status": "complete" if all(r["status"] == "complete" for r in reports.values()) else "partial",
              "evidence_kind": reference["evidence_kind"], "identical_training_settings_and_order": True,
              "compute_match": matched, "final_test_accessed": False,
              "models": {name: {k: report[k] for k in ("completed_epochs", "best", "profile", "checkpoint_verification")}
                         for name, report in reports.items()}}
    atomic_json(output / "comparison.json", result)
    return result
