"""Production B0 train/validation pipeline over immutable full manifests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ebackbone_v3.b0_models import (
    B0_BOUNDED_CPU_BATCH_SIZE,
    B0_CLASS_COUNT,
    B0_INPUT_SHAPE,
    PRODUCTION_MODEL_NAME,
    build_b0_model,
    trainable_parameter_count,
)
from ebackbone_v3.errors import TrainingError
from ebackbone_v3.n_imagenet_mini_dataset import (
    DEFAULT_N_IMAGENET_MINI_DATASET_ROOT,
    ManifestSample,
    open_dataset,
)
from ebackbone_v3.representations import (
    CACHE_SCHEMA_VERSION,
    CONTRACT_NAME,
    DATASET_RELEASE,
    DEFAULT_RENDERER_CONFIG,
)


PRODUCTION_CHECKPOINT_SCHEMA_VERSION = 3
PRODUCTION_REPORT_SCHEMA_VERSION = 3
CHECKPOINT_LAST_FILENAME = "checkpoint_last.pt"
CHECKPOINT_BEST_FILENAME = "checkpoint_best.pt"
LOG_FILENAME = "metrics.jsonl"
REPORT_FILENAME = "report.json"
CHECKPOINT_SELECTION_RULE = (
    "highest full project-validation top-1; ties use lower validation loss; "
    "remaining ties keep the earlier epoch"
)


@dataclass(frozen=True)
class ManifestIdentity:
    role: str
    path: str
    filename: str
    sha256: str
    size_bytes: int
    sample_count: int


@dataclass
class SplitAudit:
    expected_split: str
    sample_count: int = 0
    first_sample_id: str | None = None
    last_sample_id: str | None = None
    observed_project_splits: set[str] | None = None
    observed_source_splits: set[str] | None = None
    _sample_id_hasher: Any = None

    def __post_init__(self) -> None:
        self.observed_project_splits = set()
        self.observed_source_splits = set()
        self._sample_id_hasher = hashlib.sha256()

    def update(self, batch: dict[str, Any]) -> None:
        sample_ids = batch["sample_ids"]
        project_splits = set(batch["project_splits"])
        source_splits = set(batch["source_splits"])
        if project_splits != {self.expected_split}:
            raise TrainingError(
                f"{self.expected_split} loader yielded project split(s) {sorted(project_splits)}"
            )
        if source_splits != {"train"}:
            raise TrainingError(
                f"{self.expected_split} loader yielded forbidden source split(s) "
                f"{sorted(source_splits)}"
            )
        if self.first_sample_id is None:
            self.first_sample_id = sample_ids[0]
        self.last_sample_id = sample_ids[-1]
        for sample_id in sample_ids:
            self._sample_id_hasher.update(sample_id.encode("utf-8") + b"\n")
        self.sample_count += len(sample_ids)
        assert self.observed_project_splits is not None
        assert self.observed_source_splits is not None
        self.observed_project_splits.update(project_splits)
        self.observed_source_splits.update(source_splits)

    def report(self) -> dict[str, Any]:
        return {
            "expected_project_split": self.expected_split,
            "sample_count": self.sample_count,
            "first_sample_id": self.first_sample_id,
            "last_sample_id": self.last_sample_id,
            "ordered_sample_id_sha256": self._sample_id_hasher.hexdigest(),
            "observed_project_splits": sorted(self.observed_project_splits or ()),
            "observed_source_splits": sorted(self.observed_source_splits or ()),
            "project_test_samples_opened": 0,
        }


def collate_production_b0(samples: Sequence[ManifestSample]) -> dict[str, Any]:
    """Drop raw arrays in worker processes and transfer frame-only batches."""

    if not samples:
        raise TrainingError("production B0 collate received an empty batch")
    frames: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    project_splits: list[str] = []
    source_splits: list[str] = []
    for sample in samples:
        if set(sample.tensors) != {"event_frame"}:
            raise TrainingError("production B0 loader exposed a non-frame representation")
        frame = sample.tensors["event_frame"]
        if (
            frame.shape != B0_INPUT_SHAPE
            or frame.dtype != np.float32
            or not bool(np.isfinite(frame).all())
        ):
            raise TrainingError("production B0 frame violates the finite [2,480,640] float32 contract")
        frames.append(frame)
        labels.append(sample.metadata.label)
        sample_ids.append(sample.metadata.sample_id)
        project_splits.append(sample.metadata.split)
        source_splits.append(sample.metadata.source_split)
    return {
        "event_frames": torch.from_numpy(np.stack(frames, axis=0)),
        "labels": torch.tensor(labels, dtype=torch.long),
        "sample_ids": sample_ids,
        "project_splits": project_splits,
        "source_splits": source_splits,
    }


def topk_correct(logits: Tensor, labels: Tensor, *, ks: Iterable[int] = (1, 5)) -> dict[int, int]:
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise TrainingError("top-k metrics require logits [B,C] and labels [B]")
    requested = tuple(ks)
    if not requested or any(k <= 0 or k > logits.shape[1] for k in requested):
        raise TrainingError("requested top-k metric is invalid for the classifier output")
    predictions = logits.topk(max(requested), dim=1, largest=True, sorted=True).indices
    matches = predictions.eq(labels.view(-1, 1))
    return {k: int(matches[:, :k].any(dim=1).sum().item()) for k in requested}


def run_production_b0(
    *,
    manifest_dir: str | Path,
    dataset_root: str | Path = DEFAULT_N_IMAGENET_MINI_DATASET_ROOT,
    output_dir: str | Path,
    epochs: int,
    batch_size: int = B0_BOUNDED_CPU_BATCH_SIZE,
    learning_rate: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    seed: int = 20260715,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    amp: bool = True,
    device_name: str = "cpu",
    stop_after_epoch: int | None = None,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    """Train random-initialized production B0 and select on full validation only."""

    _validate_arguments(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        device_name=device_name,
        stop_after_epoch=stop_after_epoch,
    )
    output = Path(output_dir).expanduser().resolve()
    resume_path = Path(resume).expanduser().resolve() if resume is not None else None
    device = _resolve_device(device_name)
    _prepare_output_directory(output, resume_path=resume_path)
    _seed_everything(seed, include_cuda=device.type == "cuda")
    use_amp = bool(amp and device.type == "cuda")

    # Roles are explicit at the canonical boundary; no filename selects a split.
    train_dataset = open_dataset(
        manifest_dir=manifest_dir,
        dataset_root=dataset_root,
        baseline="b0",
        cache="off",
        split="train",
    )
    validation_dataset = open_dataset(
        manifest_dir=manifest_dir,
        dataset_root=dataset_root,
        baseline="b0",
        cache="off",
        split="validation",
    )
    if train_dataset.renderer_fingerprint != validation_dataset.renderer_fingerprint:
        raise TrainingError("train and validation renderer provenance differs")

    train_manifest = _manifest_identity(train_dataset.manifest_path, "train", len(train_dataset))
    validation_manifest = _manifest_identity(
        validation_dataset.manifest_path, "validation", len(validation_dataset)
    )
    run_config = _run_config(
        train_manifest=train_manifest,
        validation_manifest=validation_manifest,
        dataset_root=dataset_root,
        renderer_fingerprint=train_dataset.renderer_fingerprint,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        seed=seed,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        amp=use_amp,
        device=device,
    )

    model = build_b0_model(class_count=B0_CLASS_COUNT).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.0
    )
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    start_epoch = 1
    resume_report: dict[str, Any] | None = None
    if resume_path is not None:
        checkpoint = _load_checkpoint(resume_path, device=device)
        _validate_resume_checkpoint(checkpoint, run_config=run_config)
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise TrainingError("strict production model resume reported incompatible keys")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        history = list(checkpoint["history"])
        best = checkpoint["best"]
        loaded_epoch = int(checkpoint["epoch"])
        start_epoch = loaded_epoch + 1
        _restore_rng_state(checkpoint["rng_state"])
        resume_report = {
            "checkpoint_path": str(resume_path),
            "loaded_epoch": loaded_epoch,
            "continued_from_epoch": start_epoch,
            "strict_load_missing_keys": list(incompatible.missing_keys),
            "strict_load_unexpected_keys": list(incompatible.unexpected_keys),
            "optimizer_state_loaded": True,
            "scheduler_state_loaded": True,
        }

    final_epoch = epochs if stop_after_epoch is None else stop_after_epoch
    if start_epoch > final_epoch:
        raise TrainingError(
            f"resume checkpoint epoch {start_epoch - 1} leaves no epoch to run through {final_epoch}"
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    log_path = output / LOG_FILENAME
    log_mode = "a" if resume_path is not None else "x"
    with log_path.open(log_mode, encoding="utf-8", buffering=1) as log_handle:
        _write_log(
            log_handle,
            {
                "event": "resume" if resume_path is not None else "run_start",
                "start_epoch": start_epoch,
                "final_epoch_this_command": final_epoch,
                "run_config_sha256": _canonical_sha256(run_config),
            },
        )
        for epoch in range(start_epoch, final_epoch + 1):
            train_loader = _make_loader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=device.type == "cuda",
                seed=seed + epoch,
            )
            validation_loader = _make_loader(
                validation_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=device.type == "cuda",
                seed=seed + 1_000_000 + epoch,
            )
            learning_rate_this_epoch = float(optimizer.param_groups[0]["lr"])
            train_metrics = _run_train_epoch(
                model=model,
                optimizer=optimizer,
                loader=train_loader,
                device=device,
                use_amp=use_amp,
                epoch=epoch,
                expected_sample_count=len(train_dataset),
                log_handle=log_handle,
            )
            validation_metrics = _run_validation_epoch(
                model=model,
                loader=validation_loader,
                device=device,
                use_amp=use_amp,
                epoch=epoch,
                expected_sample_count=len(validation_dataset),
            )
            scheduler.step()
            epoch_record = {
                "epoch": epoch,
                "learning_rate": learning_rate_this_epoch,
                "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(epoch_record)
            is_best = _is_better(validation_metrics, best)
            if is_best:
                best = {
                    "epoch": epoch,
                    "validation_top1_accuracy": validation_metrics["top1_accuracy"],
                    "validation_loss": validation_metrics["loss"],
                    "selection_rule": CHECKPOINT_SELECTION_RULE,
                }
            checkpoint_payload = _checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                run_config=run_config,
                history=history,
                best=best,
            )
            _atomic_torch_save(output / CHECKPOINT_LAST_FILENAME, checkpoint_payload)
            if is_best:
                _atomic_torch_save(output / CHECKPOINT_BEST_FILENAME, checkpoint_payload)
            _write_log(log_handle, {"event": "epoch", **epoch_record, "is_best": is_best})

    strict_reload = {
        "last": verify_production_checkpoint(output / CHECKPOINT_LAST_FILENAME),
        "best": verify_production_checkpoint(output / CHECKPOINT_BEST_FILENAME),
    }
    completed = final_epoch == epochs
    report = {
        "schema_version": PRODUCTION_REPORT_SCHEMA_VERSION,
        "status": "PASS" if completed else "PARTIAL",
        "mode": "production_full_manifest_b0",
        "completed_requested_epochs": completed,
        "run_config": run_config,
        "model": {
            "name": PRODUCTION_MODEL_NAME,
            "architecture": (
                "ResNet-18 basic blocks [2,2,2,2]; Conv2d(2,64,7,s2,p3,bias=False) "
                "stem; BatchNorm2d; MaxPool2d(3,s2,p1); AdaptiveAvgPool2d(1); "
                "Linear(512,100)"
            ),
            "initialization": "random Kaiming-normal convolution initialization",
            "model_side_normalization": "standard ResNet-18 BatchNorm2d layers",
            "global_pooling": "adaptive average pooling to 1x1",
            "embedding_dimension": 512,
            "classifier": "exactly one Linear(512,100)",
            "trainable_parameter_count": trainable_parameter_count(model),
        },
        "history": history,
        "best": best,
        "resume": resume_report,
        "checkpoints": strict_reload,
        "checkpoint_paths": {
            "last": str((output / CHECKPOINT_LAST_FILENAME).resolve()),
            "best": str((output / CHECKPOINT_BEST_FILENAME).resolve()),
        },
        "jsonl_log": str(log_path.resolve()),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "test_isolation": {
            "test_manifest_argument_supported": False,
            "test_manifest_rejected_before_dataset_construction": True,
            "accepted_project_splits": ["train", "validation"],
            "accepted_source_splits": ["train"],
            "project_test_samples_opened": 0,
        },
    }
    _atomic_json_write(output / REPORT_FILENAME, report)
    return report


def verify_production_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint = _load_checkpoint(checkpoint_path, device=torch.device("cpu"))
    model_config = checkpoint.get("model")
    if not isinstance(model_config, dict):
        raise TrainingError("checkpoint model metadata is missing")
    model = build_b0_model(
        str(model_config.get("name")), class_count=int(model_config.get("class_count", 0))
    )
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loaded_state = model.state_dict()
    exact = all(
        torch.equal(loaded_state[key].cpu(), value.cpu())
        for key, value in checkpoint["model_state_dict"].items()
    )
    return {
        "path": str(checkpoint_path),
        "exists": checkpoint_path.is_file(),
        "sha256": _sha256_file(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "strict_load_missing_keys": list(incompatible.missing_keys),
        "strict_load_unexpected_keys": list(incompatible.unexpected_keys),
        "state_tensors_exactly_equal": exact,
        "verified": exact and not incompatible.missing_keys and not incompatible.unexpected_keys,
    }


def _make_loader(
    dataset: Sequence[ManifestSample],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "collate_fn": collate_production_b0,
        "generator": generator,
        "worker_init_fn": _seed_worker,
        "persistent_workers": False,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def _run_train_epoch(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader[Any],
    device: torch.device,
    use_amp: bool,
    epoch: int,
    expected_sample_count: int,
    log_handle: Any,
) -> dict[str, Any]:
    model.train()
    audit = SplitAudit("train")
    loss_sum = 0.0
    top1_sum = 0
    top5_sum = 0
    first_loss: float | None = None
    last_loss: float | None = None
    minimum_loss = math.inf
    maximum_loss = -math.inf
    batch_count = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, start=1):
        audit.update(batch)
        frames = batch["event_frames"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            enabled=use_amp,
        ):
            logits = model(frames)
            loss = F.cross_entropy(logits, labels)
        if not bool(torch.isfinite(loss).item()):
            raise TrainingError(f"non-finite train loss at epoch {epoch}, batch {batch_index}")
        loss.backward()
        if not all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        ):
            raise TrainingError(f"non-finite gradient at epoch {epoch}, batch {batch_index}")
        optimizer.step()
        batch_loss = float(loss.detach().item())
        correct = topk_correct(logits.detach(), labels)
        sample_count = int(labels.numel())
        loss_sum += batch_loss * sample_count
        top1_sum += correct[1]
        top5_sum += correct[5]
        first_loss = batch_loss if first_loss is None else first_loss
        last_loss = batch_loss
        minimum_loss = min(minimum_loss, batch_loss)
        maximum_loss = max(maximum_loss, batch_loss)
        batch_count = batch_index
        _write_log(
            log_handle,
            {
                "event": "train_batch",
                "epoch": epoch,
                "batch": batch_index,
                "sample_count": sample_count,
                "loss": batch_loss,
                "top1_correct": correct[1],
                "top5_correct": correct[5],
                "first_sample_id": batch["sample_ids"][0],
                "last_sample_id": batch["sample_ids"][-1],
            },
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    _require_full_split(audit, expected_sample_count)
    return {
        "loss": loss_sum / audit.sample_count,
        "top1_accuracy": top1_sum / audit.sample_count,
        "top5_accuracy": top5_sum / audit.sample_count,
        "batch_count": batch_count,
        "batch_loss_first": first_loss,
        "batch_loss_last": last_loss,
        "batch_loss_min": minimum_loss,
        "batch_loss_max": maximum_loss,
        "batch_loss_changed": bool(first_loss != last_loss or minimum_loss != maximum_loss),
        "elapsed_seconds": elapsed,
        "throughput_samples_per_second": audit.sample_count / elapsed,
        "split_audit": audit.report(),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }


def _run_validation_epoch(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    use_amp: bool,
    epoch: int,
    expected_sample_count: int,
) -> dict[str, Any]:
    model.eval()
    audit = SplitAudit("validation")
    loss_sum = 0.0
    top1_sum = 0
    top5_sum = 0
    batch_count = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            audit.update(batch)
            frames = batch["event_frames"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                enabled=use_amp,
            ):
                logits = model(frames)
                loss = F.cross_entropy(logits, labels)
            if not bool(torch.isfinite(loss).item()):
                raise TrainingError(
                    f"non-finite validation loss at epoch {epoch}, batch {batch_index}"
                )
            correct = topk_correct(logits, labels)
            sample_count = int(labels.numel())
            loss_sum += float(loss.item()) * sample_count
            top1_sum += correct[1]
            top5_sum += correct[5]
            batch_count = batch_index
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    _require_full_split(audit, expected_sample_count)
    return {
        "loss": loss_sum / audit.sample_count,
        "top1_accuracy": top1_sum / audit.sample_count,
        "top5_accuracy": top5_sum / audit.sample_count,
        "batch_count": batch_count,
        "elapsed_seconds": elapsed,
        "throughput_samples_per_second": audit.sample_count / elapsed,
        "split_audit": audit.report(),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }


def _require_full_split(audit: SplitAudit, expected_sample_count: int) -> None:
    if audit.sample_count != expected_sample_count:
        raise TrainingError(
            f"{audit.expected_split} epoch processed {audit.sample_count} samples; "
            f"expected full manifest count {expected_sample_count}"
        )


def _is_better(validation: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    top1 = float(validation["top1_accuracy"])
    best_top1 = float(best["validation_top1_accuracy"])
    return top1 > best_top1 or (
        top1 == best_top1 and float(validation["loss"]) < float(best["validation_loss"])
    )


def _checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    run_config: dict[str, Any],
    history: Sequence[dict[str, Any]],
    best: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": PRODUCTION_CHECKPOINT_SCHEMA_VERSION,
        "epoch": epoch,
        "model": {
            "name": PRODUCTION_MODEL_NAME,
            "class_count": B0_CLASS_COUNT,
            "initialization": "random",
            "global_pooling": "AdaptiveAvgPool2d((1,1))",
            "embedding_dimension": 512,
            "classifier": "Linear(512,100)",
        },
        "model_state_dict": model.state_dict(),
        "optimizer": run_config["optimizer"],
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler": run_config["scheduler"],
        "scheduler_state_dict": scheduler.state_dict(),
        "seed": run_config["seed"],
        "manifests": run_config["manifests"],
        "renderer_provenance": run_config["renderer_provenance"],
        "run_config": run_config,
        "run_config_sha256": _canonical_sha256(run_config),
        "history": list(history),
        "best": best,
        "rng_state": _rng_state(),
    }


def _run_config(
    *,
    train_manifest: ManifestIdentity,
    validation_manifest: ManifestIdentity,
    dataset_root: str | Path,
    renderer_fingerprint: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    seed: int,
    num_workers: int,
    prefetch_factor: int,
    amp: bool,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "baseline": "b0",
        "input": {
            "representation": "production event frame only",
            "shape": list(B0_INPUT_SHAPE),
            "dtype": "float32",
            "cache": "off; no representation cache read or write",
            "augmentation": "none",
        },
        "model": {
            "name": PRODUCTION_MODEL_NAME,
            "class_count": B0_CLASS_COUNT,
            "initialization": "random",
            "model_side_normalization": "standard ResNet-18 BatchNorm2d layers",
            "embedding_dimension": 512,
        },
        "objective": "cross_entropy",
        "optimizer": {
            "name": "SGD",
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
        },
        "scheduler": {
            "name": "CosineAnnealingLR",
            "T_max": epochs,
            "eta_min": 0.0,
            "step_unit": "epoch",
        },
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "deterministic": {
            "python_numpy_torch_seeded": True,
            "epoch_shuffle_seed": "seed + one-based epoch",
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
        "loader": {
            "implementation": "canonical open_dataset(manifest_dir=..., split=...) adapter",
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "drop_last": False,
        },
        "manifests": {
            "train": asdict(train_manifest),
            "validation": asdict(validation_manifest),
            "checkpoint_selection": CHECKPOINT_SELECTION_RULE,
            "test_manifest": "forbidden and not accepted by this command",
        },
        "renderer_provenance": {
            "dataset_release": DATASET_RELEASE,
            "representation_contract": CONTRACT_NAME,
            "representation_cache_schema_version": CACHE_SCHEMA_VERSION,
            "renderer_config": DEFAULT_RENDERER_CONFIG.to_dict(),
            "renderer_fingerprint": renderer_fingerprint,
        },
        "device": {
            "type": device.type,
            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "amp": "bfloat16 autocast" if amp else "off",
        },
    }


def _validate_resume_checkpoint(checkpoint: dict[str, Any], *, run_config: dict[str, Any]) -> None:
    expected_hash = _canonical_sha256(run_config)
    if checkpoint.get("run_config_sha256") != expected_hash:
        raise TrainingError(
            "resume checkpoint run configuration differs from the current model, optimizer, "
            "scheduler, seed, manifests, renderer, or loader configuration"
        )
    if checkpoint.get("run_config") != run_config:
        raise TrainingError("resume checkpoint run configuration payload is not exact")


def _load_checkpoint(path: Path, *, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise TrainingError(f"checkpoint does not exist or is not a regular file: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise TrainingError(f"could not load production B0 checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise TrainingError("production checkpoint must contain a mapping")
    if checkpoint.get("schema_version") != PRODUCTION_CHECKPOINT_SCHEMA_VERSION:
        raise TrainingError("production checkpoint schema version mismatch")
    return checkpoint


def _manifest_identity(path: Path, role: str, sample_count: int) -> ManifestIdentity:
    return ManifestIdentity(
        role=role,
        path=str(path),
        filename=path.name,
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
        sample_count=sample_count,
    )


def _prepare_output_directory(output: Path, *, resume_path: Path | None) -> None:
    if resume_path is None:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise TrainingError(f"output_dir must be new or empty: {output}")
        if output.exists():
            output.rmdir()
        output.mkdir(parents=True, exist_ok=False)
        return
    if not output.is_dir():
        raise TrainingError("resume requires the existing output_dir")
    if resume_path.parent != output:
        raise TrainingError("resume checkpoint must be inside the selected output_dir")
    if resume_path.name != CHECKPOINT_LAST_FILENAME:
        raise TrainingError("resume must use checkpoint_last.pt, not the selected best checkpoint")
    required = {CHECKPOINT_LAST_FILENAME, CHECKPOINT_BEST_FILENAME, LOG_FILENAME}
    missing = sorted(name for name in required if not (output / name).is_file())
    if missing:
        raise TrainingError(f"resume output_dir is missing required artifact(s): {missing}")


def _validate_arguments(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    num_workers: int,
    prefetch_factor: int,
    device_name: str,
    stop_after_epoch: int | None,
) -> None:
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TrainingError(f"{name} must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise TrainingError("learning_rate must be finite and positive")
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise TrainingError("momentum must be finite and in [0,1)")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise TrainingError("weight_decay must be finite and non-negative")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise TrainingError("num_workers must be a non-negative integer")
    if isinstance(prefetch_factor, bool) or not isinstance(prefetch_factor, int) or prefetch_factor <= 0:
        raise TrainingError("prefetch_factor must be a positive integer")
    if device_name not in {"cpu", "cuda"}:
        raise TrainingError("device_name must be 'cpu' or 'cuda'")
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= epochs:
        raise TrainingError("stop_after_epoch must be between one and epochs")


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA was explicitly requested but is not available")
    return torch.device(device_name)


def _seed_everything(seed: int, *, include_cuda: bool) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if include_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.set_num_threads(1)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": _numpy_rng_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _numpy_rng_state() -> dict[str, Any]:
    state = np.random.get_state()
    return {
        "bit_generator": state[0],
        "keys": state[1].tolist(),
        "position": state[2],
        "has_gauss": state[3],
        "cached_gaussian": state[4],
    }


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_log(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "CHECKPOINT_BEST_FILENAME",
    "CHECKPOINT_LAST_FILENAME",
    "LOG_FILENAME",
    "collate_production_b0",
    "run_production_b0",
    "topk_correct",
    "verify_production_checkpoint",
]
