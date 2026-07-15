"""Minimal, bounded B0 training on immutable production event frames.

This module intentionally implements only the first required real-data
validation: deterministic overfitting of 4--16 project-train samples.  It
does not load validation/test rows, alter the manifest-backed adapter, write a
representation cache, augment inputs, or provide a full-dataset training mode.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ebackbone_v3.errors import TrainingError
from ebackbone_v3.b0_models import (
    B0_CLASS_COUNT,
    B0_INPUT_SHAPE,
    CompactDebugB0FrameClassifier,
    DEBUG_MODEL_NAME,
    build_b0_model,
    native_input_batch_bytes,
    trainable_parameter_count,
)
from ebackbone_v3.n_imagenet_mini_dataset import (
    DEFAULT_N_IMAGENET_MINI_DATASET_ROOT,
    ManifestSample,
    open_dataset,
)


B0_TRAINING_SCHEMA_VERSION = 3
TINY_SUBSET_MIN_SIZE = 4
TINY_SUBSET_MAX_SIZE = 16
TINY_LOSS_REDUCTION_RATIO = 0.25
TINY_SELECTION_NAMESPACE = b"ebackbone-v3/b0/tiny-overfit/v1\0"


@dataclass(frozen=True)
class TinySubsetSelection:
    """Label-independent deterministic selection from immutable sample identities."""

    seed: int
    subset_size: int
    sample_ids: tuple[str, ...]
    source_indices: tuple[int, ...]
    selection_rule: str


def select_tiny_subset(sample_ids: Sequence[str], *, subset_size: int, seed: int) -> TinySubsetSelection:
    """Select the lowest SHA-256 ranks without inspecting labels or tensors."""

    if not TINY_SUBSET_MIN_SIZE <= subset_size <= TINY_SUBSET_MAX_SIZE:
        raise TrainingError(
            f"subset_size must be between {TINY_SUBSET_MIN_SIZE} and {TINY_SUBSET_MAX_SIZE}"
        )
    if len(sample_ids) < subset_size:
        raise TrainingError("immutable train manifest has fewer rows than requested tiny subset")
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise TrainingError("tiny subset selection requires unique non-empty stable sample IDs")
    seed_bytes = str(seed).encode("ascii")
    ranked = sorted(
        (
            hashlib.sha256(TINY_SELECTION_NAMESPACE + seed_bytes + b"\0" + sample_id.encode("utf-8")).digest(),
            sample_id,
            index,
        )
        for index, sample_id in enumerate(sample_ids)
    )
    selected = ranked[:subset_size]
    return TinySubsetSelection(
        seed=seed,
        subset_size=subset_size,
        sample_ids=tuple(item[1] for item in selected),
        source_indices=tuple(item[2] for item in selected),
        selection_rule=(
            "lowest SHA-256 ranks of "
            "b'ebackbone-v3/b0/tiny-overfit/v1\\0' + ASCII(seed) + b'\\0' + "
            "UTF-8(stable_sample_id); labels are not read for selection"
        ),
    )


def train_one_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    event_frames: Tensor,
    labels: Tensor,
) -> dict[str, Any]:
    """Run exactly one CE backward/optimizer step and return safety evidence."""

    optimizer.zero_grad(set_to_none=True)
    logits = model(event_frames)
    loss = F.cross_entropy(logits, labels)
    if not bool(torch.isfinite(loss).item()):
        raise TrainingError("cross-entropy loss is non-finite")
    loss.backward()
    gradients_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if not gradients_finite:
        raise TrainingError("a trainable B0 parameter has a missing or non-finite gradient")
    before = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
    optimizer.step()
    update_squared_norm = sum(
        float(torch.sum((parameter.detach() - previous).square()).item())
        for parameter, previous in zip(
            (parameter for parameter in model.parameters() if parameter.requires_grad), before
        )
    )
    return {
        "loss": float(loss.detach().item()),
        "correct": int((logits.detach().argmax(dim=1) == labels).sum().item()),
        "sample_count": int(labels.numel()),
        "gradients_finite": gradients_finite,
        "parameter_updated": update_squared_norm > 0.0,
        "parameter_update_l2": update_squared_norm**0.5,
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: CompactDebugB0FrameClassifier,
    optimizer: torch.optim.Optimizer,
    selection: TinySubsetSelection,
    epochs: Sequence[dict[str, Any]],
) -> Path:
    """Save a self-contained B0 checkpoint; callers must provide a new path."""

    target = Path(path)
    if target.exists():
        raise TrainingError(f"checkpoint path already exists and will not be overwritten: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": B0_TRAINING_SCHEMA_VERSION,
            "model": {"name": DEBUG_MODEL_NAME, "class_count": model.class_count},
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tiny_subset": asdict(selection),
            "epochs": list(epochs),
        },
        target,
    )
    return target


def verify_checkpoint_round_trip(
    path: str | Path,
    *,
    reference_model: CompactDebugB0FrameClassifier,
    reference_inputs: Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Strict-load the saved model and compare deterministic logits exactly."""

    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint.get("schema_version") != B0_TRAINING_SCHEMA_VERSION:
        raise TrainingError("checkpoint schema version does not match B0 training")
    model_config = checkpoint.get("model")
    if (
        not isinstance(model_config, dict)
        or set(model_config) != {"name", "class_count"}
        or model_config["name"] != DEBUG_MODEL_NAME
    ):
        raise TrainingError("checkpoint model configuration is invalid")
    reloaded = build_b0_model(
        str(model_config["name"]), class_count=int(model_config["class_count"])
    ).to(device)
    incompatible = reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
    reference_model.eval()
    reloaded.eval()
    with torch.no_grad():
        reference_logits = reference_model(reference_inputs)
        reloaded_logits = reloaded(reference_inputs)
    max_abs_difference = float((reference_logits - reloaded_logits).abs().max().item())
    exact = bool(torch.equal(reference_logits, reloaded_logits))
    return {
        "checkpoint_path": str(Path(path).resolve()),
        "strict_load_missing_keys": list(incompatible.missing_keys),
        "strict_load_unexpected_keys": list(incompatible.unexpected_keys),
        "logits_exactly_equal": exact,
        "logits_max_abs_difference": max_abs_difference,
        "verified": exact
        and not incompatible.missing_keys
        and not incompatible.unexpected_keys
        and max_abs_difference == 0.0,
    }


def run_tiny_overfit(
    *,
    manifest_dir: str | Path,
    dataset_root: str | Path = DEFAULT_N_IMAGENET_MINI_DATASET_ROOT,
    output_dir: str | Path,
    subset_size: int = 8,
    epochs: int = 120,
    batch_size: int = 4,
    learning_rate: float = 0.01,
    seed: int = 20260715,
    target_train_accuracy: float = 0.95,
) -> dict[str, Any]:
    """Run a deterministic, CPU-only 4--16 sample overfit check."""

    _validate_run_arguments(
        subset_size=subset_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        target_train_accuracy=target_train_accuracy,
    )
    output = Path(output_dir).expanduser().resolve()
    _require_new_or_empty_output_dir(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _run_tiny_overfit_in_output_dir(
            manifest_dir=manifest_dir,
            dataset_root=dataset_root,
            output=output,
            subset_size=subset_size,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            target_train_accuracy=target_train_accuracy,
        )
    except Exception:
        # Do not leave a directory that looks like a completed bounded run.
        shutil.rmtree(output, ignore_errors=True)
        raise


def _run_tiny_overfit_in_output_dir(
    *,
    manifest_dir: str | Path,
    dataset_root: str | Path,
    output: Path,
    subset_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    target_train_accuracy: float,
) -> dict[str, Any]:
    _seed_everything(seed)
    device = torch.device("cpu")
    dataset = open_dataset(
        manifest_dir=manifest_dir,
        dataset_root=dataset_root,
        baseline="b0",
        cache="off",
        split="train",
    )
    # The canonical API validates all rows before exposing this identity list.
    sample_ids = dataset.sample_ids
    selection = select_tiny_subset(sample_ids, subset_size=subset_size, seed=seed)
    frames, labels, first_metadata = _materialize_tiny_subset(dataset, selection)
    loader = DataLoader(
        TensorDataset(frames, labels), batch_size=batch_size, shuffle=False, num_workers=0
    )
    model = CompactDebugB0FrameClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    initial_metrics = _training_metrics(model, loader, device)
    epoch_records: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_sum = 0
        gradients_finite = True
        parameter_updated = False
        update_l2_sum = 0.0
        for frame_batch, label_batch in loader:
            result = train_one_optimizer_step(
                model,
                optimizer,
                frame_batch.to(device, non_blocking=True),
                label_batch.to(device, non_blocking=True),
            )
            loss_sum += result["loss"] * result["sample_count"]
            sample_sum += result["sample_count"]
            gradients_finite = gradients_finite and result["gradients_finite"]
            parameter_updated = parameter_updated or result["parameter_updated"]
            update_l2_sum += result["parameter_update_l2"]
        evaluation = _training_metrics(model, loader, device)
        epoch_records.append(
            {
                "epoch": epoch,
                "loss": evaluation["loss"],
                "optimizer_step_loss": loss_sum / sample_sum,
                "training_accuracy": evaluation["accuracy"],
                "gradients_finite": gradients_finite,
                "parameter_updated": parameter_updated,
                "parameter_update_l2": update_l2_sum,
            }
        )

    checkpoint_path = save_checkpoint(
        output / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        selection=selection,
        epochs=epoch_records,
    )
    verification_inputs = frames[: min(batch_size, subset_size)].to(device)
    checkpoint_verification = verify_checkpoint_round_trip(
        checkpoint_path,
        reference_model=model,
        reference_inputs=verification_inputs,
        device=device,
    )
    final_accuracy = epoch_records[-1]["training_accuracy"]
    final_loss = epoch_records[-1]["loss"]
    loss_reduction_ratio = final_loss / initial_metrics["loss"]
    substantial_loss_reduction = loss_reduction_ratio <= TINY_LOSS_REDUCTION_RATIO
    report = {
        "schema_version": B0_TRAINING_SCHEMA_VERSION,
        "status": (
            "PASS"
            if final_accuracy >= target_train_accuracy and substantial_loss_reduction
            else "PARTIAL"
        ),
        "command": "train-b0-debug",
        "mode": "deterministic_tiny_overfit_only",
        "baseline": "b0",
        "initialization": "random",
        "objective": "cross_entropy",
        "input_contract": {
            "representation": "production event frame",
            "shape": list(B0_INPUT_SHAPE),
            "dtype": "float32",
            "manifest": str(dataset.manifest_path),
            "project_split": "train",
            "representation_cache": "off; no cache entry was read or written",
        },
        "model": {
            "name": DEBUG_MODEL_NAME,
            "role": "engineering_debug_only",
            "architecture": "Conv2d(2,16,7,s4)-ReLU-Conv2d(16,32,3,s2)-ReLU-"
            "Conv2d(32,64,3,s2)-ReLU-Conv2d(64,64,3,s2)-ReLU-GAP-Linear(64,100)",
            "model_side_normalization": "none",
            "trainable_parameter_count": trainable_parameter_count(model),
            "class_count": B0_CLASS_COUNT,
        },
        "device": {
            "type": "cpu",
            "name": None,
            "gpu_peak_memory_bytes": None,
        },
        "tiny_subset": {
            **asdict(selection),
            "materialization": "one in-memory tensor per selected production frame; no persistent cache",
            "first_sample_metadata": first_metadata,
        },
        "optimizer": {"name": "Adam", "learning_rate": learning_rate},
        "target_train_accuracy": target_train_accuracy,
        "loss_reduction": {
            "initial_evaluation_loss": initial_metrics["loss"],
            "final_evaluation_loss": final_loss,
            "final_to_initial_ratio": loss_reduction_ratio,
            "required_maximum_ratio": TINY_LOSS_REDUCTION_RATIO,
            "substantial_reduction_verified": substantial_loss_reduction,
        },
        "bounded_memory": {
            "materialized_subset_bytes": int(frames.numel() * frames.element_size()),
            "native_input_batch_bytes": native_input_batch_bytes(batch_size),
            "batch_size": batch_size,
        },
        "test_isolation": {
            "project_train_samples_opened": subset_size,
            "project_validation_samples_opened": 0,
            "project_test_samples_opened": 0,
        },
        "epochs": epoch_records,
        "checkpoint": checkpoint_verification,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _materialize_tiny_subset(
    dataset: Sequence[ManifestSample],
    selection: TinySubsetSelection,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    frames: list[Tensor] = []
    labels: list[int] = []
    first_metadata: dict[str, Any] | None = None
    for index, expected_sample_id in zip(selection.source_indices, selection.sample_ids):
        sample = dataset[index]
        if sample.metadata.sample_id != expected_sample_id or set(sample.tensors) != {"event_frame"}:
            raise TrainingError("manifest B0 sample identity or frame-only contract changed during loading")
        frame = sample.tensors["event_frame"]
        if frame.shape != B0_INPUT_SHAPE or frame.dtype != np.float32 or not bool(np.isfinite(frame).all()):
            raise TrainingError("production B0 frame does not satisfy [2,480,640] finite float32 contract")
        frames.append(torch.from_numpy(np.array(frame, copy=True)))
        labels.append(sample.metadata.label)
        if first_metadata is None:
            first_metadata = asdict(sample.metadata)
    assert first_metadata is not None
    return torch.stack(frames), torch.tensor(labels, dtype=torch.long), first_metadata


def _training_metrics(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for frame_batch, label_batch in loader:
            labels = label_batch.to(device, non_blocking=True)
            logits = model(frame_batch.to(device, non_blocking=True))
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(label_batch.numel())
    return {"loss": loss_sum / total, "accuracy": correct / total}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _validate_run_arguments(
    *,
    subset_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    target_train_accuracy: float,
) -> None:
    if not TINY_SUBSET_MIN_SIZE <= subset_size <= TINY_SUBSET_MAX_SIZE:
        raise TrainingError(
            f"subset_size must be between {TINY_SUBSET_MIN_SIZE} and {TINY_SUBSET_MAX_SIZE}"
        )
    if epochs <= 0 or batch_size <= 0 or batch_size > subset_size:
        raise TrainingError("epochs and batch_size must be positive and batch_size cannot exceed subset_size")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise TrainingError("learning_rate must be finite and positive")
    if not 0.0 < target_train_accuracy <= 1.0:
        raise TrainingError("target_train_accuracy must be in (0, 1]")


def _require_new_or_empty_output_dir(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise TrainingError(f"output_dir must be new or empty and will not be overwritten: {path}")
    if path.exists():
        path.rmdir()


__all__ = [
    "B0_CLASS_COUNT",
    "B0_INPUT_SHAPE",
    "CompactDebugB0FrameClassifier",
    "TinySubsetSelection",
    "run_tiny_overfit",
    "save_checkpoint",
    "select_tiny_subset",
    "train_one_optimizer_step",
    "verify_checkpoint_round_trip",
]
