"""Bounded real-train integration diagnostic for the production B0 ResNet-18.

This module is an engineering diagnostic, not a training or accuracy entrypoint.
It exposes no split selector: the only dataset request is project ``train`` with
frame-only B0 rendering and caching disabled.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import resource
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ebackbone_v3.b0_models import (
    B0_CLASS_COUNT,
    B0_INPUT_SHAPE,
    PRODUCTION_MODEL_NAME,
    ProductionB0ResNet18,
    build_b0_model,
    trainable_parameter_count,
)
from ebackbone_v3.b0_production import collate_production_b0
from ebackbone_v3.errors import TrainingError
from ebackbone_v3.n_imagenet_mini_dataset import (
    DEFAULT_N_IMAGENET_MINI_DATASET_ROOT,
    ManifestSample,
    open_dataset,
)


INTEGRATION_SCHEMA_VERSION = 1
INTEGRATION_CHECKPOINT_SCHEMA_VERSION = 1
INTEGRATION_MODEL_ROLE = "production_resnet18_engineering_diagnostic"
SUBSET_MIN_SIZE = 8
SUBSET_MAX_SIZE = 16
MIN_BATCH_SIZE = 4
MAX_OPTIMIZER_STEPS = 200
LOSS_REDUCTION_RATIO = 0.25
TARGET_TRAIN_ACCURACY = 0.875
SELECTION_NAMESPACE = b"ebackbone-v3/b0/resnet18-integration/v1\0"
CHECKPOINT_FILENAME = "checkpoint.pt"
REPORT_FILENAME = "report.json"
REPORT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "mode",
        "engineering_only",
        "command",
        "model",
        "input",
        "subset",
        "optimization",
        "trajectory",
        "success_criteria",
        "gradient_checks",
        "component_update_checks",
        "batchnorm",
        "checkpoint",
        "access_audit",
        "device",
    }
)


@dataclass(frozen=True)
class FixedTrainSubset:
    seed: int
    subset_size: int
    sample_ids: tuple[str, ...]
    source_indices: tuple[int, ...]
    selection_rule: str


def select_fixed_train_subset(
    sample_ids: Sequence[str],
    *,
    subset_size: int,
    seed: int,
) -> FixedTrainSubset:
    """Select a deterministic, label-independent subset of project-train IDs."""

    if not SUBSET_MIN_SIZE <= subset_size <= SUBSET_MAX_SIZE:
        raise TrainingError(
            f"subset_size must be between {SUBSET_MIN_SIZE} and {SUBSET_MAX_SIZE}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TrainingError("seed must be an integer")
    if len(sample_ids) < subset_size:
        raise TrainingError("project train manifest has fewer rows than the requested subset")
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise TrainingError("subset selection requires unique non-empty sample IDs")

    seed_bytes = str(seed).encode("ascii")
    ranked = sorted(
        (
            hashlib.sha256(
                SELECTION_NAMESPACE + seed_bytes + b"\0" + sample_id.encode("utf-8")
            ).digest(),
            sample_id,
            index,
        )
        for index, sample_id in enumerate(sample_ids)
    )
    selected = ranked[:subset_size]
    return FixedTrainSubset(
        seed=seed,
        subset_size=subset_size,
        sample_ids=tuple(item[1] for item in selected),
        source_indices=tuple(item[2] for item in selected),
        selection_rule=(
            "lowest SHA-256 ranks of b'ebackbone-v3/b0/resnet18-integration/v1\\0' "
            "+ ASCII(seed) + b'\\0' + UTF-8(project-train sample ID); labels are not read"
        ),
    )


def run_b0_train_integration(
    *,
    manifest_dir: str | Path,
    output_dir: str | Path,
    dataset_root: str | Path = DEFAULT_N_IMAGENET_MINI_DATASET_ROOT,
    subset_size: int = 8,
    batch_size: int = 4,
    max_steps: int = 100,
    evaluation_interval: int = 10,
    learning_rate: float = 0.05,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    seed: int = 20260715,
    device_name: str = "cpu",
) -> dict[str, Any]:
    """Run one bounded, train-only ResNet-18 memorization diagnostic."""

    _validate_arguments(
        subset_size=subset_size,
        batch_size=batch_size,
        max_steps=max_steps,
        evaluation_interval=evaluation_interval,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        seed=seed,
        device_name=device_name,
    )
    output = Path(output_dir).expanduser().resolve()
    _require_new_output_directory(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _run_in_output_directory(
            manifest_dir=manifest_dir,
            output=output,
            dataset_root=dataset_root,
            subset_size=subset_size,
            batch_size=batch_size,
            max_steps=max_steps,
            evaluation_interval=evaluation_interval,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            seed=seed,
            device_name=device_name,
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def _run_in_output_directory(
    *,
    manifest_dir: str | Path,
    output: Path,
    dataset_root: str | Path,
    subset_size: int,
    batch_size: int,
    max_steps: int,
    evaluation_interval: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    device = _resolve_device(device_name)
    _seed_everything(seed, include_cuda=device.type == "cuda")
    dataset = _open_project_train_dataset(
        manifest_dir=manifest_dir,
        dataset_root=dataset_root,
    )
    selection = select_fixed_train_subset(
        dataset.sample_ids,
        subset_size=subset_size,
        seed=seed,
    )
    samples = [dataset[index] for index in selection.source_indices]
    frames, labels, sample_records = _collate_and_audit_samples(samples, selection)
    renderer_fingerprints = {record["renderer_fingerprint"] for record in sample_records}
    if renderer_fingerprints != {dataset.renderer_fingerprint}:
        raise TrainingError("selected samples do not share the dataset renderer fingerprint")
    del samples

    model = build_b0_model().to(device)
    if not isinstance(model, ProductionB0ResNet18):
        raise TrainingError("integration diagnostic did not construct production ResNet-18")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    frames = frames.to(device)
    labels = labels.to(device)
    use_amp = device.type == "cuda"
    component_initial = _component_parameter_snapshot(model)
    batchnorm_initial = _batchnorm_state(model)

    initial_metrics = _evaluate_fixed_subset(
        model,
        frames,
        labels,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
    )
    trajectory = [_trajectory_entry(step=0, metrics=initial_metrics)]
    gradient_checks = {
        "stem": True,
        "residual_body": True,
        "classifier": True,
        "all_steps_finite_and_nonzero": True,
    }
    component_update_checks = {
        "checked_after_step": 1,
        "stem": False,
        "residual_body": False,
        "classifier": False,
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    steps_completed = 0
    for step in range(1, max_steps + 1):
        indices = _cyclic_batch_indices(
            step=step,
            subset_size=subset_size,
            batch_size=batch_size,
            device=device,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            logits = model(frames.index_select(0, indices))
            loss = F.cross_entropy(logits, labels.index_select(0, indices))
        if not bool(torch.isfinite(loss)):
            raise TrainingError(f"non-finite integration loss at step {step}")
        loss.backward()
        step_gradient_checks = _gradient_checks(model)
        for name, passed in step_gradient_checks.items():
            gradient_checks[name] = bool(gradient_checks[name] and passed)
        gradient_checks["all_steps_finite_and_nonzero"] = all(
            gradient_checks[name] for name in ("stem", "residual_body", "classifier")
        )
        optimizer.step()
        steps_completed = step

        if step == 1:
            component_update_checks.update(
                _component_update_checks(model, component_initial)
            )

        should_evaluate = step % evaluation_interval == 0 or step == max_steps
        if should_evaluate:
            metrics = _evaluate_fixed_subset(
                model,
                frames,
                labels,
                batch_size=batch_size,
                device=device,
                use_amp=use_amp,
            )
            trajectory.append(_trajectory_entry(step=step, metrics=metrics))
            if _meets_loss_accuracy_criteria(initial_metrics, metrics):
                break

    final_metrics = trajectory[-1]
    batchnorm_final = _batchnorm_state(model)
    batchnorm_running_keys = tuple(
        key
        for key in batchnorm_initial
        if key.endswith("running_mean") or key.endswith("running_var")
    )
    batchnorm_changed = any(
        not torch.equal(batchnorm_initial[key], batchnorm_final[key])
        for key in batchnorm_running_keys
    )
    checkpoint_path = output / CHECKPOINT_FILENAME
    checkpoint_payload = _checkpoint_payload(
        model=model,
        optimizer=optimizer,
        selection=selection,
        sample_records=sample_records,
        renderer_fingerprint=dataset.renderer_fingerprint,
        seed=seed,
        device_name=device_name,
        use_amp=use_amp,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        batch_size=batch_size,
        max_steps=max_steps,
        steps_completed=steps_completed,
        trajectory=trajectory,
    )
    _atomic_torch_save(checkpoint_path, checkpoint_payload)
    checkpoint_verification = verify_integration_checkpoint(
        checkpoint_path,
        reference_model=model,
        frames=frames,
        labels=labels,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
    )

    criteria = {
        "final_loss_at_most_initial_ratio": LOSS_REDUCTION_RATIO,
        "required_accuracy": TARGET_TRAIN_ACCURACY,
        "initial_loss": initial_metrics["loss"],
        "final_loss": final_metrics["loss"],
        "loss_ratio": final_metrics["loss"] / initial_metrics["loss"],
        "final_accuracy": final_metrics["accuracy"],
        "loss_passed": final_metrics["loss"] <= initial_metrics["loss"] * LOSS_REDUCTION_RATIO,
        "accuracy_passed": final_metrics["accuracy"] >= TARGET_TRAIN_ACCURACY,
    }
    passed = bool(
        criteria["loss_passed"]
        and criteria["accuracy_passed"]
        and gradient_checks["all_steps_finite_and_nonzero"]
        and all(component_update_checks[name] for name in ("stem", "residual_body", "classifier"))
        and batchnorm_changed
        and checkpoint_verification["verified"]
    )
    device_report = _device_report(device, use_amp=use_amp)
    report = {
        "schema_version": INTEGRATION_SCHEMA_VERSION,
        "status": "PASS" if passed else "PARTIAL",
        "mode": "bounded_real_project_train_production_b0_integration",
        "engineering_only": True,
        "command": {
            "name": "diagnose-b0-train",
            "full_epoch": False,
            "maximum_optimizer_steps": MAX_OPTIMIZER_STEPS,
        },
        "model": {
            "name": PRODUCTION_MODEL_NAME,
            "role": INTEGRATION_MODEL_ROLE,
            "trainable_parameter_count": trainable_parameter_count(model),
            "input_shape": list(B0_INPUT_SHAPE),
            "embedding_dimension": 512,
            "classifier": "exactly one Linear(512,100)",
            "random_initialization": True,
        },
        "input": {
            "representation": "production event frame only",
            "shape": [subset_size, *B0_INPUT_SHAPE],
            "storage_dtype": "float32",
            "cache": "off",
            "renderer_fingerprint": dataset.renderer_fingerprint,
        },
        "subset": {
            **asdict(selection),
            "samples": sample_records,
        },
        "optimization": {
            "optimizer": "SGD",
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "steps_completed": steps_completed,
            "evaluation_interval": evaluation_interval,
            "objective": "cross_entropy",
            "scheduler": None,
        },
        "trajectory": trajectory,
        "success_criteria": criteria,
        "gradient_checks": gradient_checks,
        "component_update_checks": component_update_checks,
        "batchnorm": {
            "module_count": sum(isinstance(module, nn.BatchNorm2d) for module in model.modules()),
            "state_key_count": len(batchnorm_final),
            "running_statistics_changed_during_training": batchnorm_changed,
            "checkpoint_state_restored_exactly": checkpoint_verification[
                "batchnorm_state_restored_exactly"
            ],
        },
        "checkpoint": checkpoint_verification,
        "access_audit": {
            "requested_project_split": "train",
            "opened_manifest_filenames": [dataset.manifest_path.name],
            "observed_project_splits": sorted(
                {record["project_split"] for record in sample_records}
            ),
            "observed_source_splits": sorted(
                {record["source_split"] for record in sample_records}
            ),
            "opened_source_locators": [record["source_locator"] for record in sample_records],
            "validation_manifest_opened": False,
            "final_test_manifest_opened": False,
            "validation_or_final_test_archive_member_opened": False,
            "representations_requested": ["event_frame"],
            "voxel_grid_requested": False,
            "time_surface_requested": False,
        },
        "device": device_report,
    }
    validate_integration_report(report)
    _atomic_json_save(output / REPORT_FILENAME, report)
    return report


def verify_integration_checkpoint(
    path: str | Path,
    *,
    reference_model: nn.Module,
    frames: Tensor,
    labels: Tensor,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    """Strict-load a diagnostic checkpoint and verify BatchNorm and eval logits."""

    checkpoint_path = Path(path).expanduser().resolve()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception as exc:
        raise TrainingError(f"could not load B0 integration checkpoint: {exc}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get(
        "schema_version"
    ) != INTEGRATION_CHECKPOINT_SCHEMA_VERSION:
        raise TrainingError("B0 integration checkpoint schema mismatch")
    model_metadata = checkpoint.get("model")
    if model_metadata != {
        "name": PRODUCTION_MODEL_NAME,
        "class_count": B0_CLASS_COUNT,
        "role": INTEGRATION_MODEL_ROLE,
    }:
        raise TrainingError("B0 integration checkpoint model metadata mismatch")

    reloaded = build_b0_model().to(device)
    incompatible = reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
    reference_state = reference_model.state_dict()
    reloaded_state = reloaded.state_dict()
    state_exact = set(reference_state) == set(reloaded_state) and all(
        torch.equal(reference_state[key], reloaded_state[key]) for key in reference_state
    )
    batchnorm_keys = _batchnorm_state_keys(reference_model)
    batchnorm_exact = all(
        torch.equal(reference_state[key], reloaded_state[key]) for key in batchnorm_keys
    )
    reference_metrics = _evaluate_fixed_subset(
        reference_model,
        frames,
        labels,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
    )
    reloaded_metrics = _evaluate_fixed_subset(
        reloaded,
        frames,
        labels,
        batch_size=batch_size,
        device=device,
        use_amp=use_amp,
    )
    reference_logits = reference_metrics["logits"]
    reloaded_logits = reloaded_metrics["logits"]
    max_abs_difference = float((reference_logits - reloaded_logits).abs().max().item())
    logits_exact = bool(torch.equal(reference_logits, reloaded_logits))
    verified = bool(
        not incompatible.missing_keys
        and not incompatible.unexpected_keys
        and state_exact
        and batchnorm_exact
        and logits_exact
        and max_abs_difference == 0.0
        and not reloaded.training
    )
    return {
        "path": str(checkpoint_path),
        "sha256": _sha256_file(checkpoint_path),
        "strict_load_missing_keys": list(incompatible.missing_keys),
        "strict_load_unexpected_keys": list(incompatible.unexpected_keys),
        "model_state_restored_exactly": state_exact,
        "batchnorm_state_key_count": len(batchnorm_keys),
        "batchnorm_state_restored_exactly": batchnorm_exact,
        "reloaded_in_evaluation_mode": not reloaded.training,
        "logits_exactly_equal": logits_exact,
        "logits_max_abs_difference": max_abs_difference,
        "verified": verified,
    }


def validate_integration_report(report: dict[str, Any]) -> None:
    """Fail closed on unbounded or structurally ambiguous diagnostic reports."""

    if set(report) != REPORT_KEYS:
        raise TrainingError("B0 integration report keys do not match schema")
    if report.get("schema_version") != INTEGRATION_SCHEMA_VERSION:
        raise TrainingError("B0 integration report schema version mismatch")
    if report.get("status") not in {"PASS", "PARTIAL"}:
        raise TrainingError("B0 integration report status is invalid")
    if report.get("mode") != "bounded_real_project_train_production_b0_integration":
        raise TrainingError("B0 integration report mode is invalid")
    if report.get("engineering_only") is not True:
        raise TrainingError("B0 integration report must remain engineering-only")
    subset = report.get("subset")
    optimization = report.get("optimization")
    access = report.get("access_audit")
    if not isinstance(subset, dict) or not SUBSET_MIN_SIZE <= int(
        subset.get("subset_size", 0)
    ) <= SUBSET_MAX_SIZE:
        raise TrainingError("B0 integration report subset size is unbounded")
    if not isinstance(optimization, dict) or not 1 <= int(
        optimization.get("steps_completed", 0)
    ) <= MAX_OPTIMIZER_STEPS:
        raise TrainingError("B0 integration report optimizer step count is unbounded")
    if int(optimization.get("batch_size", 0)) < MIN_BATCH_SIZE:
        raise TrainingError("B0 integration report batch size is below the required bound")
    if not isinstance(access, dict) or access.get("requested_project_split") != "train":
        raise TrainingError("B0 integration report did not request project train")
    if access.get("opened_manifest_filenames") != ["train.jsonl"]:
        raise TrainingError("B0 integration report opened a non-train manifest")
    if any(
        access.get(key) is not False
        for key in (
            "validation_manifest_opened",
            "final_test_manifest_opened",
            "validation_or_final_test_archive_member_opened",
            "voxel_grid_requested",
            "time_surface_requested",
        )
    ):
        raise TrainingError("B0 integration report records forbidden data access")
    if access.get("representations_requested") != ["event_frame"]:
        raise TrainingError("B0 integration report is not frame-only")


def _open_project_train_dataset(
    *,
    manifest_dir: str | Path,
    dataset_root: str | Path,
) -> Any:
    return open_dataset(
        manifest_dir=manifest_dir,
        dataset_root=dataset_root,
        baseline="b0",
        cache="off",
        split="train",
    )


def _collate_and_audit_samples(
    samples: Sequence[ManifestSample],
    selection: FixedTrainSubset,
) -> tuple[Tensor, Tensor, list[dict[str, Any]]]:
    if len(samples) != selection.subset_size:
        raise TrainingError("materialized sample count differs from fixed subset size")
    if tuple(sample.metadata.sample_id for sample in samples) != selection.sample_ids:
        raise TrainingError("materialized sample order differs from fixed subset selection")
    if any(set(sample.tensors) != {"event_frame"} for sample in samples):
        raise TrainingError("B0 integration materialized a non-frame representation")
    if any(
        sample.metadata.split != "train" or sample.metadata.source_split != "train"
        for sample in samples
    ):
        raise TrainingError("B0 integration materialized a non-train sample")
    batch = collate_production_b0(samples)
    if set(batch["project_splits"]) != {"train"} or set(batch["source_splits"]) != {
        "train"
    }:
        raise TrainingError("B0 integration collation exposed a non-train split")
    records = [
        {
            "sample_id": sample.metadata.sample_id,
            "class_label": sample.metadata.label,
            "project_split": sample.metadata.split,
            "source_split": sample.metadata.source_split,
            "source_locator": sample.metadata.source_locator,
            "raw_payload_sha256": sample.metadata.raw_payload_sha256,
            "raw_fingerprint": sample.metadata.raw_fingerprint,
            "renderer_fingerprint": sample.metadata.renderer_fingerprint,
            "event_count": sample.metadata.event_count,
        }
        for sample in samples
    ]
    return batch["event_frames"], batch["labels"], records


def _evaluate_fixed_subset(
    model: nn.Module,
    frames: Tensor,
    labels: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> dict[str, Any]:
    model.eval()
    logits_parts: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, labels.numel(), batch_size):
            stop = min(start + batch_size, labels.numel())
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                logits = model(frames[start:stop])
            if logits.shape != (stop - start, B0_CLASS_COUNT):
                raise TrainingError("production ResNet-18 logits violate [B,100]")
            logits_parts.append(logits.float().cpu())
    full_logits = torch.cat(logits_parts, dim=0)
    cpu_labels = labels.cpu()
    loss = F.cross_entropy(full_logits, cpu_labels)
    correct = int((full_logits.argmax(dim=1) == cpu_labels).sum().item())
    return {
        "loss": float(loss.item()),
        "accuracy": correct / int(labels.numel()),
        "correct": correct,
        "sample_count": int(labels.numel()),
        "logits": full_logits,
    }


def _trajectory_entry(*, step: int, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": step,
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "correct": metrics["correct"],
        "sample_count": metrics["sample_count"],
    }


def _meets_loss_accuracy_criteria(
    initial: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    return bool(
        current["loss"] <= initial["loss"] * LOSS_REDUCTION_RATIO
        and current["accuracy"] >= TARGET_TRAIN_ACCURACY
    )


def _cyclic_batch_indices(
    *,
    step: int,
    subset_size: int,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    start = ((step - 1) * batch_size) % subset_size
    return torch.tensor(
        [(start + offset) % subset_size for offset in range(batch_size)],
        dtype=torch.long,
        device=device,
    )


def _gradient_checks(model: ProductionB0ResNet18) -> dict[str, bool]:
    parameters = {
        "stem": model.conv1.weight,
        "residual_body": model.layer4[1].conv2.weight,
        "classifier": model.classifier.weight,
    }
    return {
        name: bool(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and torch.count_nonzero(parameter.grad)
        )
        for name, parameter in parameters.items()
    }


def _component_parameter_snapshot(model: ProductionB0ResNet18) -> dict[str, dict[str, Tensor]]:
    groups = {
        "stem": ("conv1.", "bn1."),
        "residual_body": ("layer1.", "layer2.", "layer3.", "layer4."),
        "classifier": ("classifier.",),
    }
    return {
        group: {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if name.startswith(prefixes)
        }
        for group, prefixes in groups.items()
    }


def _component_update_checks(
    model: ProductionB0ResNet18,
    initial: dict[str, dict[str, Tensor]],
) -> dict[str, bool]:
    current = dict(model.named_parameters())
    return {
        group: any(
            not torch.equal(before, current[name].detach().cpu())
            for name, before in snapshots.items()
        )
        for group, snapshots in initial.items()
    }


def _batchnorm_state_keys(model: nn.Module) -> tuple[str, ...]:
    state = model.state_dict()
    module_names = {
        name for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)
    }
    return tuple(
        key
        for key in state
        if any(key == f"{name}.{suffix}" for name in module_names for suffix in (
            "running_mean",
            "running_var",
            "num_batches_tracked",
        ))
    )


def _batchnorm_state(model: nn.Module) -> dict[str, Tensor]:
    state = model.state_dict()
    return {key: state[key].detach().cpu().clone() for key in _batchnorm_state_keys(model)}


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    selection: FixedTrainSubset,
    sample_records: list[dict[str, Any]],
    renderer_fingerprint: str,
    seed: int,
    device_name: str,
    use_amp: bool,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    batch_size: int,
    max_steps: int,
    steps_completed: int,
    trajectory: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": INTEGRATION_CHECKPOINT_SCHEMA_VERSION,
        "model": {
            "name": PRODUCTION_MODEL_NAME,
            "class_count": B0_CLASS_COUNT,
            "role": INTEGRATION_MODEL_ROLE,
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "diagnostic": {
            "engineering_only": True,
            "project_split": "train",
            "representations": ["event_frame"],
            "selection": asdict(selection),
            "sample_records": sample_records,
            "renderer_fingerprint": renderer_fingerprint,
            "seed": seed,
            "device": device_name,
            "input_storage_dtype": "float32",
            "autocast_dtype": "bfloat16" if use_amp else None,
            "optimizer": "SGD",
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "steps_completed": steps_completed,
            "trajectory": trajectory,
        },
    }


def _validate_arguments(
    *,
    subset_size: int,
    batch_size: int,
    max_steps: int,
    evaluation_interval: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    seed: int,
    device_name: str,
) -> None:
    if not SUBSET_MIN_SIZE <= subset_size <= SUBSET_MAX_SIZE:
        raise TrainingError(
            f"subset_size must be between {SUBSET_MIN_SIZE} and {SUBSET_MAX_SIZE}"
        )
    if not MIN_BATCH_SIZE <= batch_size <= subset_size:
        raise TrainingError(
            f"batch_size must be between {MIN_BATCH_SIZE} and subset_size"
        )
    if not 1 <= max_steps <= MAX_OPTIMIZER_STEPS:
        raise TrainingError(f"max_steps must be between 1 and {MAX_OPTIMIZER_STEPS}")
    if not 1 <= evaluation_interval <= max_steps:
        raise TrainingError("evaluation_interval must be between 1 and max_steps")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise TrainingError("learning_rate must be finite and positive")
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise TrainingError("momentum must be finite and in [0,1)")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise TrainingError("weight_decay must be finite and non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TrainingError("seed must be an integer")
    if device_name not in {"cpu", "cuda:1"}:
        raise TrainingError("device must be 'cpu' or the explicitly permitted 'cuda:1'")


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        raise TrainingError("cuda:1 was requested but is not available")
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    return device


def _seed_everything(seed: int, *, include_cuda: bool) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if include_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _device_report(device: torch.device, *, use_amp: bool) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return {
            "requested": "cuda:1",
            "resolved": str(device),
            "name": torch.cuda.get_device_name(device),
            "input_storage_dtype": "float32",
            "autocast_dtype": "bfloat16" if use_amp else None,
            "peak_memory_metric": "torch.cuda.max_memory_allocated",
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_memory_bytes": int(torch.cuda.max_memory_reserved(device)),
            "cuda_0_used": False,
        }
    return {
        "requested": "cpu",
        "resolved": "cpu",
        "name": None,
        "input_storage_dtype": "float32",
        "autocast_dtype": None,
        "peak_memory_metric": "resource.ru_maxrss including process overhead",
        "peak_memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "peak_reserved_memory_bytes": None,
        "cuda_0_used": False,
    }


def _require_new_output_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise TrainingError(f"output_dir must be new or empty and will not be overwritten: {path}")
    if path.exists():
        path.rmdir()


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise TrainingError(f"checkpoint target already exists: {path}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise TrainingError(f"report target already exists: {path}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FixedTrainSubset",
    "INTEGRATION_SCHEMA_VERSION",
    "MAX_OPTIMIZER_STEPS",
    "REPORT_KEYS",
    "run_b0_train_integration",
    "select_fixed_train_subset",
    "validate_integration_report",
    "verify_integration_checkpoint",
]
