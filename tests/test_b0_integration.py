from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.nn import functional as F

import ebackbone_v3.b0_integration as integration_module
import ebackbone_v3.n_imagenet_mini_dataset as dataset_module
from ebackbone_v3.b0_integration import (
    INTEGRATION_CHECKPOINT_SCHEMA_VERSION,
    INTEGRATION_MODEL_ROLE,
    INTEGRATION_SCHEMA_VERSION,
    REPORT_KEYS,
    run_b0_train_integration,
    select_fixed_train_subset,
    validate_integration_report,
    verify_integration_checkpoint,
)
from ebackbone_v3.b0_models import (
    B0_CLASS_COUNT,
    PRODUCTION_MODEL_NAME,
    ProductionB0ResNet18,
)
from ebackbone_v3.b0_production import collate_production_b0
from ebackbone_v3.errors import TrainingError
from ebackbone_v3.n_imagenet_mini_dataset import open_dataset


ROOT = Path(__file__).resolve().parents[1]
REAL_MANIFEST_DIR = ROOT / "manifests" / "n_imagenet_mini" / "supervised-v1"
REAL_DATASET_ROOT = Path("/mnt/hdd1/datasets/event/n_imagenet")


def _sample_ids(count: int = 40) -> list[str]:
    return [f"train/n00000000/sample_{index:03d}.npz" for index in range(count)]


def test_fixed_train_subset_selection_is_deterministic_and_label_independent() -> None:
    first = select_fixed_train_subset(_sample_ids(), subset_size=8, seed=20260715)
    second = select_fixed_train_subset(_sample_ids(), subset_size=8, seed=20260715)
    changed_seed = select_fixed_train_subset(_sample_ids(), subset_size=8, seed=20260716)

    assert first == second
    assert first.sample_ids != changed_seed.sample_ids
    assert first.subset_size == 8
    assert len(set(first.sample_ids)) == 8
    assert "labels are not read" in first.selection_rule


def test_integration_dataset_entrypoint_is_hard_coded_train_and_frame_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls: list[dict[str, Any]] = []

    def fake_open_dataset(**kwargs: Any) -> object:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(integration_module, "open_dataset", fake_open_dataset)
    result = integration_module._open_project_train_dataset(  # noqa: SLF001
        manifest_dir="manifests",
        dataset_root="dataset",
    )

    assert result is sentinel
    assert calls == [
        {
            "manifest_dir": "manifests",
            "dataset_root": "dataset",
            "baseline": "b0",
            "cache": "off",
            "split": "train",
        }
    ]
    assert "split" not in inspect.signature(run_b0_train_integration).parameters
    assert "allow_final_test" not in inspect.signature(run_b0_train_integration).parameters


@pytest.mark.local_integration
@pytest.mark.skipif(
    not REAL_DATASET_ROOT.is_dir() or not (REAL_MANIFEST_DIR / "train.jsonl").is_file(),
    reason="local real N-ImageNet mini release is unavailable",
)
def test_real_project_train_item_to_resnet18_loss_is_frame_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_reads: list[str] = []
    archive_source_splits: list[str] = []
    original_read_bytes = dataset_module._read_bytes
    original_read_archive_payload = dataset_module._read_archive_payload

    def recording_read_bytes(path: Path, description: str) -> bytes:
        manifest_reads.append(path.name)
        return original_read_bytes(path, description)

    def recording_read_archive_payload(dataset_root: Path, row: Any) -> bytes:
        archive_source_splits.append(row.source_split)
        return original_read_archive_payload(dataset_root, row)

    monkeypatch.setattr(dataset_module, "_read_bytes", recording_read_bytes)
    monkeypatch.setattr(dataset_module, "_read_archive_payload", recording_read_archive_payload)
    monkeypatch.setattr(
        dataset_module,
        "render_production_representations",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("B0 requested voxel grid or time surface")
        ),
    )
    dataset = open_dataset(
        manifest_dir=REAL_MANIFEST_DIR,
        dataset_root=REAL_DATASET_ROOT,
        split="train",
        baseline="b0",
        cache="off",
    )
    selection = select_fixed_train_subset(dataset.sample_ids, subset_size=8, seed=20260715)
    sample = dataset[selection.source_indices[0]]
    batch = collate_production_b0([sample])
    model = ProductionB0ResNet18().eval()
    with torch.no_grad():
        logits = model(batch["event_frames"])
        loss = F.cross_entropy(logits, batch["labels"])

    assert sample.metadata.sample_id == selection.sample_ids[0]
    assert sample.metadata.split == "train"
    assert sample.metadata.source_split == "train"
    assert set(sample.tensors) == {"event_frame"}
    assert logits.shape == (1, 100)
    assert torch.isfinite(loss)
    assert "train.jsonl" in manifest_reads
    assert "validation.jsonl" not in manifest_reads
    assert "test.jsonl" not in manifest_reads
    assert archive_source_splits == ["train"]


def test_batchnorm_state_and_eval_logits_restore_exactly(tmp_path: Path) -> None:
    torch.manual_seed(31)
    model = ProductionB0ResNet18()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    frames = torch.rand((1, 2, 480, 640), dtype=torch.float32)
    labels = torch.tensor([4], dtype=torch.long)
    before = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key.endswith("running_mean") or key.endswith("running_var")
    }
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(frames), labels)
    loss.backward()
    optimizer.step()
    after = model.state_dict()
    assert any(not torch.equal(value, after[key]) for key, value in before.items())

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema_version": INTEGRATION_CHECKPOINT_SCHEMA_VERSION,
            "model": {
                "name": PRODUCTION_MODEL_NAME,
                "class_count": B0_CLASS_COUNT,
                "role": INTEGRATION_MODEL_ROLE,
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "diagnostic": {},
        },
        checkpoint_path,
    )
    verification = verify_integration_checkpoint(
        checkpoint_path,
        reference_model=model,
        frames=frames,
        labels=labels,
        batch_size=1,
        device=torch.device("cpu"),
        use_amp=False,
    )

    assert verification["strict_load_missing_keys"] == []
    assert verification["strict_load_unexpected_keys"] == []
    assert verification["batchnorm_state_restored_exactly"] is True
    assert verification["batchnorm_state_key_count"] == 60
    assert verification["reloaded_in_evaluation_mode"] is True
    assert verification["logits_exactly_equal"] is True
    assert verification["logits_max_abs_difference"] == 0.0
    assert verification["verified"] is True


def _valid_report() -> dict[str, Any]:
    report = {key: {} for key in REPORT_KEYS}
    report.update(
        {
            "schema_version": INTEGRATION_SCHEMA_VERSION,
            "status": "PASS",
            "mode": "bounded_real_project_train_production_b0_integration",
            "engineering_only": True,
            "subset": {"subset_size": 8},
            "optimization": {"steps_completed": 10, "batch_size": 4},
            "access_audit": {
                "requested_project_split": "train",
                "opened_manifest_filenames": ["train.jsonl"],
                "validation_manifest_opened": False,
                "final_test_manifest_opened": False,
                "validation_or_final_test_archive_member_opened": False,
                "voxel_grid_requested": False,
                "time_surface_requested": False,
                "representations_requested": ["event_frame"],
            },
        }
    )
    return report


def test_bounded_report_schema_rejects_unbounded_or_forbidden_access() -> None:
    validate_integration_report(_valid_report())

    unbounded = _valid_report()
    unbounded["optimization"]["steps_completed"] = 201
    with pytest.raises(TrainingError, match="step count is unbounded"):
        validate_integration_report(unbounded)

    validation_access = _valid_report()
    validation_access["access_audit"]["validation_manifest_opened"] = True
    with pytest.raises(TrainingError, match="forbidden data access"):
        validate_integration_report(validation_access)

    extra_key = _valid_report()
    extra_key["unbounded_extension"] = True
    with pytest.raises(TrainingError, match="keys do not match schema"):
        validate_integration_report(extra_key)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_integration_api_rejects_validation_and_test_split_overrides(
    split: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'split'"):
        run_b0_train_integration(
            manifest_dir="unused",
            dataset_root="unused",
            output_dir=tmp_path / split,
            split=split,  # type: ignore[call-arg]
        )


def test_integration_bounds_fail_before_dataset_access(tmp_path: Path) -> None:
    with pytest.raises(TrainingError, match="max_steps must be between 1 and 200"):
        run_b0_train_integration(
            manifest_dir="unused",
            dataset_root="unused",
            output_dir=tmp_path / "unbounded",
            max_steps=201,
        )
    with pytest.raises(TrainingError, match="batch_size must be between 4 and subset_size"):
        run_b0_train_integration(
            manifest_dir="unused",
            dataset_root="unused",
            output_dir=tmp_path / "small-batch",
            batch_size=3,
        )
