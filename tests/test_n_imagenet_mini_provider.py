from __future__ import annotations

import io
import json
import math
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ebackbone_v3.errors import ProbeError
from ebackbone_v3.probe import run_probe
from ebackbone_v3.providers.n_imagenet_mini import (
    EXPECTED_EVENT_DTYPE,
    TIME_SURFACE_TAU,
    load_sample,
    render_probe_representations,
)


def _events() -> np.ndarray:
    events = np.empty(6, dtype=EXPECTED_EVENT_DTYPE)
    events["x"] = [0, 0, 0, 1, 2, 2]
    events["y"] = [0, 0, 0, 1, 1, 1]
    events["t"] = [0, 1, 1, 2, 3, 3]
    events["p"] = [False, False, False, True, True, False]
    return events


def _fields(events: np.ndarray | None = None) -> dict[str, np.ndarray]:
    source = _events() if events is None else events
    return {name: source[name].copy() for name in ("x", "y", "t", "p")}


def _npz_payload(events: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(output, event_data=events)
    return output.getvalue()


def _class_tar_payload(class_id: str, filename: str, events: np.ndarray) -> bytes:
    npz_payload = _npz_payload(events)
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as handle:
        info = tarfile.TarInfo(name=f"{class_id}/{filename}")
        info.size = len(npz_payload)
        info.mtime = 0
        handle.addfile(info, io.BytesIO(npz_payload))
    return output.getvalue()


def _write_archive_fixture(dataset_root: Path, *, validation_class_count: int = 100) -> None:
    archive_dir = dataset_root / "mini_zenodo" / "archives"
    archive_dir.mkdir(parents=True)
    events = _events()
    class_ids = [f"n{index:08d}" for index in range(100)]
    for part_index in range(1, 11):
        zip_path = archive_dir / f"train_Part_{part_index}.zip"
        part_classes = class_ids[(part_index - 1) * 10 : part_index * 10]
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_STORED) as handle:
            for class_id in part_classes:
                handle.writestr(
                    f"Part_{part_index}/{class_id}.tar.gz",
                    _class_tar_payload(class_id, "sample_train.npz", events),
                )
    with zipfile.ZipFile(
        archive_dir / "mini_validation_split.zip",
        mode="w",
        compression=zipfile.ZIP_STORED,
    ) as handle:
        payload = _npz_payload(events)
        for class_id in class_ids[:validation_class_count]:
            handle.writestr(
                f"extracted_val/{class_id}/sample_validation.npz",
                payload,
            )


def test_renderer_uses_every_event_and_explicit_latest_timestamp_max() -> None:
    fields = _fields()
    rendered = render_probe_representations(
        fields,
        spatial_resolution=(2, 3),
        voxel_bins=4,
    )

    frame = rendered["event_frame"]
    voxel = rendered["voxel_grid"]
    surface = rendered["time_surface"]
    assert frame.shape == (2, 2, 3)
    assert voxel.shape == (4, 2, 3)
    assert surface.shape == (2, 2, 3)
    assert frame.dtype == voxel.dtype == surface.dtype == np.float32
    assert float(frame.sum()) == len(fields["t"])
    assert float(voxel.sum()) == len(fields["t"])
    assert frame[0, 0, 0] == 3
    assert frame[1, 1, 1] == 1
    assert voxel[0, 0, 0] == 1
    assert voxel[1, 0, 0] == 2
    assert voxel[2, 1, 1] == 1
    assert voxel[3, 1, 2] == 2

    # Three duplicate negative events hit (x=0,y=0); max normalized t is 1/3.
    expected_duplicate_value = math.exp(-(1.0 - (1.0 / 3.0)) / TIME_SURFACE_TAU)
    assert surface[0, 0, 0] == pytest.approx(expected_duplicate_value)
    assert surface[1, 1, 2] == pytest.approx(1.0)
    assert surface[0, 0, 2] == pytest.approx(math.exp(-1.0 / TIME_SURFACE_TAU))


def test_renderer_is_deterministic_and_has_no_class_input() -> None:
    fields = _fields()
    first = render_probe_representations(fields, spatial_resolution=(2, 3), voxel_bins=4)
    second = render_probe_representations(fields, spatial_resolution=(2, 3), voxel_bins=4)
    assert set(first) == {"event_frame", "voxel_grid", "time_surface"}
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_renderer_rejects_invalid_coordinates_instead_of_filtering() -> None:
    fields = _fields()
    fields["x"][0] = 3
    with pytest.raises(ProbeError, match="outside zero-based"):
        render_probe_representations(fields, spatial_resolution=(2, 3), voxel_bins=4)


def test_renderer_rejects_unsorted_timestamps_and_changed_polarity_dtype() -> None:
    unsorted = _fields()
    unsorted["t"] = np.asarray([0, 2, 1, 2, 3, 3], dtype=np.uint16)
    with pytest.raises(ProbeError, match="not nondecreasing"):
        render_probe_representations(unsorted, spatial_resolution=(2, 3), voxel_bins=4)

    changed_polarity = _fields()
    changed_polarity["p"] = changed_polarity["p"].astype(np.uint8)
    with pytest.raises(ProbeError, match="field p must have dtype bool"):
        render_probe_representations(
            changed_polarity,
            spatial_resolution=(2, 3),
            voxel_bins=4,
        )


def test_renderer_requires_at_least_two_temporal_bins() -> None:
    with pytest.raises(ProbeError, match=">= 2"):
        render_probe_representations(_fields(), spatial_resolution=(2, 3), voxel_bins=1)


def test_archive_provider_preserves_raw_contract_and_keeps_labels_out_of_rendering(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "n_imagenet"
    _write_archive_fixture(dataset_root)
    train = load_sample(
        dataset_root=dataset_root,
        split="train",
        sample_id="train/n00000000/sample_train.npz",
        voxel_bins=4,
    )
    validation = load_sample(
        dataset_root=dataset_root,
        split="validation",
        sample_id="validation/n00000099/sample_validation.npz",
        voxel_bins=4,
    )

    assert train.split == "train"
    assert validation.split == "validation"
    assert train.raw_events.event_count_before_filter == 6
    assert train.raw_events.event_count_after_filter == 6
    assert train.raw_events.temporal_start == 0
    assert train.raw_events.temporal_end == 3
    assert train.raw_events.interval_closure == "[]"
    assert train.raw_events.timestamp_unit == "microsecond"
    assert train.raw_events.timestamp_ordering == "nondecreasing"
    assert train.raw_events.polarity_encoding == "false/true"
    assert train.raw_events.spatial_resolution == (480, 640)
    assert train.raw_events.fields["x"].dtype == np.dtype("uint16")
    assert train.raw_events.fields["t"].dtype == np.dtype("uint16")
    assert train.raw_events.fields["p"].dtype == np.dtype("bool")
    assert train.classification is not None
    assert validation.classification is not None
    assert train.classification.class_id == "n00000000"
    assert train.classification.class_index == 0
    assert validation.classification.class_id == "n00000099"
    assert validation.classification.class_index == 99
    assert train.classification.class_count == 100

    # The two archive classes carry identical event arrays in this contract
    # fixture. Their render tensors and fingerprints must therefore be equal.
    for name in train.representations:
        np.testing.assert_array_equal(
            train.representations[name].tensor,
            validation.representations[name].tensor,
        )
        assert (
            train.representations[name].event_subset_id
            == validation.representations[name].event_subset_id
        )


def test_real_probe_contract_accepts_archive_provider_mapping(tmp_path: Path) -> None:
    dataset_root = tmp_path / "n_imagenet"
    _write_archive_fixture(dataset_root)
    config = tmp_path / "probe.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider": {
                    "callable": "ebackbone_v3.providers.n_imagenet_mini:load_sample",
                    "kwargs": {
                        "dataset_root": str(dataset_root),
                        "split": "validation",
                        "sample_id": "validation/n00000000/sample_validation.npz",
                        "voxel_bins": 4,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    report = run_probe(config)
    assert report["status"] == "PASS"
    assert report["dataset"]["split"] == "validation"
    assert report["classification"]["class_count"] == 100
    assert report["classification"]["class_to_index"]["n00000099"] == 99
    assert report["raw_events"]["fields"]["t"]["dtype"] == "uint16"
    assert report["representations"]["event_frame"]["shape"] == [2, 480, 640]
    assert report["representations"]["voxel_grid"]["shape"] == [4, 480, 640]
    assert report["representations"]["time_surface"]["shape"] == [2, 480, 640]
    assert report["temporal_alignment"]["status"] == "VERIFIED_FROM_PROVIDER_EVIDENCE"


def test_provider_rejects_test_split_without_aliasing_validation(tmp_path: Path) -> None:
    with pytest.raises(ProbeError, match="no independent test split"):
        load_sample(
            dataset_root=tmp_path,
            split="test",
            sample_id="test/n00000000/sample.npz",
            voxel_bins=4,
        )


def test_provider_rejects_split_mismatch_before_archive_access(tmp_path: Path) -> None:
    with pytest.raises(ProbeError, match="does not match provider split"):
        load_sample(
            dataset_root=tmp_path,
            split="train",
            sample_id="validation/n00000000/sample.npz",
            voxel_bins=4,
        )


def test_provider_rejects_different_train_validation_class_sets(tmp_path: Path) -> None:
    dataset_root = tmp_path / "n_imagenet"
    _write_archive_fixture(dataset_root, validation_class_count=99)
    with pytest.raises(ProbeError, match="train/validation class sets differ"):
        load_sample(
            dataset_root=dataset_root,
            split="train",
            sample_id="train/n00000000/sample_train.npz",
            voxel_bins=4,
        )
