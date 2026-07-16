from __future__ import annotations

import gc
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

import ebackbone_v3.n_imagenet_mini_dataset as dataset_module
from ebackbone_v3.errors import DatasetError
from ebackbone_v3.n_imagenet_mini_dataset import (
    open_dataset,
)
from ebackbone_v3.n_imagenet_mini_index import build_archive_index
from ebackbone_v3.representations import cache_path
from ebackbone_v3.splits import (
    CHECKSUM_FILENAME,
    MANIFEST_FILENAMES,
    publish_split_manifests,
)


ROOT = Path(__file__).resolve().parents[1]
EVENT_DTYPE = np.dtype(
    [("x", "<u2"), ("y", "<u2"), ("t", "<u2"), ("p", "?")],
    align=False,
)


@dataclass(frozen=True)
class FixtureRelease:
    dataset_root: Path
    manifest_dir: Path


def _class_ids() -> list[str]:
    return [f"n{index:08d}" for index in range(100)]


def _npz_payload(*, class_index: int, sample_index: int, source_offset: int) -> bytes:
    event_data = np.empty(4, dtype=EVENT_DTYPE)
    event_data["x"] = np.asarray(
        [class_index, (class_index * 7 + sample_index + source_offset) % 640, 639, 1],
        dtype="<u2",
    )
    event_data["y"] = np.asarray(
        [class_index % 480, (class_index * 11 + sample_index + source_offset) % 480, 479, 2],
        dtype="<u2",
    )
    event_data["t"] = np.asarray(
        [0, 4 + sample_index, 10 + class_index, 20 + class_index + sample_index],
        dtype="<u2",
    )
    event_data["p"] = np.asarray([False, True, bool(sample_index % 2), True], dtype="?")
    output = io.BytesIO()
    np.savez(output, event_data=event_data)
    return output.getvalue()


def _train_tar(class_id: str, class_index: int) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as handle:
        directory = tarfile.TarInfo(class_id)
        directory.type = tarfile.DIRTYPE
        directory.mtime = 0
        handle.addfile(directory)
        for sample_index in range(2):
            payload = _npz_payload(
                class_index=class_index,
                sample_index=sample_index,
                source_offset=0,
            )
            info = tarfile.TarInfo(f"{class_id}/sample_{sample_index:02d}.npz")
            info.size = len(payload)
            info.mtime = 0
            handle.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _write_valid_release(dataset_root: Path) -> None:
    archive_dir = dataset_root / "mini_zenodo" / "archives"
    archive_dir.mkdir(parents=True)
    class_ids = _class_ids()
    for part_index in range(1, 11):
        with zipfile.ZipFile(
            archive_dir / f"train_Part_{part_index}.zip",
            mode="w",
            compression=zipfile.ZIP_STORED,
        ) as archive:
            archive.writestr(f"Part_{part_index}/", b"")
            for class_index in range((part_index - 1) * 10, part_index * 10):
                class_id = class_ids[class_index]
                archive.writestr(
                    f"Part_{part_index}/{class_id}.tar.gz",
                    _train_tar(class_id, class_index),
                )
    with zipfile.ZipFile(
        archive_dir / "mini_validation_split.zip",
        mode="w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.writestr("extracted_val/", b"")
        for class_index, class_id in enumerate(class_ids):
            archive.writestr(f"extracted_val/{class_id}/", b"")
            archive.writestr(
                f"extracted_val/{class_id}/validation.npz",
                _npz_payload(
                    class_index=class_index,
                    sample_index=0,
                    source_offset=1000,
                ),
            )
    (archive_dir / "train_list.txt").write_text("/stale/train/path.npz\n", encoding="utf-8")
    (archive_dir / "val_list.txt").write_text("/stale/validation/path.npz\n", encoding="utf-8")


@pytest.fixture(scope="module")
def fixture_release(tmp_path_factory: pytest.TempPathFactory) -> FixtureRelease:
    root = tmp_path_factory.mktemp("manifest-dataset") / "n_imagenet"
    _write_valid_release(root)
    manifest_dir = root.parent / "manifests"
    index = build_archive_index(root)
    publish_split_manifests(
        index,
        manifest_dir=manifest_dir,
        seed=20260715,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )
    return FixtureRelease(dataset_root=root, manifest_dir=manifest_dir)


def _dataset(
    release: FixtureRelease,
    split: str,
    *,
    baseline: str = "b1",
    cache: str = "off",
    cache_root: Path | None = None,
    allow_final_test: bool = False,
) -> object:
    return open_dataset(
        manifest_dir=release.manifest_dir,
        dataset_root=release.dataset_root,
        baseline=baseline,  # type: ignore[arg-type]
        cache=cache,  # type: ignore[arg-type]
        cache_root=cache_root,
        split=split,  # type: ignore[arg-type]
        allow_final_test=allow_final_test,
    )


def _manifest_rows(directory: Path, split: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (directory / MANIFEST_FILENAMES[split]).read_text(encoding="utf-8").splitlines()
    ]


def _rewrite_manifest_integrity(directory: Path, split: str, rows: list[dict[str, object]]) -> None:
    manifest_path = directory / MANIFEST_FILENAMES[split]
    manifest_path.write_bytes(
        b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in rows
        )
    )
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    payload = manifest_path.read_bytes()
    provenance["manifest_files"][split]["sha256"] = hashlib.sha256(payload).hexdigest()
    provenance["manifest_files"][split]["size_bytes"] = len(payload)
    provenance["manifest_files"][split]["sample_count"] = len(rows)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_lines = []
    for name in sorted(
        path.name for path in directory.iterdir() if path.name != CHECKSUM_FILENAME
    ):
        checksum_lines.append(
            f"{hashlib.sha256((directory / name).read_bytes()).hexdigest()}  {name}\n"
        )
    (directory / CHECKSUM_FILENAME).write_text("".join(checksum_lines), encoding="ascii")


def _rewrite_checksum_file(directory: Path) -> None:
    checksum_lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in sorted(directory.iterdir(), key=lambda candidate: candidate.name)
        if path.name != CHECKSUM_FILENAME
    ]
    (directory / CHECKSUM_FILENAME).write_text("".join(checksum_lines), encoding="ascii")


def _with_changed_npz_comment(payload: bytes) -> bytes:
    output = io.BytesIO(payload)
    with zipfile.ZipFile(output, mode="a") as archive:
        archive.comment = b"same-event-data-different-npz-container"
    return output.getvalue()


def _replace_train_payload(dataset_root: Path, row: dict[str, object]) -> None:
    archive_path = dataset_root / str(row["source_archive"])
    tar_member, npz_member = tuple(row["source_members"])
    replacement_zip = archive_path.with_suffix(".replacement.zip")
    with zipfile.ZipFile(archive_path) as source, zipfile.ZipFile(
        replacement_zip, mode="w", compression=zipfile.ZIP_STORED
    ) as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == tar_member:
                rewritten_tar = io.BytesIO()
                with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as old_tar, tarfile.open(
                    fileobj=rewritten_tar, mode="w:gz"
                ) as new_tar:
                    for member in old_tar:
                        copied = tarfile.TarInfo(member.name)
                        copied.type = member.type
                        copied.mode = member.mode
                        copied.mtime = 0
                        if member.isfile():
                            extracted = old_tar.extractfile(member)
                            assert extracted is not None
                            member_payload = extracted.read()
                            if member.name == npz_member:
                                member_payload = _with_changed_npz_comment(member_payload)
                            copied.size = len(member_payload)
                            new_tar.addfile(copied, io.BytesIO(member_payload))
                        else:
                            copied.size = 0
                            new_tar.addfile(copied)
                payload = rewritten_tar.getvalue()
            target.writestr(info.filename, payload)
    replacement_zip.replace(archive_path)


def _tamper_cached_manifest(cache_file: Path, *, field: str) -> None:
    with np.load(cache_file, allow_pickle=False) as archive:
        tensors = {
            name: archive[name].copy()
            for name in ("event_frame", "voxel_grid", "time_surface")
        }
        manifest = json.loads(bytes(archive["manifest_utf8"]).decode("utf-8"))
    if field == "renderer":
        manifest["renderer"]["voxel_bins"] = 4
    elif field == "contract":
        manifest["contract_name"] = "stale-contract"
    else:
        raise AssertionError(field)
    with cache_file.open("wb") as handle:
        np.savez(
            handle,
            **tensors,
            manifest_utf8=np.frombuffer(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
                dtype=np.uint8,
            ),
        )


def test_resolves_explicit_train_validation_and_opted_in_synthetic_final_test_rows(
    fixture_release: FixtureRelease,
) -> None:
    train_sample = _dataset(fixture_release, "train", baseline="b0")[0]
    validation_sample = _dataset(fixture_release, "validation", baseline="b0")[0]

    with pytest.raises(DatasetError, match="allow_final_test=True"):
        _dataset(fixture_release, "test", baseline="b0")

    test_sample = _dataset(
        fixture_release,
        "test",
        baseline="b0",
        allow_final_test=True,
    )[0]

    assert train_sample.metadata.split == "train"
    assert train_sample.metadata.source_split == "train"
    assert validation_sample.metadata.split == "validation"
    assert validation_sample.metadata.source_split == "train"
    assert test_sample.metadata.split == "test"
    assert test_sample.metadata.source_split == "validation"
    assert test_sample.metadata.sample_id.startswith("validation/")
    for sample in (train_sample, validation_sample, test_sample):
        assert sample.metadata.label == sample.metadata.class_index
        assert sample.metadata.sample_id.split("/")[1] == sample.metadata.synset
        assert sample.metadata.source_locator.startswith("mini_zenodo/archives/")
        assert len(sample.metadata.raw_payload_sha256) == 64
        assert sample.raw_events.sample_id == sample.metadata.sample_id
        assert sample.raw_events.temporal_start == sample.metadata.temporal_start
        assert sample.raw_events.temporal_end == sample.metadata.temporal_end


def test_missing_invalid_and_ambiguous_split_arguments_fail_closed(
    fixture_release: FixtureRelease,
) -> None:
    for invalid in (None, "", "val", False, True):
        with pytest.raises(DatasetError, match="explicitly set"):
            open_dataset(
                manifest_dir=fixture_release.manifest_dir,
                dataset_root=fixture_release.dataset_root,
                baseline="b0",
                cache="off",
                split=invalid,  # type: ignore[arg-type]
            )
    for legacy in ({"train": False}, {"eval": True}):
        with pytest.raises(TypeError):
            open_dataset(
                manifest_dir=fixture_release.manifest_dir,
                split="train",
                dataset_root=fixture_release.dataset_root,
                **legacy,  # type: ignore[arg-type]
            )


def test_denied_final_test_access_reads_no_manifest_or_archive(
    fixture_release: FixtureRelease,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dataset_module,
        "_read_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("manifest read")),
    )
    monkeypatch.setattr(
        dataset_module,
        "_read_archive_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("archive read")),
    )

    with pytest.raises(DatasetError, match="no test manifest or archive member was read"):
        open_dataset(
            manifest_dir=fixture_release.manifest_dir,
            split="test",
            dataset_root=fixture_release.dataset_root,
            baseline="b0",
        )


def test_manifest_checksum_and_provenance_mismatches_fail_before_archive_access(
    fixture_release: FixtureRelease,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dataset_module,
        "_read_archive_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("archive read")),
    )

    checksum_dir = tmp_path / "checksum-mismatch"
    shutil.copytree(fixture_release.manifest_dir, checksum_dir)
    with (checksum_dir / MANIFEST_FILENAMES["train"]).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(DatasetError, match="manifest SHA-256 does not match"):
        open_dataset(
            manifest_dir=checksum_dir,
            split="train",
            dataset_root=fixture_release.dataset_root,
        )

    provenance_dir = tmp_path / "provenance-mismatch"
    shutil.copytree(fixture_release.manifest_dir, provenance_dir)
    provenance_path = provenance_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["dataset_release"] = "wrong release"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rewrite_checksum_file(provenance_dir)
    with pytest.raises(DatasetError, match="dataset release mismatch"):
        open_dataset(
            manifest_dir=provenance_dir,
            split="train",
            dataset_root=fixture_release.dataset_root,
        )


def test_runtime_membership_uses_only_selected_manifests_and_never_scans_archives(
    fixture_release: FixtureRelease,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_reads: list[str] = []
    original_read_bytes = dataset_module._read_bytes

    def recording_read_bytes(path: Path, description: str) -> bytes:
        manifest_reads.append(path.name)
        return original_read_bytes(path, description)

    monkeypatch.setattr(dataset_module, "_read_bytes", recording_read_bytes)
    monkeypatch.setattr(
        Path,
        "glob",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("archive glob")),
    )
    monkeypatch.setattr(
        Path,
        "rglob",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("archive rglob")),
    )

    train = open_dataset(
        manifest_dir=fixture_release.manifest_dir,
        split="train",
        dataset_root=fixture_release.dataset_root,
        baseline="b0",
    )
    validation = open_dataset(
        manifest_dir=fixture_release.manifest_dir,
        split="validation",
        dataset_root=fixture_release.dataset_root,
        baseline="b0",
    )

    assert "test.jsonl" not in manifest_reads
    assert {row.split for row in train._rows} == {"train"}  # noqa: SLF001
    assert {row.source_split for row in train._rows} == {"train"}  # noqa: SLF001
    assert {row.split for row in validation._rows} == {"validation"}  # noqa: SLF001
    assert {row.source_split for row in validation._rows} == {"train"}  # noqa: SLF001
    train_ids = {row.sample_id for row in train._rows}  # noqa: SLF001
    validation_ids = {row.sample_id for row in validation._rows}  # noqa: SLF001
    assert not (train_ids & validation_ids)


def test_open_dataset_preserves_train_and_validation_manifest_order_and_labels(
    fixture_release: FixtureRelease,
) -> None:
    for split in ("train", "validation"):
        dataset = open_dataset(
            manifest_dir=fixture_release.manifest_dir,
            split=split,  # type: ignore[arg-type]
            dataset_root=fixture_release.dataset_root,
            baseline="b0",
        )
        rows = _manifest_rows(fixture_release.manifest_dir, split)

        assert dataset.sample_ids == tuple(str(row["sample_id"]) for row in rows)
        first = dataset[0]
        assert first.metadata.sample_id == rows[0]["sample_id"]
        assert first.metadata.label == rows[0]["class_label"]
        assert first.metadata.split == split
        assert first.metadata.source_split == "train"


def test_supervised_runtime_consumers_use_the_canonical_open_dataset_entrypoint() -> None:
    runtime_sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in (
            ROOT / "ebackbone_v3" / "b0_training.py",
            ROOT / "ebackbone_v3" / "b0_production.py",
            ROOT / "ebackbone_v3" / "cli.py",
        )
    }
    for name in ("b0_training.py", "b0_production.py"):
        source = runtime_sources[name]
        assert "open_dataset(" in source
        assert "ManifestBackedNImageNetMiniDataset" not in source
        assert "manifest_path=" not in source
        assert "MANIFEST_FILENAMES" not in source

    adapter_source = (ROOT / "ebackbone_v3" / "n_imagenet_mini_dataset.py").read_text(
        encoding="utf-8"
    )
    inspect_source = adapter_source.split("def inspect_sample(", maxsplit=1)[1].split(
        "\ndef _resolve_project_split", maxsplit=1
    )[0]
    assert "open_dataset(" in inspect_source
    assert "_ManifestBackedNImageNetMiniDataset(" not in inspect_source
    assert '"--manifest",' not in runtime_sources["cli.py"]
    assert '"--train-manifest",' not in runtime_sources["cli.py"]
    assert '"--validation-manifest",' not in runtime_sources["cli.py"]


def test_declared_archive_and_member_are_opened_without_extraction(
    fixture_release: FixtureRelease,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(fixture_release, "train", baseline="b0")
    expected_archive = dataset._rows[0].source_archive  # type: ignore[attr-defined]
    opened: list[str] = []
    original_init = zipfile.ZipFile.__init__

    def recording_init(self: zipfile.ZipFile, file: object, *args: object, **kwargs: object) -> None:
        if isinstance(file, (str, Path)):
            opened.append(Path(file).name)
        original_init(self, file, *args, **kwargs)

    def forbidden_extractall(*args: object, **kwargs: object) -> None:
        raise AssertionError("full archive extraction must not be used")

    monkeypatch.setattr(zipfile.ZipFile, "__init__", recording_init)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", forbidden_extractall)
    sample = dataset[0]

    assert sample.metadata.source_locator.endswith("sample_00.npz") or sample.metadata.source_locator.endswith(
        "sample_01.npz"
    )
    assert opened == [Path(expected_archive).name]


def test_decoding_is_deterministic_and_preserves_verified_raw_dtypes(
    fixture_release: FixtureRelease,
) -> None:
    dataset = _dataset(fixture_release, "train", baseline="b0")
    first = dataset[0]
    second = dataset[0]

    for name, dtype in (("x", "<u2"), ("y", "<u2"), ("t", "<u2"), ("p", "?")):
        assert first.raw_events.fields[name].dtype == np.dtype(dtype)
        np.testing.assert_array_equal(first.raw_events.fields[name], second.raw_events.fields[name])
    assert first.metadata.raw_fingerprint == second.metadata.raw_fingerprint
    assert first.metadata.interval_closure == "[]"
    assert first.raw_events.timestamp_unit == "microsecond"


def test_b0_frame_only_and_b1_bundle_share_identity_and_interval(
    fixture_release: FixtureRelease,
) -> None:
    b0 = _dataset(fixture_release, "validation", baseline="b0")[0]
    b1 = _dataset(fixture_release, "validation", baseline="b1")[0]

    assert set(b0.tensors) == {"event_frame"}
    assert set(b1.tensors) == {"event_frame", "voxel_grid", "time_surface"}
    np.testing.assert_array_equal(b0.tensors["event_frame"], b1.tensors["event_frame"])
    assert b1.tensors["event_frame"].shape == (2, 480, 640)
    assert b1.tensors["voxel_grid"].shape == (2, 5, 480, 640)
    assert b1.tensors["time_surface"].shape == (2, 480, 640)
    assert all(tensor.dtype == np.float32 for tensor in b1.tensors.values())
    assert b0.metadata.raw_fingerprint == b1.metadata.raw_fingerprint
    assert (
        b0.metadata.temporal_start,
        b0.metadata.temporal_end,
        b0.metadata.interval_closure,
    ) == (
        b1.metadata.temporal_start,
        b1.metadata.temporal_end,
        b1.metadata.interval_closure,
    )
    assert np.expm1(b1.tensors["event_frame"]).sum(dtype=np.float64) == pytest.approx(
        b1.metadata.event_count
    )
    assert np.expm1(b1.tensors["voxel_grid"]).sum(dtype=np.float64) == pytest.approx(
        b1.metadata.event_count,
        abs=1e-5,
    )


def test_b0_frame_cache_never_calls_tri_representation_renderer(
    fixture_release: FixtureRelease,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dataset_module,
        "render_production_representations",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("B0 requested voxel grid or time surface")
        ),
    )
    sample = _dataset(fixture_release, "train", baseline="b0", cache="off")[0]
    assert set(sample.tensors) == {"event_frame"}
    assert sample.metadata.cache_status == "off"

    cached = _dataset(
        fixture_release, "train", baseline="b0", cache="on", cache_root=tmp_path / "cache"
    )[0]
    assert cached.metadata.cache_status == "miss"
    cache_file = Path(cached.metadata.cache_path or "")
    with np.load(cache_file, allow_pickle=False) as archive:
        assert set(archive.files) == {"event_frame", "manifest_utf8"}


def test_b0_warm_frame_cache_opens_no_source_archive(
    fixture_release: FixtureRelease,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(
        fixture_release, "train", baseline="b0", cache="on", cache_root=tmp_path / "frames"
    )
    cold = dataset[0]
    assert cold.metadata.cache_status == "miss"
    archive_opens = 0
    original = dataset_module._read_archive_payload

    def counted(*args: object, **kwargs: object) -> bytes:
        nonlocal archive_opens
        archive_opens += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(dataset_module, "_read_archive_payload", counted)
    warm = dataset[0]
    assert warm.metadata.cache_status == "hit"
    assert warm.raw_events is None
    assert warm.metadata.archive_decode_seconds == 0.0
    assert archive_opens == 0
    np.testing.assert_array_equal(cold.tensors["event_frame"], warm.tensors["event_frame"])


def test_cache_hit_miss_and_stale_renderer_or_contract_entries_are_rebuilt(
    fixture_release: FixtureRelease,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    dataset = _dataset(
        fixture_release,
        "train",
        baseline="b1",
        cache="on",
        cache_root=cache_root,
    )
    first = dataset[0]
    cache_file = Path(first.metadata.cache_path or "")
    assert first.metadata.cache_status == "miss"
    assert cache_file.is_file()
    assert cache_file == cache_path(cache_root, first.metadata.cache_key)
    with np.load(cache_file, allow_pickle=False) as archive:
        cache_manifest = json.loads(bytes(archive["manifest_utf8"]).decode("utf-8"))
    assert cache_manifest["source"]["raw_content_sha256"] == first.metadata.raw_payload_sha256
    assert cache_manifest["source"]["project_split"] == first.metadata.split

    with monkeypatch.context() as local_patch:
        local_patch.setattr(
            dataset_module,
            "render_production_representations",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
        )
        hit = dataset[0]
    assert hit.metadata.cache_status == "hit"

    for field in ("renderer", "contract"):
        _tamper_cached_manifest(cache_file, field=field)
        rebuilt = dataset[0]
        assert rebuilt.metadata.cache_status == "stale_rebuilt"
        assert rebuilt.metadata.cache_key == first.metadata.cache_key


def test_raw_payload_hash_change_uses_a_new_cache_entry(
    fixture_release: FixtureRelease,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    original = _dataset(
        fixture_release,
        "train",
        baseline="b1",
        cache="on",
        cache_root=cache_root,
    )[0]

    changed_root = tmp_path / "changed-data"
    shutil.copytree(fixture_release.dataset_root, changed_root)
    row = _manifest_rows(fixture_release.manifest_dir, "train")[0]
    _replace_train_payload(changed_root, row)
    changed_manifest_dir = tmp_path / "changed-manifests"
    publish_split_manifests(
        build_archive_index(changed_root),
        manifest_dir=changed_manifest_dir,
        seed=20260715,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )
    changed_release = FixtureRelease(changed_root, changed_manifest_dir)
    changed = _dataset(
        changed_release,
        "train",
        baseline="b1",
        cache="on",
        cache_root=cache_root,
    )[0]

    assert changed.metadata.cache_status == "miss"
    assert changed.metadata.raw_payload_sha256 != original.metadata.raw_payload_sha256
    assert changed.metadata.cache_key != original.metadata.cache_key
    assert changed.metadata.raw_fingerprint == original.metadata.raw_fingerprint


def test_corrupted_raw_payload_or_hash_is_rejected_before_decode(
    fixture_release: FixtureRelease,
    tmp_path: Path,
) -> None:
    corrupted_root = tmp_path / "corrupted-data"
    shutil.copytree(fixture_release.dataset_root, corrupted_root)
    _replace_train_payload(corrupted_root, _manifest_rows(fixture_release.manifest_dir, "train")[0])
    corrupted_release = FixtureRelease(corrupted_root, fixture_release.manifest_dir)

    with pytest.raises(DatasetError, match="raw payload (size|SHA-256) mismatch"):
        _dataset(corrupted_release, "train", baseline="b0")[0]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda row: row.__setitem__("class_index", 99),
            "class index/label",
        ),
        (
            lambda row: row.pop("source_locator"),
            "row keys",
        ),
    ],
)
def test_malformed_manifest_rows_are_rejected_before_archive_access(
    fixture_release: FixtureRelease,
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    malformed_dir = tmp_path / "malformed-manifests"
    shutil.copytree(fixture_release.manifest_dir, malformed_dir)
    rows = _manifest_rows(malformed_dir, "train")
    assert callable(mutate)
    mutate(rows[0])
    _rewrite_manifest_integrity(malformed_dir, "train", rows)
    malformed = FixtureRelease(fixture_release.dataset_root, malformed_dir)

    with pytest.raises(DatasetError, match=message):
        _dataset(malformed, "train", baseline="b0")


def test_repeated_access_closes_archive_handles_and_never_initializes_cuda(
    fixture_release: FixtureRelease,
) -> None:
    assert not torch.cuda.is_initialized()
    dataset = _dataset(fixture_release, "train", baseline="b0")
    baseline_fds = len(list(Path("/proc/self/fd").iterdir()))
    for _ in range(20):
        sample = dataset[0]
        assert sample.metadata.cache_status == "off"
        del sample
    gc.collect()
    after_fds = len(list(Path("/proc/self/fd").iterdir()))

    assert after_fds <= baseline_fds + 2
    assert not torch.cuda.is_initialized()


def test_inspect_sample_cli_reports_metadata_and_tensor_summaries_only(
    fixture_release: FixtureRelease,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "inspect-sample",
            "--manifest-dir",
            str(fixture_release.manifest_dir),
            "--index",
            "0",
            "--baseline",
            "b0",
            "--cache",
            "off",
            "--split",
            "train",
            "--dataset-root",
            str(fixture_release.dataset_root),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "PASS"
    assert report["device"] == "cpu"
    assert set(report) == {"status", "device", "adapter", "baseline", "metadata", "tensors"}
    assert set(report["tensors"]) == {"event_frame"}
    assert report["tensors"]["event_frame"]["shape"] == [2, 480, 640]
    assert "model" not in result.stdout.lower()
