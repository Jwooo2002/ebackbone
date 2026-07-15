from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from ebackbone_v3.n_imagenet_mini_index import (
    ArchiveIndexError,
    build_archive_index,
    compute_source_file_checksums,
)


def _class_ids() -> list[str]:
    return [f"n{index:08d}" for index in range(100)]


def _tar_payload(
    class_id: str,
    *,
    reverse_members: bool,
    train_samples_per_class: int = 2,
    duplicate_member: bool = False,
) -> bytes:
    output = io.BytesIO()
    filenames = [f"sample_{index:02d}.npz" for index in range(train_samples_per_class)]
    if reverse_members:
        filenames.reverse()
    with tarfile.open(fileobj=output, mode="w:gz") as handle:
        directory = tarfile.TarInfo(class_id)
        directory.type = tarfile.DIRTYPE
        directory.mtime = 0
        handle.addfile(directory)
        for filename in filenames:
            payload = f"train|{class_id}|{filename}".encode("ascii")
            info = tarfile.TarInfo(f"{class_id}/{filename}")
            info.size = len(payload)
            info.mtime = 0
            handle.addfile(info, io.BytesIO(payload))
        if duplicate_member:
            filename = "sample_00.npz"
            payload = f"train|{class_id}|{filename}".encode("ascii")
            info = tarfile.TarInfo(f"{class_id}/{filename}")
            info.size = len(payload)
            info.mtime = 0
            handle.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _write_release(
    dataset_root: Path,
    *,
    reverse_members: bool = False,
    train_samples_per_class: int = 2,
    duplicate_train_member: bool = False,
    unexpected_validation_member: bool = False,
) -> None:
    archive_dir = dataset_root / "mini_zenodo" / "archives"
    archive_dir.mkdir(parents=True)
    class_ids = _class_ids()
    for part_index in range(1, 11):
        part_classes = class_ids[(part_index - 1) * 10 : part_index * 10]
        if reverse_members:
            part_classes.reverse()
        with zipfile.ZipFile(
            archive_dir / f"train_Part_{part_index}.zip",
            mode="w",
            compression=zipfile.ZIP_STORED,
        ) as handle:
            root = f"Part_{part_index}"
            handle.writestr(f"{root}/", b"")
            for class_id in part_classes:
                handle.writestr(
                    f"{root}/{class_id}.tar.gz",
                    _tar_payload(
                        class_id,
                        reverse_members=reverse_members,
                        train_samples_per_class=train_samples_per_class,
                        duplicate_member=(
                            duplicate_train_member and class_id == class_ids[0]
                        ),
                    ),
                )

    validation_classes = list(class_ids)
    if reverse_members:
        validation_classes.reverse()
    with zipfile.ZipFile(
        archive_dir / "mini_validation_split.zip",
        mode="w",
        compression=zipfile.ZIP_STORED,
    ) as handle:
        handle.writestr("extracted_val/", b"")
        for class_id in validation_classes:
            handle.writestr(f"extracted_val/{class_id}/", b"")
            filename = "sample_validation.npz"
            handle.writestr(
                f"extracted_val/{class_id}/{filename}",
                f"validation|{class_id}|{filename}".encode("ascii"),
            )
        if unexpected_validation_member:
            handle.writestr("README.txt", b"not a sample")

    # These intentionally contain stale, unrelated absolute paths. Their bytes
    # are provenance inputs, but they must not contribute any sample identity.
    (archive_dir / "train_list.txt").write_text(
        "/stale/full-1000-class-root/n99999999/not_in_mini.npz\n",
        encoding="utf-8",
    )
    (archive_dir / "val_list.txt").write_text(
        "/stale/full-1000-class-root/n99999999/not_in_mini_val.npz\n",
        encoding="utf-8",
    )


def _sample_projection(index: object) -> list[tuple[object, ...]]:
    samples = getattr(index, "samples")
    return [
        (
            sample.sample_id,
            sample.class_id,
            sample.class_index,
            sample.source_split,
            sample.source_archive_relative_path,
            sample.source_member_paths,
            sample.raw_member_size_bytes,
            sample.raw_member_sha256,
        )
        for sample in samples
    ]


def test_archive_index_is_canonical_repeatable_and_serializable(tmp_path: Path) -> None:
    dataset_root = tmp_path / "n_imagenet"
    _write_release(dataset_root, reverse_members=True)

    first = build_archive_index(dataset_root)
    second = build_archive_index(dataset_root)

    assert first == second
    assert len(first.samples) == 300
    assert [sample.sample_id for sample in first.samples] == sorted(
        sample.sample_id for sample in first.samples
    )
    assert first.class_to_index[0] == ("n00000000", 0)
    assert first.class_to_index[-1] == ("n00000099", 99)
    assert len(first.source_files) == 13
    assert all(source.sha256 is not None for source in first.source_files)
    assert all(sample.raw_member_sha256 is not None for sample in first.samples)

    train = next(
        sample
        for sample in first.samples
        if sample.sample_id == "train/n00000000/sample_00.npz"
    )
    assert train.source_split == "train"
    assert train.class_id == "n00000000"
    assert train.class_index == 0
    assert train.source_archive_relative_path.endswith("train_Part_1.zip")
    assert train.source_member_paths == (
        "Part_1/n00000000.tar.gz",
        "n00000000/sample_00.npz",
    )
    assert train.source_locator.endswith(
        "train_Part_1.zip!Part_1/n00000000.tar.gz!n00000000/sample_00.npz"
    )
    source_checksum = next(
        source
        for source in first.source_files
        if source.relative_path == train.source_archive_relative_path
    )
    assert train.source_archive_sha256 == source_checksum.sha256
    assert train.source_archive_size_bytes == source_checksum.size_bytes

    test_source = next(
        sample
        for sample in first.samples
        if sample.sample_id == "validation/n00000099/sample_validation.npz"
    )
    assert test_source.source_split == "validation"
    assert test_source.source_member_paths == (
        "extracted_val/n00000099/sample_validation.npz",
    )
    assert test_source.nested_member_path is None

    first_bytes = json.dumps(
        first.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    second_bytes = json.dumps(
        second.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert first_bytes == second_bytes


def test_member_discovery_does_not_depend_on_archive_iteration_order(
    tmp_path: Path,
) -> None:
    forward_root = tmp_path / "forward"
    reverse_root = tmp_path / "reverse"
    _write_release(forward_root, reverse_members=False)
    _write_release(reverse_root, reverse_members=True)

    forward = build_archive_index(forward_root, hash_source_files=False)
    reverse = build_archive_index(reverse_root, hash_source_files=False)

    assert _sample_projection(forward) == _sample_projection(reverse)


def test_bundled_lists_are_checksummed_but_never_define_membership(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "n_imagenet"
    _write_release(dataset_root)

    index = build_archive_index(dataset_root)
    list_sources = [
        source
        for source in index.source_files
        if source.role == "bundled_full_dataset_list"
    ]

    assert len(list_sources) == 2
    assert all(not source.authoritative_for_membership for source in list_sources)
    assert all("n99999999" not in sample.sample_id for sample in index.samples)
    for source in list_sources:
        path = dataset_root / source.relative_path
        assert source.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_hashes_can_be_skipped_only_as_an_explicit_bounded_option(tmp_path: Path) -> None:
    dataset_root = tmp_path / "n_imagenet"
    _write_release(dataset_root)

    index = build_archive_index(
        dataset_root,
        hash_raw_members=False,
        hash_source_files=False,
    )

    assert all(source.sha256 is None for source in index.source_files)
    assert all(sample.source_archive_sha256 is None for sample in index.samples)
    assert all(sample.raw_member_sha256 is None for sample in index.samples)
    assert index.raw_content_duplicates() == ()


def test_exact_raw_member_hash_duplicates_are_reported(tmp_path: Path) -> None:
    dataset_root = tmp_path / "n_imagenet"
    _write_release(dataset_root)
    archive_dir = dataset_root / "mini_zenodo" / "archives"
    # Repackage validation with two distinct stable identities carrying exactly
    # the same stored bytes, while retaining all 100 official source classes.
    validation_path = archive_dir / "mini_validation_split.zip"
    replacement = archive_dir / "replacement.zip"
    with zipfile.ZipFile(validation_path) as source, zipfile.ZipFile(
        replacement, mode="w", compression=zipfile.ZIP_STORED
    ) as target:
        for info in source.infolist():
            target.writestr(info, source.read(info.filename))
        target.writestr(
            "extracted_val/n00000000/copy_validation.npz",
            b"validation|n00000001|sample_validation.npz",
        )
    replacement.replace(validation_path)

    index = build_archive_index(dataset_root)
    duplicates = index.raw_content_duplicates()

    matching = [
        group
        for group in duplicates
        if "validation/n00000000/copy_validation.npz" in group.sample_ids
    ]
    assert len(matching) == 1
    assert matching[0].sample_ids == (
        "validation/n00000000/copy_validation.npz",
        "validation/n00000001/sample_validation.npz",
    )


def test_duplicate_stable_identity_is_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "n_imagenet"
    _write_release(dataset_root, duplicate_train_member=True)

    with pytest.raises(ArchiveIndexError, match="duplicate train TAR member path"):
        build_archive_index(dataset_root, hash_source_files=False)


def test_unexpected_member_layout_is_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "n_imagenet"
    _write_release(dataset_root, unexpected_validation_member=True)

    with pytest.raises(
        ArchiveIndexError, match="unexpected official-validation ZIP member layout"
    ):
        build_archive_index(dataset_root, hash_source_files=False)


def test_source_checksum_api_has_fixed_canonical_order(tmp_path: Path) -> None:
    dataset_root = tmp_path / "n_imagenet"
    _write_release(dataset_root)

    checksums = compute_source_file_checksums(dataset_root)

    assert [record.relative_path for record in checksums] == sorted(
        record.relative_path for record in checksums
    )
    assert len({record.relative_path for record in checksums}) == 13
    assert all(record.size_bytes > 0 for record in checksums)
