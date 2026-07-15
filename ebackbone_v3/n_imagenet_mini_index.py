"""Deterministic archive/member index for the N-ImageNet mini release.

The index is deliberately independent of representation rendering.  It reads
only archive metadata and, when requested, the stored NPZ member bytes.  The
bundled full-dataset path lists are checksummed for release provenance but are
never consulted to discover samples or labels.
"""

from __future__ import annotations

import hashlib
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, Mapping, Sequence

from ebackbone_v3.errors import EBackboneV3Error


ARCHIVE_DIRECTORY = Path("mini_zenodo") / "archives"
TRAIN_ARCHIVE_NAMES = tuple(f"train_Part_{index}.zip" for index in range(1, 11))
VALIDATION_ARCHIVE_NAME = "mini_validation_split.zip"
BUNDLED_LIST_NAMES = ("train_list.txt", "val_list.txt")
EXPECTED_CLASS_COUNT = 100
CLASS_ID_PATTERN = re.compile(r"^n[0-9]{8}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HASH_CHUNK_SIZE = 1024 * 1024


class ArchiveIndexError(EBackboneV3Error):
    """Raised when release membership cannot be indexed unambiguously."""


@dataclass(frozen=True)
class SourceFileChecksum:
    """Checksum metadata for one local file that identifies the release."""

    relative_path: str
    role: str
    authoritative_for_membership: bool
    size_bytes: int
    sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "role": self.role,
            "authoritative_for_membership": self.authoritative_for_membership,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ArchiveSampleRecord:
    """Stable source identity and archive locator for one stored NPZ sample."""

    source_split: str
    class_id: str
    class_index: int
    sample_id: str
    source_archive_relative_path: str
    archive_member_path: str
    nested_member_path: str | None
    source_archive_size_bytes: int
    source_archive_sha256: str | None
    raw_member_size_bytes: int
    raw_member_sha256: str | None

    @property
    def source_member_paths(self) -> tuple[str, ...]:
        if self.nested_member_path is None:
            return (self.archive_member_path,)
        return (self.archive_member_path, self.nested_member_path)

    @property
    def source_locator(self) -> str:
        return "!".join((self.source_archive_relative_path, *self.source_member_paths))

    def to_dict(self) -> dict[str, object]:
        return {
            "source_split": self.source_split,
            "class_id": self.class_id,
            "class_index": self.class_index,
            "sample_id": self.sample_id,
            "source_archive_relative_path": self.source_archive_relative_path,
            "source_member_paths": list(self.source_member_paths),
            "source_locator": self.source_locator,
            "source_archive_size_bytes": self.source_archive_size_bytes,
            "source_archive_sha256": self.source_archive_sha256,
            "raw_member_size_bytes": self.raw_member_size_bytes,
            "raw_member_sha256": self.raw_member_sha256,
        }


@dataclass(frozen=True)
class RawContentDuplicate:
    """One exact stored-NPZ SHA-256 shared by distinct stable sample IDs."""

    raw_member_sha256: str
    sample_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_member_sha256": self.raw_member_sha256,
            "sample_ids": list(self.sample_ids),
        }


@dataclass(frozen=True)
class NImageNetMiniArchiveIndex:
    """Complete deterministic source index plus release-file provenance."""

    dataset_root: str
    class_to_index: tuple[tuple[str, int], ...]
    source_files: tuple[SourceFileChecksum, ...]
    samples: tuple[ArchiveSampleRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_root": self.dataset_root,
            "class_to_index": dict(self.class_to_index),
            "source_files": [record.to_dict() for record in self.source_files],
            "samples": [record.to_dict() for record in self.samples],
        }

    def raw_content_duplicates(self) -> tuple[RawContentDuplicate, ...]:
        return find_raw_content_duplicates(self.samples)


@dataclass(frozen=True)
class _UnresolvedSample:
    source_split: str
    class_id: str
    filename: str
    source_archive_relative_path: str
    archive_member_path: str
    nested_member_path: str | None
    raw_member_size_bytes: int
    raw_member_sha256: str | None

    @property
    def sample_id(self) -> str:
        return f"{self.source_split}/{self.class_id}/{self.filename}"

    @property
    def source_locator(self) -> str:
        paths = (self.archive_member_path,)
        if self.nested_member_path is not None:
            paths += (self.nested_member_path,)
        return "!".join((self.source_archive_relative_path, *paths))


def build_archive_index(
    dataset_root: str | Path,
    *,
    hash_raw_members: bool = True,
    hash_source_files: bool = True,
) -> NImageNetMiniArchiveIndex:
    """Index the fixed archive-defined mini release in canonical sorted order.

    ``hash_raw_members`` hashes the exact stored NPZ bytes and permits exact
    duplicate-content auditing without decoding arrays. ``hash_source_files``
    hashes all ten train ZIPs, the official-validation ZIP, and the two bundled
    non-authoritative path lists. Disabling either option is intended only for
    bounded inspection; a production manifest should enable both.
    """

    root = Path(dataset_root).expanduser().resolve()
    archive_dir = root / ARCHIVE_DIRECTORY
    if not archive_dir.is_dir():
        raise FileNotFoundError(f"N-ImageNet mini archive directory does not exist: {archive_dir}")

    source_files = compute_source_file_checksums(
        root,
        hash_contents=hash_source_files,
    )
    checksum_by_relative_path = {
        record.relative_path: record for record in source_files
    }

    train_archives = _validate_train_archive_set(archive_dir)
    train_class_locators = _discover_train_class_locators(root, train_archives)
    validation_path = archive_dir / VALIDATION_ARCHIVE_NAME
    _require_file(validation_path, "official-validation archive")
    validation_classes = _discover_validation_classes(validation_path)
    train_classes = set(train_class_locators)
    if train_classes != validation_classes:
        raise ArchiveIndexError(
            "N-ImageNet mini train/official-validation class sets differ: "
            f"train_only={sorted(train_classes - validation_classes)}, "
            f"validation_only={sorted(validation_classes - train_classes)}"
        )
    if len(train_classes) != EXPECTED_CLASS_COUNT:
        raise ArchiveIndexError(
            f"expected {EXPECTED_CLASS_COUNT} classes; observed {len(train_classes)}"
        )
    class_to_index = {
        class_id: index for index, class_id in enumerate(sorted(train_classes))
    }

    unresolved: list[_UnresolvedSample] = []
    for class_id in sorted(train_class_locators):
        zip_path, tar_member_path = train_class_locators[class_id]
        unresolved.extend(
            _index_train_class(
                root,
                zip_path=zip_path,
                tar_member_path=tar_member_path,
                class_id=class_id,
                hash_raw_members=hash_raw_members,
            )
        )
    unresolved.extend(
        _index_validation_archive(
            root,
            validation_path=validation_path,
            hash_raw_members=hash_raw_members,
        )
    )
    unresolved.sort(key=lambda record: record.sample_id)
    _validate_unique_samples(unresolved)

    samples: list[ArchiveSampleRecord] = []
    for record in unresolved:
        source_file = checksum_by_relative_path[record.source_archive_relative_path]
        samples.append(
            ArchiveSampleRecord(
                source_split=record.source_split,
                class_id=record.class_id,
                class_index=class_to_index[record.class_id],
                sample_id=record.sample_id,
                source_archive_relative_path=record.source_archive_relative_path,
                archive_member_path=record.archive_member_path,
                nested_member_path=record.nested_member_path,
                source_archive_size_bytes=source_file.size_bytes,
                source_archive_sha256=source_file.sha256,
                raw_member_size_bytes=record.raw_member_size_bytes,
                raw_member_sha256=record.raw_member_sha256,
            )
        )
    return NImageNetMiniArchiveIndex(
        dataset_root=str(root),
        class_to_index=tuple(class_to_index.items()),
        source_files=source_files,
        samples=tuple(samples),
    )


def compute_source_file_checksums(
    dataset_root: str | Path,
    *,
    hash_contents: bool = True,
) -> tuple[SourceFileChecksum, ...]:
    """Return canonical provenance metadata for archives and bundled lists."""

    root = Path(dataset_root).expanduser().resolve()
    archive_dir = root / ARCHIVE_DIRECTORY
    paths_and_roles: list[tuple[Path, str, bool]] = [
        *((archive_dir / name, "train_archive", True) for name in TRAIN_ARCHIVE_NAMES),
        (archive_dir / VALIDATION_ARCHIVE_NAME, "official_validation_archive", True),
        *((archive_dir / name, "bundled_full_dataset_list", False) for name in BUNDLED_LIST_NAMES),
    ]
    records: list[SourceFileChecksum] = []
    for path, role, authoritative in paths_and_roles:
        _require_file(path, role.replace("_", " "))
        relative_path = _relative_posix(path, root)
        records.append(
            SourceFileChecksum(
                relative_path=relative_path,
                role=role,
                authoritative_for_membership=authoritative,
                size_bytes=path.stat().st_size,
                sha256=_sha256_file(path) if hash_contents else None,
            )
        )
    return tuple(sorted(records, key=lambda record: record.relative_path))


def find_raw_content_duplicates(
    samples: Sequence[ArchiveSampleRecord],
) -> tuple[RawContentDuplicate, ...]:
    """Group exact stored-NPZ hashes shared by two or more sample IDs."""

    by_hash: dict[str, set[str]] = {}
    for record in samples:
        digest = record.raw_member_sha256
        if digest is None:
            continue
        if not SHA256_PATTERN.fullmatch(digest):
            raise ArchiveIndexError(
                f"invalid raw member SHA-256 for {record.sample_id}: {digest!r}"
            )
        by_hash.setdefault(digest, set()).add(record.sample_id)
    return tuple(
        RawContentDuplicate(digest, tuple(sorted(sample_ids)))
        for digest, sample_ids in sorted(by_hash.items())
        if len(sample_ids) > 1
    )


def _validate_train_archive_set(archive_dir: Path) -> tuple[Path, ...]:
    expected = tuple(archive_dir / name for name in TRAIN_ARCHIVE_NAMES)
    for path in expected:
        _require_file(path, "train archive")
    unexpected = sorted(
        path.name
        for path in archive_dir.glob("train_Part_*.zip")
        if path.name not in TRAIN_ARCHIVE_NAMES
    )
    if unexpected:
        raise ArchiveIndexError(
            "unexpected train archives make the selected release ambiguous: "
            + ", ".join(unexpected)
        )
    return expected


def _discover_train_class_locators(
    dataset_root: Path,
    zip_paths: Iterable[Path],
) -> dict[str, tuple[Path, str]]:
    locators: dict[str, tuple[Path, str]] = {}
    seen_member_locators: set[str] = set()
    for zip_path in sorted(zip_paths, key=lambda path: path.name):
        expected_root = f"Part_{_train_part_number(zip_path.name)}"
        try:
            with zipfile.ZipFile(zip_path) as handle:
                for info in handle.infolist():
                    if info.is_dir():
                        if info.filename.rstrip("/") != expected_root:
                            raise ArchiveIndexError(
                                f"unexpected train ZIP directory layout: {zip_path}!{info.filename}"
                            )
                        continue
                    parts = _safe_member_parts(info.filename, "train ZIP member")
                    if len(parts) != 2 or parts[0] != expected_root:
                        raise ArchiveIndexError(
                            f"unexpected train ZIP member layout: {zip_path}!{info.filename}"
                        )
                    class_id = PurePosixPath(parts[1]).name.removesuffix(".tar.gz")
                    if parts[1] != f"{class_id}.tar.gz":
                        raise ArchiveIndexError(
                            f"train ZIP payload must be <class_id>.tar.gz: {info.filename}"
                        )
                    _validate_class_id(class_id)
                    relative_zip = _relative_posix(zip_path, dataset_root)
                    locator = f"{relative_zip}!{info.filename}"
                    if locator in seen_member_locators:
                        raise ArchiveIndexError(f"duplicate train ZIP member locator: {locator}")
                    seen_member_locators.add(locator)
                    if class_id in locators:
                        previous = locators[class_id]
                        raise ArchiveIndexError(
                            f"duplicate train class payload for {class_id}: "
                            f"{previous[0]}!{previous[1]} and {zip_path}!{info.filename}"
                        )
                    locators[class_id] = (zip_path, info.filename)
        except zipfile.BadZipFile as exc:
            raise ArchiveIndexError(f"invalid train ZIP archive: {zip_path}") from exc
    return locators


def _discover_validation_classes(validation_path: Path) -> set[str]:
    classes: set[str] = set()
    seen_member_paths: set[str] = set()
    try:
        with zipfile.ZipFile(validation_path) as handle:
            for info in handle.infolist():
                normalized = info.filename.rstrip("/")
                if info.is_dir():
                    parts = _safe_member_parts(normalized, "official-validation directory")
                    if parts == ("extracted_val",):
                        continue
                    if len(parts) == 2 and parts[0] == "extracted_val":
                        _validate_class_id(parts[1])
                        continue
                    raise ArchiveIndexError(
                        f"unexpected official-validation ZIP directory layout: "
                        f"{validation_path}!{info.filename}"
                    )
                if info.filename in seen_member_paths:
                    raise ArchiveIndexError(
                        f"duplicate official-validation ZIP member path: {info.filename}"
                    )
                seen_member_paths.add(info.filename)
                class_id, _ = _parse_validation_member(info.filename)
                classes.add(class_id)
    except zipfile.BadZipFile as exc:
        raise ArchiveIndexError(
            f"invalid official-validation ZIP archive: {validation_path}"
        ) from exc
    return classes


def _index_train_class(
    dataset_root: Path,
    *,
    zip_path: Path,
    tar_member_path: str,
    class_id: str,
    hash_raw_members: bool,
) -> list[_UnresolvedSample]:
    records: list[_UnresolvedSample] = []
    relative_zip = _relative_posix(zip_path, dataset_root)
    seen_inner_paths: set[str] = set()
    try:
        with zipfile.ZipFile(zip_path) as zip_handle:
            with zip_handle.open(tar_member_path) as compressed_tar:
                with tarfile.open(fileobj=compressed_tar, mode="r|gz") as tar_handle:
                    for member in tar_handle:
                        if member.isdir():
                            if member.name.rstrip("/") != class_id:
                                raise ArchiveIndexError(
                                    f"unexpected train TAR directory layout: "
                                    f"{relative_zip}!{tar_member_path}!{member.name}"
                                )
                            continue
                        if not member.isfile():
                            raise ArchiveIndexError(
                                f"unsupported train TAR member type: "
                                f"{relative_zip}!{tar_member_path}!{member.name}"
                            )
                        parts = _safe_member_parts(member.name, "train TAR member")
                        if len(parts) != 2 or parts[0] != class_id:
                            raise ArchiveIndexError(
                                f"unexpected train TAR member layout: "
                                f"{relative_zip}!{tar_member_path}!{member.name}"
                            )
                        filename = parts[1]
                        _validate_npz_filename(filename)
                        if member.name in seen_inner_paths:
                            raise ArchiveIndexError(
                                f"duplicate train TAR member path: "
                                f"{relative_zip}!{tar_member_path}!{member.name}"
                            )
                        seen_inner_paths.add(member.name)
                        digest = None
                        if hash_raw_members:
                            extracted = tar_handle.extractfile(member)
                            if extracted is None:
                                raise ArchiveIndexError(
                                    f"unreadable train TAR member: "
                                    f"{relative_zip}!{tar_member_path}!{member.name}"
                                )
                            digest, size = _sha256_stream(extracted)
                            if size != member.size:
                                raise ArchiveIndexError(
                                    f"train TAR member size mismatch for {member.name}: "
                                    f"header={member.size}, read={size}"
                                )
                        records.append(
                            _UnresolvedSample(
                                source_split="train",
                                class_id=class_id,
                                filename=filename,
                                source_archive_relative_path=relative_zip,
                                archive_member_path=tar_member_path,
                                nested_member_path=member.name,
                                raw_member_size_bytes=member.size,
                                raw_member_sha256=digest,
                            )
                        )
    except ArchiveIndexError:
        raise
    except (zipfile.BadZipFile, tarfile.TarError, OSError, EOFError) as exc:
        raise ArchiveIndexError(
            f"could not index train class locator {relative_zip}!{tar_member_path}: {exc}"
        ) from exc
    if not records:
        raise ArchiveIndexError(
            f"train class archive contains no NPZ samples: {relative_zip}!{tar_member_path}"
        )
    return records


def _index_validation_archive(
    dataset_root: Path,
    *,
    validation_path: Path,
    hash_raw_members: bool,
) -> list[_UnresolvedSample]:
    relative_zip = _relative_posix(validation_path, dataset_root)
    records: list[_UnresolvedSample] = []
    seen_member_paths: set[str] = set()
    try:
        with zipfile.ZipFile(validation_path) as handle:
            file_infos = sorted(
                (info for info in handle.infolist() if not info.is_dir()),
                key=lambda info: info.filename,
            )
            for info in file_infos:
                if info.filename in seen_member_paths:
                    raise ArchiveIndexError(
                        f"duplicate official-validation ZIP member path: {info.filename}"
                    )
                seen_member_paths.add(info.filename)
                class_id, filename = _parse_validation_member(info.filename)
                digest = None
                if hash_raw_members:
                    with handle.open(info) as stream:
                        digest, size = _sha256_stream(stream)
                    if size != info.file_size:
                        raise ArchiveIndexError(
                            f"official-validation member size mismatch for {info.filename}: "
                            f"header={info.file_size}, read={size}"
                        )
                records.append(
                    _UnresolvedSample(
                        source_split="validation",
                        class_id=class_id,
                        filename=filename,
                        source_archive_relative_path=relative_zip,
                        archive_member_path=info.filename,
                        nested_member_path=None,
                        raw_member_size_bytes=info.file_size,
                        raw_member_sha256=digest,
                    )
                )
    except ArchiveIndexError:
        raise
    except (zipfile.BadZipFile, OSError, EOFError) as exc:
        raise ArchiveIndexError(
            f"could not index official-validation archive {relative_zip}: {exc}"
        ) from exc
    if not records:
        raise ArchiveIndexError(
            f"official-validation archive contains no NPZ samples: {relative_zip}"
        )
    return records


def _validate_unique_samples(records: Sequence[_UnresolvedSample]) -> None:
    seen_ids: set[str] = set()
    seen_locators: set[str] = set()
    for record in records:
        if record.sample_id in seen_ids:
            raise ArchiveIndexError(f"duplicate stable sample ID: {record.sample_id}")
        seen_ids.add(record.sample_id)
        if record.source_locator in seen_locators:
            raise ArchiveIndexError(f"duplicate source locator: {record.source_locator}")
        seen_locators.add(record.source_locator)


def _parse_validation_member(member_path: str) -> tuple[str, str]:
    parts = _safe_member_parts(member_path, "official-validation ZIP member")
    if len(parts) != 3 or parts[0] != "extracted_val":
        raise ArchiveIndexError(
            f"unexpected official-validation ZIP member layout: {member_path}"
        )
    class_id, filename = parts[1], parts[2]
    _validate_class_id(class_id)
    _validate_npz_filename(filename)
    return class_id, filename


def _validate_npz_filename(filename: str) -> None:
    if PurePosixPath(filename).name != filename or not filename.endswith(".npz"):
        raise ArchiveIndexError(f"sample member filename must end with .npz: {filename!r}")


def _validate_class_id(class_id: str) -> None:
    if not CLASS_ID_PATTERN.fullmatch(class_id):
        raise ArchiveIndexError(f"invalid WordNet synset class ID: {class_id!r}")


def _safe_member_parts(member_path: str, description: str) -> tuple[str, ...]:
    if not isinstance(member_path, str) or not member_path:
        raise ArchiveIndexError(f"{description} must be a non-empty POSIX path")
    path = PurePosixPath(member_path)
    parts = path.parts
    if path.is_absolute() or not parts or any(part in {"", ".", ".."} for part in parts):
        raise ArchiveIndexError(f"unsafe {description}: {member_path!r}")
    return parts


def _train_part_number(filename: str) -> int:
    match = re.fullmatch(r"train_Part_([1-9]|10)\.zip", filename)
    if match is None:
        raise ArchiveIndexError(f"unexpected train archive filename: {filename}")
    return int(match.group(1))


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ArchiveIndexError(f"source path escapes dataset root: {path}") from exc


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"N-ImageNet mini {description} does not exist: {path}")


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        digest, _ = _sha256_stream(stream)
    return digest


def _sha256_stream(stream: BinaryIO) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(_HASH_CHUNK_SIZE)
        if not chunk:
            break
        hasher.update(chunk)
        size += len(chunk)
    return hasher.hexdigest(), size


__all__ = [
    "ARCHIVE_DIRECTORY",
    "ArchiveIndexError",
    "ArchiveSampleRecord",
    "BUNDLED_LIST_NAMES",
    "EXPECTED_CLASS_COUNT",
    "NImageNetMiniArchiveIndex",
    "RawContentDuplicate",
    "SourceFileChecksum",
    "TRAIN_ARCHIVE_NAMES",
    "VALIDATION_ARCHIVE_NAME",
    "build_archive_index",
    "compute_source_file_checksums",
    "find_raw_content_duplicates",
]
