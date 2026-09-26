"""Immutable supervised split manifests for the N-ImageNet mini release.

This module indexes source archives only.  It does not decode event arrays,
render representations, create caches, or perform any model computation.
Dataset-native source split names and project evaluation roles are stored in
separate fields so the official validation source can be reserved as project
``test`` without fabricating a dataset-native test split.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping, Sequence

from ebackbone_v3.errors import SplitError
from ebackbone_v3.n_imagenet_mini_index import (
    ARCHIVE_DIRECTORY,
    ArchiveSampleRecord,
    BUNDLED_LIST_NAMES,
    EXPECTED_CLASS_COUNT,
    NImageNetMiniArchiveIndex,
    TRAIN_ARCHIVE_NAMES,
    VALIDATION_ARCHIVE_NAME,
    build_archive_index,
)


CONFIG_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_VERSION = 1
PROTOCOL_ID = "n-imagenet-mini-supervised-v1"
DATASET_RELEASE = "N-ImageNet mini 100-class original train/validation release"
SELECTION_ALGORITHM = "sha256-rank-per-class-v1"
SELECTION_DOMAIN = b"ebackbone-v3/n-imagenet-mini/internal-validation/v1\0"
DEFAULT_SEED = 20260715
DEFAULT_INTERNAL_VALIDATION_PER_CLASS = 50
EXPECTED_OFFICIAL_TRAIN_SAMPLES = 129_395
EXPECTED_OFFICIAL_VALIDATION_SAMPLES = 5_000
EXPECTED_OFFICIAL_VALIDATION_PER_CLASS = 50
PROJECT_SPLITS = ("train", "validation", "test")
MANIFEST_FILENAMES = MappingProxyType(
    {
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
    }
)
PROVENANCE_FILENAME = "provenance.json"
CHECKSUM_FILENAME = "SHA256SUMS"
EXPECTED_ARTIFACT_FILENAMES = frozenset(
    (*MANIFEST_FILENAMES.values(), PROVENANCE_FILENAME, CHECKSUM_FILENAME)
)
_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "dataset_root",
        "manifest_dir",
        "seed",
        "internal_validation_per_class",
    }
)
_MANIFEST_ROW_KEYS = frozenset(
    {
        "schema_version",
        "sample_id",
        "class_id",
        "class_index",
        "class_label",
        "split",
        "source_split",
        "source_archive",
        "source_members",
        "source_locator",
        "raw_content_size_bytes",
        "raw_content_sha256",
    }
)
_CLASS_ID_PATTERN = re.compile(r"^n[0-9]{8}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SplitBuildConfig:
    """Resolved production split-build configuration."""

    dataset_root: Path
    manifest_dir: Path
    seed: int
    internal_validation_per_class: int


@dataclass(frozen=True)
class ProjectSplitRecord:
    """One archive-native sample plus its project evaluation role."""

    source: ArchiveSampleRecord
    split: str

    def to_manifest_dict(self) -> dict[str, object]:
        raw_digest = self.source.raw_member_sha256
        if raw_digest is None:
            raise SplitError(
                f"raw-content SHA-256 is missing for {self.source.sample_id}; "
                "production split manifests require full content hashing"
            )
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "sample_id": self.source.sample_id,
            "class_id": self.source.class_id,
            "class_index": self.source.class_index,
            "class_label": self.source.class_index,
            "split": self.split,
            "source_split": self.source.source_split,
            "source_archive": self.source.source_archive_relative_path,
            "source_members": list(self.source.source_member_paths),
            "source_locator": self.source.source_locator,
            "raw_content_size_bytes": self.source.raw_member_size_bytes,
            "raw_content_sha256": raw_digest,
        }


def load_split_build_config(config_path: str | Path) -> SplitBuildConfig:
    """Load a strict JSON configuration and resolve paths relative to it."""

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise SplitError(f"split configuration does not exist: {path}")
    try:
        root = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonstandard_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise SplitError(f"split configuration is not UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SplitError(
            f"invalid split configuration JSON at line {exc.lineno} column {exc.colno}: {path}"
        ) from exc
    except ValueError as exc:
        raise SplitError(f"invalid split configuration JSON value in {path}: {exc}") from exc
    if not isinstance(root, dict):
        raise SplitError("split configuration root must be a JSON object")
    unknown = sorted(set(root) - _CONFIG_KEYS)
    missing = sorted(_CONFIG_KEYS - set(root))
    if unknown:
        raise SplitError(f"unsupported split configuration keys: {', '.join(unknown)}")
    if missing:
        raise SplitError(f"missing split configuration keys: {', '.join(missing)}")
    if root["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise SplitError(
            f"split configuration schema_version must be {CONFIG_SCHEMA_VERSION}"
        )
    dataset_root = _resolve_config_path(root["dataset_root"], path, "dataset_root")
    manifest_dir = _resolve_config_path(root["manifest_dir"], path, "manifest_dir")
    seed = _require_integer(root["seed"], "seed", minimum=0)
    quota = _require_integer(
        root["internal_validation_per_class"],
        "internal_validation_per_class",
        minimum=1,
    )
    return SplitBuildConfig(
        dataset_root=dataset_root,
        manifest_dir=manifest_dir,
        seed=seed,
        internal_validation_per_class=quota,
    )


def build_splits(config_path: str | Path) -> dict[str, object]:
    """Index the fixed real release and atomically publish immutable manifests."""

    config = load_split_build_config(config_path)
    try:
        index = build_archive_index(
            config.dataset_root,
            hash_raw_members=True,
            hash_source_files=True,
        )
    except FileNotFoundError as exc:
        raise SplitError(f"could not index the configured dataset release: {exc}") from exc
    return publish_split_manifests(
        index,
        manifest_dir=config.manifest_dir,
        seed=config.seed,
        internal_validation_per_class=config.internal_validation_per_class,
        enforce_official_release=True,
    )


def assign_project_splits(
    index: NImageNetMiniArchiveIndex,
    *,
    seed: int,
    internal_validation_per_class: int,
) -> tuple[ProjectSplitRecord, ...]:
    """Assign project roles using stable hash ranking, independent of input order."""

    seed = _require_integer(seed, "seed", minimum=0)
    quota = _require_integer(
        internal_validation_per_class,
        "internal_validation_per_class",
        minimum=1,
    )
    by_class: dict[str, list[ArchiveSampleRecord]] = defaultdict(list)
    official_validation: list[ArchiveSampleRecord] = []
    for record in index.samples:
        if record.source_split == "train":
            by_class[record.class_id].append(record)
        elif record.source_split == "validation":
            official_validation.append(record)
        else:
            raise SplitError(
                f"unsupported source split for {record.sample_id}: {record.source_split!r}"
            )
    expected_classes = tuple(class_id for class_id, _ in index.class_to_index)
    if set(by_class) != set(expected_classes):
        raise SplitError("official training source does not expose the indexed 100-class set")
    selected_validation_ids: set[str] = set()
    for class_id in expected_classes:
        candidates = by_class[class_id]
        if len(candidates) <= quota:
            raise SplitError(
                f"class {class_id} has {len(candidates)} official-training samples; "
                f"must exceed internal validation quota {quota}"
            )
        ranked = sorted(
            candidates,
            key=lambda record: (
                _selection_rank(record.sample_id, seed),
                record.sample_id,
            ),
        )
        selected_validation_ids.update(
            record.sample_id for record in ranked[:quota]
        )

    assigned: list[ProjectSplitRecord] = []
    for record in index.samples:
        if record.source_split == "validation":
            split = "test"
        elif record.sample_id in selected_validation_ids:
            split = "validation"
        else:
            split = "train"
        assigned.append(ProjectSplitRecord(source=record, split=split))
    assigned.sort(key=lambda record: (PROJECT_SPLITS.index(record.split), record.source.sample_id))
    return tuple(assigned)


def publish_split_manifests(
    index: NImageNetMiniArchiveIndex,
    *,
    manifest_dir: str | Path,
    seed: int,
    internal_validation_per_class: int,
    enforce_official_release: bool = True,
) -> dict[str, object]:
    """Generate canonical bytes and publish once without overwriting any artifact.

    ``enforce_official_release=False`` exists for compact synthetic contract
    fixtures.  The user-facing ``build-splits`` command always enforces the
    verified 129,395/5,000 production release counts.
    """

    target = Path(manifest_dir).expanduser().resolve()
    files = _generate_artifact_bytes(
        index,
        seed=seed,
        internal_validation_per_class=internal_validation_per_class,
        enforce_official_release=enforce_official_release,
    )
    publication = _publish_immutable(target, files)
    report = verify_splits(target)
    report["publication"] = publication
    return report


def verify_splits(manifest_dir: str | Path) -> dict[str, object]:
    """Verify manifest bytes, provenance, counts, identities, and split policy."""

    directory = Path(manifest_dir).expanduser().resolve()
    _validate_artifact_directory(directory)
    artifact_bytes = {
        name: _read_regular_file(directory / name) for name in EXPECTED_ARTIFACT_FILENAMES
    }
    _verify_checksum_file(artifact_bytes)
    provenance = _decode_json_object(
        artifact_bytes[PROVENANCE_FILENAME],
        description=PROVENANCE_FILENAME,
    )
    if _canonical_json_bytes(provenance, pretty=True) != artifact_bytes[PROVENANCE_FILENAME]:
        raise SplitError("provenance.json is not canonical deterministic JSON")
    _verify_provenance_shape(provenance)

    class_to_index = _parse_class_mapping(provenance["class_to_index"])
    generation = _require_mapping(provenance["generation_parameters"], "generation_parameters")
    seed = _require_integer(generation.get("seed"), "generation_parameters.seed", minimum=0)
    quota = _require_integer(
        generation.get("internal_validation_per_class"),
        "generation_parameters.internal_validation_per_class",
        minimum=1,
    )
    if generation != _generation_parameters(seed, quota):
        raise SplitError("generation_parameters do not match the supported protocol")
    expected_config_hash = _sha256_bytes(_canonical_json_bytes(generation, pretty=False))
    if provenance["canonical_configuration_sha256"] != expected_config_hash:
        raise SplitError("canonical configuration SHA-256 does not match generation_parameters")

    rows_by_split: dict[str, list[dict[str, object]]] = {}
    manifest_metadata = _require_mapping(provenance["manifest_files"], "manifest_files")
    if set(manifest_metadata) != set(PROJECT_SPLITS):
        raise SplitError("provenance manifest_files must describe train, validation, and test")
    for split in PROJECT_SPLITS:
        filename = MANIFEST_FILENAMES[split]
        payload = artifact_bytes[filename]
        metadata = _require_mapping(manifest_metadata[split], f"manifest_files.{split}")
        if metadata.get("filename") != filename:
            raise SplitError(f"manifest filename mismatch for project split {split}")
        if metadata.get("sha256") != _sha256_bytes(payload):
            raise SplitError(f"manifest SHA-256 mismatch for project split {split}")
        if metadata.get("size_bytes") != len(payload):
            raise SplitError(f"manifest byte-size mismatch for project split {split}")
        rows = _decode_manifest_rows(payload, filename=filename, expected_split=split)
        if metadata.get("sample_count") != len(rows):
            raise SplitError(f"manifest sample-count mismatch for project split {split}")
        rows_by_split[split] = rows

    audit = _audit_rows(
        rows_by_split,
        class_to_index=class_to_index,
        seed=seed,
        internal_validation_per_class=quota,
    )
    _compare_audit_to_provenance(audit, provenance)

    catalog_hash = _source_catalog_sha256(
        [row for split in PROJECT_SPLITS for row in rows_by_split[split]]
    )
    if provenance["source_catalog_sha256"] != catalog_hash:
        raise SplitError("source catalog SHA-256 does not match manifest rows")
    fingerprint_payload = _generation_fingerprint_payload(
        generation_parameters=generation,
        class_to_index=class_to_index,
        source_files=provenance["source_files"],
        source_counts=provenance["source_counts"],
        source_per_class=provenance["source_per_class"],
        source_catalog_sha256=catalog_hash,
        release_contract=provenance["release_contract"],
    )
    expected_fingerprint = _sha256_bytes(
        _canonical_json_bytes(fingerprint_payload, pretty=False)
    )
    if provenance["generation_fingerprint_sha256"] != expected_fingerprint:
        raise SplitError("generation fingerprint does not match provenance inputs")

    manifest_hashes = {
        split: {
            "path": str(directory / MANIFEST_FILENAMES[split]),
            "sha256": _sha256_bytes(artifact_bytes[MANIFEST_FILENAMES[split]]),
        }
        for split in PROJECT_SPLITS
    }
    manifest_hashes["provenance"] = {
        "path": str(directory / PROVENANCE_FILENAME),
        "sha256": _sha256_bytes(artifact_bytes[PROVENANCE_FILENAME]),
    }
    return {
        "status": "PASS",
        "protocol_id": PROTOCOL_ID,
        "manifest_dir": str(directory),
        "generation_fingerprint_sha256": expected_fingerprint,
        "seed": seed,
        "internal_validation_per_class": quota,
        "sample_counts": audit["project_counts"],
        "class_count": len(class_to_index),
        "classes_in_every_split": True,
        "pairwise_sample_id_disjoint": True,
        "official_validation_project_role": "test",
        "duplicate_audit": audit["duplicate_audit"],
        "manifest_files": manifest_hashes,
    }


def _generate_artifact_bytes(
    index: NImageNetMiniArchiveIndex,
    *,
    seed: int,
    internal_validation_per_class: int,
    enforce_official_release: bool,
) -> dict[str, bytes]:
    _validate_index_for_manifest(index)
    assignments = assign_project_splits(
        index,
        seed=seed,
        internal_validation_per_class=internal_validation_per_class,
    )
    rows_by_split: dict[str, list[dict[str, object]]] = {split: [] for split in PROJECT_SPLITS}
    for assignment in assignments:
        rows_by_split[assignment.split].append(assignment.to_manifest_dict())
    for rows in rows_by_split.values():
        rows.sort(key=lambda row: str(row["sample_id"]))
    class_to_index = dict(index.class_to_index)
    audit = _audit_rows(
        rows_by_split,
        class_to_index=class_to_index,
        seed=seed,
        internal_validation_per_class=internal_validation_per_class,
    )
    _validate_release_counts(audit, enforce_official_release=enforce_official_release)

    manifest_bytes = {
        MANIFEST_FILENAMES[split]: _canonical_jsonl_bytes(rows_by_split[split])
        for split in PROJECT_SPLITS
    }
    generation = _generation_parameters(seed, internal_validation_per_class)
    source_files = [record.to_dict() for record in index.source_files]
    release_contract = {
        "production_release_enforced": enforce_official_release,
        "expected_class_count": EXPECTED_CLASS_COUNT,
        "expected_official_train_samples": (
            EXPECTED_OFFICIAL_TRAIN_SAMPLES
            if enforce_official_release
            else audit["source_counts"]["train"]
        ),
        "expected_official_validation_samples": (
            EXPECTED_OFFICIAL_VALIDATION_SAMPLES
            if enforce_official_release
            else audit["source_counts"]["validation"]
        ),
        "expected_official_validation_per_class": (
            EXPECTED_OFFICIAL_VALIDATION_PER_CLASS if enforce_official_release else None
        ),
    }
    catalog_hash = _source_catalog_sha256(
        [row for split in PROJECT_SPLITS for row in rows_by_split[split]]
    )
    fingerprint_payload = _generation_fingerprint_payload(
        generation_parameters=generation,
        class_to_index=class_to_index,
        source_files=source_files,
        source_counts=audit["source_counts"],
        source_per_class=audit["source_per_class"],
        source_catalog_sha256=catalog_hash,
        release_contract=release_contract,
    )
    provenance: dict[str, object] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "dataset_release": DATASET_RELEASE,
        "generation_parameters": generation,
        "canonical_configuration_sha256": _sha256_bytes(
            _canonical_json_bytes(generation, pretty=False)
        ),
        "generation_fingerprint_sha256": _sha256_bytes(
            _canonical_json_bytes(fingerprint_payload, pretty=False)
        ),
        "source_catalog_sha256": catalog_hash,
        "class_to_index": class_to_index,
        "source_files": source_files,
        "source_counts": audit["source_counts"],
        "source_per_class": audit["source_per_class"],
        "project_counts": audit["project_counts"],
        "project_per_class": audit["project_per_class"],
        "identity_audit": audit["identity_audit"],
        "duplicate_audit": audit["duplicate_audit"],
        "release_contract": release_contract,
        "manifest_files": {
            split: {
                "filename": MANIFEST_FILENAMES[split],
                "sample_count": len(rows_by_split[split]),
                "size_bytes": len(manifest_bytes[MANIFEST_FILENAMES[split]]),
                "sha256": _sha256_bytes(manifest_bytes[MANIFEST_FILENAMES[split]]),
            }
            for split in PROJECT_SPLITS
        },
    }
    provenance_bytes = _canonical_json_bytes(provenance, pretty=True)
    files = {**manifest_bytes, PROVENANCE_FILENAME: provenance_bytes}
    files[CHECKSUM_FILENAME] = _checksum_file_bytes(files)
    return files


def _generation_parameters(seed: int, quota: int) -> dict[str, object]:
    return {
        "seed": seed,
        "internal_validation_per_class": quota,
        "selection_algorithm": SELECTION_ALGORITHM,
        "selection_rank": (
            "SHA256(UTF8(selection_domain_utf8) || NUL || ASCII(decimal_seed) || "
            "NUL || UTF8(source_stable_sample_id)); "
            "sort by (digest, sample_id) within class"
        ),
        "selection_domain_utf8": SELECTION_DOMAIN[:-1].decode("utf-8"),
        "class_mapping": "lexicographically sorted WordNet synset ID to zero-based index",
        "stable_identity": "source_split/class_id/filename",
        "manifest_order": "lexicographic source-stable sample_id within each project split",
        "source_to_project_mapping": {
            "official_train": ["train", "validation"],
            "official_validation": ["test"],
        },
        "raw_content_hash": {
            "algorithm": "sha256",
            "definition": "exact uncompressed stored NPZ member bytes",
            "coverage": "every indexed sample",
            "semantic_equivalence_limit": (
                "differently encoded NPZ containers with equivalent decoded arrays are not detected"
            ),
        },
        "timestamps_in_output": False,
        "absolute_dataset_root_in_output": False,
    }


def _generation_fingerprint_payload(
    *,
    generation_parameters: object,
    class_to_index: object,
    source_files: object,
    source_counts: object,
    source_per_class: object,
    source_catalog_sha256: object,
    release_contract: object,
) -> dict[str, object]:
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "dataset_release": DATASET_RELEASE,
        "generation_parameters": generation_parameters,
        "class_to_index": class_to_index,
        "source_files": source_files,
        "source_counts": source_counts,
        "source_per_class": source_per_class,
        "source_catalog_sha256": source_catalog_sha256,
        "release_contract": release_contract,
    }


def _validate_index_for_manifest(index: NImageNetMiniArchiveIndex) -> None:
    class_to_index = dict(index.class_to_index)
    expected_mapping = {
        class_id: class_index
        for class_index, class_id in enumerate(sorted(class_to_index))
    }
    if len(class_to_index) != EXPECTED_CLASS_COUNT or class_to_index != expected_mapping:
        raise SplitError("archive index must use the canonical 100-class lexicographic mapping")
    if not index.samples:
        raise SplitError("archive index contains no samples")
    source_files = {record.relative_path: record for record in index.source_files}
    expected_source_paths = {
        *((ARCHIVE_DIRECTORY / name).as_posix() for name in TRAIN_ARCHIVE_NAMES),
        (ARCHIVE_DIRECTORY / VALIDATION_ARCHIVE_NAME).as_posix(),
        *((ARCHIVE_DIRECTORY / name).as_posix() for name in BUNDLED_LIST_NAMES),
    }
    if set(source_files) != expected_source_paths:
        raise SplitError("archive index source-file set differs from the fixed release")
    if any(
        source.sha256 is None or not _SHA256_PATTERN.fullmatch(source.sha256)
        for source in source_files.values()
    ):
        raise SplitError("all source archives and bundled lists require SHA-256 provenance")
    sample_ids: set[str] = set()
    locators: set[str] = set()
    raw_hashes: set[str] = set()
    for record in index.samples:
        if record.sample_id in sample_ids:
            raise SplitError(f"duplicate stable sample ID: {record.sample_id}")
        sample_ids.add(record.sample_id)
        if record.source_locator in locators:
            raise SplitError(f"duplicate source locator: {record.source_locator}")
        locators.add(record.source_locator)
        if record.raw_member_sha256 is None or not _SHA256_PATTERN.fullmatch(
            record.raw_member_sha256
        ):
            raise SplitError(f"missing or invalid raw-content SHA-256: {record.sample_id}")
        if record.raw_member_sha256 in raw_hashes:
            raise SplitError(
                "exact duplicate raw NPZ payload detected; manifests are not published: "
                f"{record.raw_member_sha256}"
            )
        raw_hashes.add(record.raw_member_sha256)
        if record.source_archive_sha256 is None or not _SHA256_PATTERN.fullmatch(
            record.source_archive_sha256
        ):
            raise SplitError(f"missing source-archive SHA-256: {record.sample_id}")
        source_file = source_files.get(record.source_archive_relative_path)
        if source_file is None or (
            record.source_archive_sha256 != source_file.sha256
            or record.source_archive_size_bytes != source_file.size_bytes
        ):
            raise SplitError(f"source-archive provenance mismatch: {record.sample_id}")
        if record.class_index != class_to_index.get(record.class_id):
            raise SplitError(f"class mapping mismatch for {record.sample_id}")
        if not record.sample_id.startswith(f"{record.source_split}/{record.class_id}/"):
            raise SplitError(f"source-stable sample ID mismatch: {record.sample_id}")


def _audit_rows(
    rows_by_split: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    class_to_index: Mapping[str, int],
    seed: int,
    internal_validation_per_class: int,
) -> dict[str, object]:
    if set(rows_by_split) != set(PROJECT_SPLITS):
        raise SplitError("row audit requires exactly train, validation, and test")
    expected_classes = set(class_to_index)
    all_ids: set[str] = set()
    all_locators: set[str] = set()
    all_raw_hashes: set[str] = set()
    ids_by_split: dict[str, set[str]] = {}
    project_per_class: dict[str, dict[str, int]] = {}
    source_per_class_counter: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
    }
    source_counts = Counter[str]()
    source_train_ids_by_class: dict[str, list[str]] = defaultdict(list)
    for split in PROJECT_SPLITS:
        ids: set[str] = set()
        per_class = Counter[str]()
        previous_id: str | None = None
        for row in rows_by_split[split]:
            _validate_manifest_row(row, expected_split=split, class_to_index=class_to_index)
            sample_id = str(row["sample_id"])
            locator = str(row["source_locator"])
            raw_digest = str(row["raw_content_sha256"])
            if previous_id is not None and sample_id <= previous_id:
                raise SplitError(f"{split} manifest rows are not strictly sorted by sample_id")
            previous_id = sample_id
            if sample_id in all_ids:
                raise SplitError(f"sample IDs are not pairwise disjoint: {sample_id}")
            all_ids.add(sample_id)
            ids.add(sample_id)
            if locator in all_locators:
                raise SplitError(f"duplicate source locator across manifests: {locator}")
            all_locators.add(locator)
            if raw_digest in all_raw_hashes:
                raise SplitError(
                    f"exact duplicate raw NPZ payload across manifest rows: {raw_digest}"
                )
            all_raw_hashes.add(raw_digest)
            class_id = str(row["class_id"])
            source_split = str(row["source_split"])
            per_class[class_id] += 1
            source_counts[source_split] += 1
            source_per_class_counter[source_split][class_id] += 1
            if source_split == "train":
                source_train_ids_by_class[class_id].append(sample_id)
        if set(per_class) != expected_classes:
            missing = sorted(expected_classes - set(per_class))
            extra = sorted(set(per_class) - expected_classes)
            raise SplitError(
                f"project split {split} does not contain exactly all 100 classes: "
                f"missing={missing}, extra={extra}"
            )
        ids_by_split[split] = ids
        project_per_class[split] = dict(sorted(per_class.items()))
    if len(ids_by_split["train"] & ids_by_split["validation"]) != 0:
        raise SplitError("project train and validation sample IDs overlap")
    if len(ids_by_split["train"] & ids_by_split["test"]) != 0:
        raise SplitError("project train and test sample IDs overlap")
    if len(ids_by_split["validation"] & ids_by_split["test"]) != 0:
        raise SplitError("project validation and test sample IDs overlap")
    if set(project_per_class["validation"].values()) != {internal_validation_per_class}:
        raise SplitError(
            "internal validation is not exactly class-stratified at the configured quota"
        )
    expected_validation_ids: set[str] = set()
    for class_id in sorted(expected_classes):
        ranked_ids = sorted(
            source_train_ids_by_class[class_id],
            key=lambda sample_id: (_selection_rank(sample_id, seed), sample_id),
        )
        expected_validation_ids.update(ranked_ids[:internal_validation_per_class])
    if ids_by_split["validation"] != expected_validation_ids:
        missing = sorted(expected_validation_ids - ids_by_split["validation"])
        unexpected = sorted(ids_by_split["validation"] - expected_validation_ids)
        raise SplitError(
            "internal validation membership does not match the configured deterministic "
            f"SHA-256 rank: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    project_counts = {
        split: len(rows_by_split[split]) for split in PROJECT_SPLITS
    }
    return {
        "project_counts": project_counts,
        "project_per_class": project_per_class,
        "source_counts": {
            source_split: source_counts[source_split]
            for source_split in ("train", "validation")
        },
        "source_per_class": {
            source_split: dict(sorted(source_per_class_counter[source_split].items()))
            for source_split in ("train", "validation")
        },
        "identity_audit": {
            "total_samples": len(all_ids),
            "unique_sample_ids": len(all_ids),
            "unique_source_locators": len(all_locators),
            "pairwise_project_split_sample_id_intersections": {
                "train_validation": 0,
                "train_test": 0,
                "validation_test": 0,
            },
        },
        "duplicate_audit": {
            "hash_algorithm": "sha256",
            "hash_definition": "exact uncompressed stored NPZ member bytes",
            "samples_hashed": len(all_raw_hashes),
            "unique_raw_content_hashes": len(all_raw_hashes),
            "exact_duplicate_groups": 0,
            "semantic_equivalence_limit": (
                "differently encoded NPZ containers with equivalent decoded arrays are not detected"
            ),
        },
    }


def _validate_release_counts(audit: Mapping[str, object], *, enforce_official_release: bool) -> None:
    if not enforce_official_release:
        return
    source_counts = _require_mapping(audit["source_counts"], "source_counts")
    if source_counts != {
        "train": EXPECTED_OFFICIAL_TRAIN_SAMPLES,
        "validation": EXPECTED_OFFICIAL_VALIDATION_SAMPLES,
    }:
        raise SplitError(
            "source release counts do not match the accepted N-ImageNet mini contract: "
            f"observed={dict(source_counts)}"
        )
    source_per_class = _require_mapping(audit["source_per_class"], "source_per_class")
    validation_counts = _require_mapping(
        source_per_class.get("validation"), "source_per_class.validation"
    )
    if set(validation_counts.values()) != {EXPECTED_OFFICIAL_VALIDATION_PER_CLASS}:
        raise SplitError("official validation must contain exactly 50 samples per class")


def _validate_manifest_row(
    row: Mapping[str, object],
    *,
    expected_split: str,
    class_to_index: Mapping[str, int],
) -> None:
    if set(row) != _MANIFEST_ROW_KEYS:
        raise SplitError(
            f"manifest row keys differ from schema: missing={sorted(_MANIFEST_ROW_KEYS - set(row))}, "
            f"extra={sorted(set(row) - _MANIFEST_ROW_KEYS)}"
        )
    if row["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise SplitError("manifest row schema_version mismatch")
    if row["split"] != expected_split:
        raise SplitError(f"manifest row split is not {expected_split!r}")
    class_id = row["class_id"]
    if not isinstance(class_id, str) or not _CLASS_ID_PATTERN.fullmatch(class_id):
        raise SplitError(f"invalid manifest class_id: {class_id!r}")
    expected_index = class_to_index.get(class_id)
    if row["class_index"] != expected_index or row["class_label"] != expected_index:
        raise SplitError(f"manifest class label/index mismatch for {class_id}")
    source_split = row["source_split"]
    if expected_split in {"train", "validation"} and source_split != "train":
        raise SplitError("project train/internal-validation rows must come from official train")
    if expected_split == "test" and source_split != "validation":
        raise SplitError("project test rows must come exclusively from official validation")
    sample_id = row["sample_id"]
    if not isinstance(sample_id, str):
        raise SplitError("manifest sample_id must be a string")
    sample_parts = _safe_posix_parts(sample_id, "sample_id")
    if len(sample_parts) != 3 or sample_parts[0] != source_split or sample_parts[1] != class_id:
        raise SplitError(f"source-stable sample identity mismatch: {sample_id}")
    if not sample_parts[2].endswith(".npz"):
        raise SplitError(f"sample identity is not an NPZ member: {sample_id}")
    source_archive = row["source_archive"]
    if not isinstance(source_archive, str):
        raise SplitError("source_archive must be a string")
    _safe_posix_parts(source_archive, "source_archive")
    source_members = row["source_members"]
    if not isinstance(source_members, list) or len(source_members) not in {1, 2}:
        raise SplitError("source_members must contain one or two member paths")
    for member in source_members:
        if not isinstance(member, str):
            raise SplitError("source member paths must be strings")
        _safe_posix_parts(member, "source member")
    filename = sample_parts[2]
    if source_split == "train":
        archive_name = PurePosixPath(source_archive).name
        match = re.fullmatch(r"train_Part_([1-9]|10)\.zip", archive_name)
        if (
            source_archive != (ARCHIVE_DIRECTORY / archive_name).as_posix()
            or archive_name not in TRAIN_ARCHIVE_NAMES
            or match is None
            or source_members
            != [
                f"Part_{match.group(1)}/{class_id}.tar.gz",
                f"{class_id}/{filename}",
            ]
        ):
            raise SplitError(f"train archive/member locator semantics mismatch for {sample_id}")
    elif source_split == "validation":
        if source_archive != (ARCHIVE_DIRECTORY / VALIDATION_ARCHIVE_NAME).as_posix() or source_members != [
            f"extracted_val/{class_id}/{filename}"
        ]:
            raise SplitError(
                f"official-validation archive/member locator semantics mismatch for {sample_id}"
            )
    expected_locator = "!".join((source_archive, *source_members))
    if row["source_locator"] != expected_locator:
        raise SplitError(f"source locator does not match its paths for {sample_id}")
    size = row["raw_content_size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise SplitError(f"raw content size must be positive for {sample_id}")
    digest = row["raw_content_sha256"]
    if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
        raise SplitError(f"invalid raw-content SHA-256 for {sample_id}")


def _compare_audit_to_provenance(
    audit: Mapping[str, object], provenance: Mapping[str, object]
) -> None:
    for key in (
        "project_counts",
        "project_per_class",
        "source_counts",
        "source_per_class",
        "identity_audit",
        "duplicate_audit",
    ):
        if provenance[key] != audit[key]:
            raise SplitError(f"provenance {key} does not match manifest rows")
    release = _require_mapping(provenance["release_contract"], "release_contract")
    expected_class_count = _require_integer(
        release.get("expected_class_count"), "release_contract.expected_class_count", minimum=1
    )
    if expected_class_count != EXPECTED_CLASS_COUNT:
        raise SplitError("release contract class count mismatch")
    source_counts = _require_mapping(audit["source_counts"], "source_counts")
    if release.get("expected_official_train_samples") != source_counts["train"]:
        raise SplitError("release contract official-train count mismatch")
    if release.get("expected_official_validation_samples") != source_counts["validation"]:
        raise SplitError("release contract official-validation count mismatch")
    if release.get("production_release_enforced") is True:
        if source_counts != {
            "train": EXPECTED_OFFICIAL_TRAIN_SAMPLES,
            "validation": EXPECTED_OFFICIAL_VALIDATION_SAMPLES,
        }:
            raise SplitError("production release counts do not match the accepted contract")
        if release.get("expected_official_validation_per_class") != 50:
            raise SplitError("production official-validation per-class contract mismatch")
        validation_counts = _require_mapping(
            _require_mapping(audit["source_per_class"], "source_per_class")["validation"],
            "source_per_class.validation",
        )
        if set(validation_counts.values()) != {50}:
            raise SplitError("production official validation is not exactly 50 samples per class")
    elif release.get("production_release_enforced") is not False:
        raise SplitError("release contract production_release_enforced must be boolean")


def _verify_provenance_shape(provenance: Mapping[str, object]) -> None:
    expected_keys = {
        "schema_version",
        "protocol_id",
        "dataset_release",
        "generation_parameters",
        "canonical_configuration_sha256",
        "generation_fingerprint_sha256",
        "source_catalog_sha256",
        "class_to_index",
        "source_files",
        "source_counts",
        "source_per_class",
        "project_counts",
        "project_per_class",
        "identity_audit",
        "duplicate_audit",
        "release_contract",
        "manifest_files",
    }
    if set(provenance) != expected_keys:
        raise SplitError("provenance keys differ from the supported schema")
    if provenance["schema_version"] != PROVENANCE_SCHEMA_VERSION:
        raise SplitError("provenance schema_version mismatch")
    if provenance["protocol_id"] != PROTOCOL_ID:
        raise SplitError("provenance protocol_id mismatch")
    if provenance["dataset_release"] != DATASET_RELEASE:
        raise SplitError("provenance dataset release mismatch")
    for field in (
        "canonical_configuration_sha256",
        "generation_fingerprint_sha256",
        "source_catalog_sha256",
    ):
        value = provenance[field]
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            raise SplitError(f"provenance {field} is not a SHA-256 digest")
    source_files = provenance["source_files"]
    if not isinstance(source_files, list) or len(source_files) != 13:
        raise SplitError("provenance must contain checksums for 11 archives and two lists")
    expected_sources = {
        (ARCHIVE_DIRECTORY / name).as_posix(): ("train_archive", True)
        for name in TRAIN_ARCHIVE_NAMES
    }
    expected_sources[(ARCHIVE_DIRECTORY / VALIDATION_ARCHIVE_NAME).as_posix()] = (
        "official_validation_archive",
        True,
    )
    expected_sources.update(
        {
            (ARCHIVE_DIRECTORY / name).as_posix(): (
                "bundled_full_dataset_list",
                False,
            )
            for name in BUNDLED_LIST_NAMES
        }
    )
    previous_path: str | None = None
    observed_sources: dict[str, tuple[object, object]] = {}
    for source in source_files:
        mapping = _require_mapping(source, "source_files entry")
        if set(mapping) != {
            "relative_path",
            "role",
            "authoritative_for_membership",
            "size_bytes",
            "sha256",
        }:
            raise SplitError("source-file provenance entry keys differ from schema")
        relative_path = mapping["relative_path"]
        if not isinstance(relative_path, str):
            raise SplitError("source-file relative_path must be a string")
        _safe_posix_parts(relative_path, "source-file relative_path")
        if previous_path is not None and relative_path <= previous_path:
            raise SplitError("source-file provenance is not strictly path-sorted")
        previous_path = relative_path
        digest = mapping["sha256"]
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise SplitError(f"invalid source-file SHA-256: {relative_path}")
        size = mapping["size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise SplitError(f"invalid source-file size: {relative_path}")
        authoritative = mapping["authoritative_for_membership"]
        if not isinstance(authoritative, bool):
            raise SplitError("authoritative_for_membership must be boolean")
        role = mapping["role"]
        if not isinstance(role, str):
            raise SplitError("source-file role must be a string")
        observed_sources[relative_path] = (role, authoritative)
    if observed_sources != expected_sources:
        raise SplitError(
            "source-file provenance paths, roles, or authority differ from the fixed release"
        )


def _parse_class_mapping(value: object) -> dict[str, int]:
    mapping = _require_mapping(value, "class_to_index")
    parsed: dict[str, int] = {}
    for class_id, class_index in mapping.items():
        if not isinstance(class_id, str) or not _CLASS_ID_PATTERN.fullmatch(class_id):
            raise SplitError(f"invalid class mapping key: {class_id!r}")
        parsed[class_id] = _require_integer(
            class_index, f"class_to_index.{class_id}", minimum=0
        )
    expected = {
        class_id: index for index, class_id in enumerate(sorted(parsed))
    }
    if len(parsed) != EXPECTED_CLASS_COUNT or parsed != expected:
        raise SplitError("class_to_index is not the canonical 100-class mapping")
    return parsed


def _source_catalog_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    hasher = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["sample_id"])):
        catalog_record = {
            key: row[key]
            for key in (
                "sample_id",
                "class_id",
                "class_index",
                "source_split",
                "source_archive",
                "source_members",
                "source_locator",
                "raw_content_size_bytes",
                "raw_content_sha256",
            )
        }
        hasher.update(_canonical_json_bytes(catalog_record, pretty=False))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _publish_immutable(target: Path, files: Mapping[str, bytes]) -> str:
    if set(files) != EXPECTED_ARTIFACT_FILENAMES:
        raise SplitError("internal artifact set differs from the immutable manifest schema")
    if target.exists():
        _validate_artifact_directory(target)
        differences = [
            name for name in sorted(files) if _read_regular_file(target / name) != files[name]
        ]
        if differences:
            raise SplitError(
                "refusing to regenerate or overwrite an existing manifest directory with "
                "different parameters, inputs, or bytes; conflicting files: "
                + ", ".join(differences)
            )
        return "verified_existing_unchanged"

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".{target.name}.build.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise SplitError(f"manifest publication lock already exists: {lock_path}") from exc
    os.close(lock_fd)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=parent))
    try:
        for name in sorted(files):
            path = staging / name
            with path.open("xb") as stream:
                stream.write(files[name])
                stream.flush()
                os.fsync(stream.fileno())
        if target.exists():
            raise SplitError(f"manifest directory appeared during publication: {target}")
        staging.rename(target)
        staging = Path()
    finally:
        if staging != Path() and staging.exists():
            shutil.rmtree(staging)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
    return "created"


def _validate_artifact_directory(directory: Path) -> None:
    if not directory.is_dir():
        raise SplitError(f"manifest directory does not exist: {directory}")
    observed = {path.name for path in directory.iterdir()}
    if observed != EXPECTED_ARTIFACT_FILENAMES:
        raise SplitError(
            "manifest directory must contain exactly the immutable artifact set: "
            f"missing={sorted(EXPECTED_ARTIFACT_FILENAMES - observed)}, "
            f"extra={sorted(observed - EXPECTED_ARTIFACT_FILENAMES)}"
        )
    for name in observed:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise SplitError(f"manifest artifact must be a regular non-symlink file: {path}")


def _verify_checksum_file(artifact_bytes: Mapping[str, bytes]) -> None:
    expected = _checksum_file_bytes(
        {name: payload for name, payload in artifact_bytes.items() if name != CHECKSUM_FILENAME}
    )
    if artifact_bytes[CHECKSUM_FILENAME] != expected:
        raise SplitError("SHA256SUMS does not match the immutable manifest artifacts")


def _checksum_file_bytes(files: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{_sha256_bytes(files[name])}  {name}\n" for name in sorted(files)
    ).encode("ascii")


def _decode_manifest_rows(
    payload: bytes, *, filename: str, expected_split: str
) -> list[dict[str, object]]:
    if not payload or not payload.endswith(b"\n"):
        raise SplitError(f"{filename} must be nonempty canonical JSONL ending in newline")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise SplitError(f"{filename}:{line_number} is blank")
        row = _decode_json_object(line, description=f"{filename}:{line_number}")
        if _canonical_json_bytes(row, pretty=False) != line:
            raise SplitError(f"{filename}:{line_number} is not canonical JSON")
        _validate_manifest_row(row, expected_split=expected_split, class_to_index={
            str(row.get("class_id")): row.get("class_index")  # structural precheck only
        })
        rows.append(row)
    return rows


def _decode_json_object(payload: bytes, *, description: str) -> dict[str, object]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SplitError(f"invalid UTF-8 JSON in {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise SplitError(f"{description} must contain a JSON object")
    return value


def _canonical_json_bytes(value: object, *, pretty: bool) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
    else:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    return (text + ("\n" if pretty else "")).encode("utf-8")


def _canonical_jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(row, pretty=False) + b"\n" for row in rows)


def _selection_rank(sample_id: str, seed: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(SELECTION_DOMAIN)
    hasher.update(str(seed).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(sample_id.encode("utf-8"))
    return hasher.digest()


def _resolve_config_path(value: object, config_path: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SplitError(f"split configuration {field} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _require_integer(value: object, field: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SplitError(f"{field} must be an integer >= {minimum}")
    return value


def _require_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise SplitError(f"{description} must be a JSON object")
    return value


def _safe_posix_parts(value: str, description: str) -> tuple[str, ...]:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise SplitError(f"unsafe {description}: {value!r}")
    return path.parts


def _read_regular_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise SplitError(f"manifest artifact must be a regular non-symlink file: {path}")
    return path.read_bytes()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric constant is not allowed: {value}")


__all__ = [
    "CHECKSUM_FILENAME",
    "CONFIG_SCHEMA_VERSION",
    "DEFAULT_INTERNAL_VALIDATION_PER_CLASS",
    "DEFAULT_SEED",
    "MANIFEST_FILENAMES",
    "PROTOCOL_ID",
    "ProjectSplitRecord",
    "SplitBuildConfig",
    "assign_project_splits",
    "build_splits",
    "load_split_build_config",
    "publish_split_manifests",
    "verify_splits",
]
