"""Read-only, manifest-backed access to production N-ImageNet mini samples.

This adapter deliberately stops at raw-event decoding, D012 representation
rendering, and optional representation caching.  It does not import PyTorch,
construct a model, batch samples, augment data, or initialize CUDA.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

from ebackbone_v3.contracts import RawEventRecord
from ebackbone_v3.errors import DatasetError
from ebackbone_v3.n_imagenet_mini_index import (
    ARCHIVE_DIRECTORY,
    BUNDLED_LIST_NAMES,
    CLASS_ID_PATTERN,
    EXPECTED_CLASS_COUNT,
    TRAIN_ARCHIVE_NAMES,
    VALIDATION_ARCHIVE_NAME,
)
from ebackbone_v3.representations import (
    CACHE_SCHEMA_VERSION,
    CONTRACT_NAME,
    DATASET_RELEASE,
    DEFAULT_RENDERER_CONFIG,
    RendererConfig,
    RepresentationError,
    SourceIdentity,
    cache_path,
    compute_event_fingerprint,
    read_representation_cache,
    render_production_representations,
    representation_cache_key,
    write_representation_cache,
)
from ebackbone_v3.splits import (
    CHECKSUM_FILENAME,
    DATASET_RELEASE as SPLIT_DATASET_RELEASE,
    EXPECTED_ARTIFACT_FILENAMES,
    MANIFEST_FILENAMES,
    MANIFEST_SCHEMA_VERSION,
    PROTOCOL_ID,
    PROVENANCE_SCHEMA_VERSION,
    PROVENANCE_FILENAME,
)


ADAPTER_SCHEMA_VERSION = 1
DATASET_NAME = "N-ImageNet mini 100-class"
DEFAULT_N_IMAGENET_MINI_DATASET_ROOT = Path("/mnt/hdd1/datasets/event/n_imagenet")
PROJECT_SPLITS = ("train", "validation", "test")
SOURCE_SPLITS = ("train", "validation")
ProjectSplit = Literal["train", "validation", "test"]
_ROW_KEYS = frozenset(
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
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_EXPECTED_EVENT_DTYPE = np.dtype(
    [("x", "<u2"), ("y", "<u2"), ("t", "<u2"), ("p", "?")],
    align=False,
)
_SOURCE_HEIGHT = 480
_SOURCE_WIDTH = 640


@dataclass(frozen=True)
class ManifestRow:
    """One validated immutable-manifest row, with only archive-facing fields."""

    sample_id: str
    class_id: str
    class_index: int
    class_label: int
    split: str
    source_split: str
    source_archive: str
    source_members: tuple[str, ...]
    source_locator: str
    raw_content_size_bytes: int
    raw_content_sha256: str


@dataclass(frozen=True)
class ManifestSampleMetadata:
    """Structured provenance returned alongside one B0 or B1 tensor selection."""

    sample_id: str
    label: int
    class_index: int
    synset: str
    split: str
    source_split: str
    source_locator: str
    raw_payload_sha256: str
    raw_payload_size_bytes: int
    raw_fingerprint: str
    temporal_start: int
    temporal_end: int
    interval_closure: str
    event_count: int
    renderer_fingerprint: str
    representation_contract: str
    cache_key: str
    cache_status: str
    cache_path: str | None


@dataclass(frozen=True)
class ManifestSample:
    """Archive-native raw record plus the requested production tensor selection."""

    raw_events: RawEventRecord
    tensors: Mapping[str, np.ndarray]
    metadata: ManifestSampleMetadata


class ManifestBackedNImageNetMiniDataset:
    """Resolve one immutable project manifest without runtime split sampling.

    ``split`` is always required.  ``test.jsonl`` additionally requires
    ``allow_final_test=True`` so a final-test archive member cannot be opened by
    a role typo, a filename inference, or an evaluation-mode boolean.
    """

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        dataset_root: str | Path = DEFAULT_N_IMAGENET_MINI_DATASET_ROOT,
        baseline: Literal["b0", "b1"] = "b1",
        cache: Literal["off", "on"] = "off",
        cache_root: str | Path | None = None,
        split: ProjectSplit,
        allow_final_test: bool = False,
        renderer_config: RendererConfig = DEFAULT_RENDERER_CONFIG,
    ) -> None:
        _validate_access_request(split=split, allow_final_test=allow_final_test)
        if baseline not in {"b0", "b1"}:
            raise DatasetError("baseline must be 'b0' or 'b1'")
        if cache not in {"off", "on"}:
            raise DatasetError("cache must be 'off' or 'on'")
        if cache == "on" and cache_root is None:
            raise DatasetError("cache='on' requires an explicit cache_root")
        if renderer_config != DEFAULT_RENDERER_CONFIG:
            raise DatasetError(
                "the manifest-backed adapter accepts only the fixed D012 production renderer"
            )

        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.split = _resolve_project_split(self.manifest_path, split)
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if not self.dataset_root.is_dir():
            raise DatasetError(f"dataset_root does not exist or is not a directory: {self.dataset_root}")
        self.baseline = baseline
        self.cache = cache
        self.cache_root = (
            Path(cache_root).expanduser().resolve() if cache_root is not None else None
        )
        self.renderer_config = renderer_config
        self.renderer_fingerprint = renderer_provenance_fingerprint(renderer_config)
        self._rows = _load_immutable_manifest(self.manifest_path, expected_split=self.split)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> ManifestSample:
        return self.get(index)

    def get(self, index: int) -> ManifestSample:
        """Resolve one manifest row through raw-byte verification and D012 rendering."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("sample index must be an integer")
        if index < 0:
            raise IndexError("sample index must be non-negative")
        try:
            row = self._rows[index]
        except IndexError as exc:
            raise IndexError(f"sample index {index} is outside manifest length {len(self)}") from exc

        payload = _read_archive_payload(self.dataset_root, row)
        _validate_raw_payload(payload, row)
        fields = _decode_event_fields(payload, sample_id=row.sample_id)
        source = _source_identity(row, fields)
        raw_events = _raw_event_record(row, fields, source)
        bundle, cache_status, resolved_cache_path = self._load_or_render(fields, source)

        requested_names = (
            ("event_frame",)
            if self.baseline == "b0"
            else ("event_frame", "voxel_grid", "time_surface")
        )
        tensors = {
            name: _readonly_array(bundle.tensors()[name]) for name in requested_names
        }
        metadata = ManifestSampleMetadata(
            sample_id=row.sample_id,
            label=row.class_label,
            class_index=row.class_index,
            synset=row.class_id,
            split=row.split,
            source_split=row.source_split,
            source_locator=row.source_locator,
            raw_payload_sha256=row.raw_content_sha256,
            raw_payload_size_bytes=row.raw_content_size_bytes,
            raw_fingerprint=source.event_subset_id,
            temporal_start=source.temporal_start,
            temporal_end=source.temporal_end,
            interval_closure=source.interval_closure,
            event_count=source.event_count,
            renderer_fingerprint=self.renderer_fingerprint,
            representation_contract=CONTRACT_NAME,
            cache_key=bundle.cache_key,
            cache_status=cache_status,
            cache_path=(str(resolved_cache_path) if resolved_cache_path is not None else None),
        )
        return ManifestSample(
            raw_events=raw_events,
            tensors=MappingProxyType(tensors),
            metadata=metadata,
        )

    def _load_or_render(
        self,
        fields: Mapping[str, np.ndarray],
        source: SourceIdentity,
    ) -> tuple[Any, str, Path | None]:
        try:
            expected_key = representation_cache_key(
                source=source,
                config=self.renderer_config,
            )
        except RepresentationError as exc:
            raise DatasetError(f"could not derive representation cache provenance: {exc}") from exc

        if self.cache == "off":
            return _render(fields, source, self.renderer_config), "off", None

        assert self.cache_root is not None
        try:
            target = cache_path(self.cache_root, expected_key)
        except RepresentationError as exc:
            raise DatasetError(f"could not derive representation cache path: {exc}") from exc
        stale_entry = False
        if target.exists():
            if not target.is_file():
                raise DatasetError(f"representation cache target is not a regular file: {target}")
            try:
                cached = read_representation_cache(
                    target,
                    expected_source=source,
                    expected_config=self.renderer_config,
                )
            except RepresentationError:
                stale_entry = True
            else:
                return cached, "hit", target

        bundle = _render(fields, source, self.renderer_config)
        try:
            write_representation_cache(target, bundle)
        except (OSError, RepresentationError) as exc:
            raise DatasetError(f"could not write representation cache {target}: {exc}") from exc
        return bundle, ("stale_rebuilt" if stale_entry else "miss"), target


def renderer_provenance_fingerprint(
    config: RendererConfig = DEFAULT_RENDERER_CONFIG,
) -> str:
    """Return a renderer-only provenance fingerprint, independent of sample identity."""

    payload = {
        "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
        "representation_cache_schema_version": CACHE_SCHEMA_VERSION,
        "representation_contract": CONTRACT_NAME,
        "dataset_release": DATASET_RELEASE,
        "raw_contract": {
            "fields": {"x": "uint16", "y": "uint16", "t": "uint16", "p": "bool"},
            "spatial_resolution": [_SOURCE_HEIGHT, _SOURCE_WIDTH],
        },
        "renderer": config.to_dict(),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def open_dataset(
    *,
    manifest_dir: str | Path,
    split: ProjectSplit,
    allow_final_test: bool = False,
    dataset_root: str | Path = DEFAULT_N_IMAGENET_MINI_DATASET_ROOT,
    baseline: Literal["b0", "b1"] = "b1",
    cache: Literal["off", "on"] = "off",
    cache_root: str | Path | None = None,
    renderer_config: RendererConfig = DEFAULT_RENDERER_CONFIG,
) -> ManifestBackedNImageNetMiniDataset:
    """Open one fixed project role from its canonical immutable manifest.

    The role gate is evaluated before resolving the manifest directory or
    touching the dataset root.  Runtime callers cannot supply an archive path,
    member glob, predicate, sampler, or alternate manifest filename.
    """

    _validate_access_request(split=split, allow_final_test=allow_final_test)
    directory = Path(manifest_dir).expanduser().resolve()
    return ManifestBackedNImageNetMiniDataset(
        manifest_path=directory / MANIFEST_FILENAMES[split],
        dataset_root=dataset_root,
        baseline=baseline,
        cache=cache,
        cache_root=cache_root,
        split=split,
        allow_final_test=allow_final_test,
        renderer_config=renderer_config,
    )


def inspect_sample(
    *,
    manifest_path: str | Path,
    index: int,
    baseline: Literal["b0", "b1"],
    cache: Literal["off", "on"],
    dataset_root: str | Path = DEFAULT_N_IMAGENET_MINI_DATASET_ROOT,
    cache_root: str | Path | None = None,
    split: ProjectSplit,
    allow_final_test: bool = False,
) -> dict[str, object]:
    """Return a JSON-ready inspection report without model construction or execution."""

    dataset = ManifestBackedNImageNetMiniDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        baseline=baseline,
        cache=cache,
        cache_root=cache_root,
        split=split,
        allow_final_test=allow_final_test,
    )
    sample = dataset[index]
    return {
        "status": "PASS",
        "device": "cpu",
        "adapter": "manifest_backed_n_imagenet_mini",
        "baseline": baseline,
        "metadata": asdict(sample.metadata),
        "tensors": {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "range": {
                    "min": float(np.min(tensor)),
                    "max": float(np.max(tensor)),
                },
            }
            for name, tensor in sample.tensors.items()
        },
    }


def _resolve_project_split(
    manifest_path: Path,
    requested_split: ProjectSplit,
) -> str:
    by_filename = {filename: split for split, filename in MANIFEST_FILENAMES.items()}
    inferred_split = by_filename.get(manifest_path.name)
    if inferred_split is None:
        raise DatasetError(
            "manifest_path must name one immutable canonical manifest: "
            + ", ".join(sorted(by_filename))
        )
    if requested_split != inferred_split:
        raise DatasetError(
            f"explicit split {requested_split!r} does not match manifest filename {manifest_path.name!r}"
        )
    return requested_split


def _validate_access_request(*, split: object, allow_final_test: object) -> None:
    if not isinstance(split, str) or split not in PROJECT_SPLITS:
        raise DatasetError("split must be explicitly set to train, validation, or test")
    if not isinstance(allow_final_test, bool):
        raise DatasetError("allow_final_test must be an explicit boolean")
    if split == "test" and allow_final_test is not True:
        raise DatasetError(
            "final-test access requires split='test' and allow_final_test=True; "
            "no test manifest or archive member was read"
        )
    if split != "test" and allow_final_test:
        raise DatasetError("allow_final_test=True is valid only with split='test'")


def _load_immutable_manifest(manifest_path: Path, *, expected_split: str) -> tuple[ManifestRow, ...]:
    if not manifest_path.is_file():
        raise DatasetError(f"manifest does not exist or is not a regular file: {manifest_path}")
    directory = manifest_path.parent
    try:
        artifacts = tuple(directory.iterdir())
    except OSError as exc:
        raise DatasetError(f"could not inspect manifest directory {directory}: {exc}") from exc
    names = {path.name for path in artifacts}
    if names != EXPECTED_ARTIFACT_FILENAMES:
        raise DatasetError(
            "manifest directory must contain exactly the immutable supervised artifact set"
        )
    if any(path.is_symlink() or not path.is_file() for path in artifacts):
        raise DatasetError("immutable manifest artifacts must be regular non-symlink files")
    if manifest_path.name != MANIFEST_FILENAMES[expected_split]:
        raise DatasetError("manifest filename does not match the selected project split")

    manifest_bytes = _read_bytes(manifest_path, "manifest")
    provenance_path = directory / PROVENANCE_FILENAME
    provenance_bytes = _read_bytes(provenance_path, "provenance")
    checksum_bytes = _read_bytes(directory / CHECKSUM_FILENAME, "checksum file")
    checksums = _parse_checksum_file(checksum_bytes)
    if _canonical_checksum_file(checksums) != checksum_bytes:
        raise DatasetError("SHA256SUMS is not canonical deterministic checksum metadata")
    expected_checksum_names = set(EXPECTED_ARTIFACT_FILENAMES) - {CHECKSUM_FILENAME}
    if set(checksums) != expected_checksum_names:
        raise DatasetError("SHA256SUMS does not describe exactly the immutable artifact files")
    if checksums[manifest_path.name] != _sha256_bytes(manifest_bytes):
        raise DatasetError("manifest SHA-256 does not match SHA256SUMS")
    if checksums[PROVENANCE_FILENAME] != _sha256_bytes(provenance_bytes):
        raise DatasetError("provenance SHA-256 does not match SHA256SUMS")

    provenance = _decode_json_object(provenance_bytes, description="provenance.json")
    if _canonical_pretty_json(provenance) != provenance_bytes:
        raise DatasetError("provenance.json is not canonical deterministic JSON")
    _validate_runtime_provenance(provenance)
    manifest_files = provenance["manifest_files"]
    assert isinstance(manifest_files, dict)
    selected_metadata = manifest_files.get(expected_split)
    if not isinstance(selected_metadata, dict):
        raise DatasetError("provenance metadata for the selected manifest is invalid")
    if selected_metadata.get("filename") != manifest_path.name:
        raise DatasetError("provenance manifest filename does not match selected manifest")
    if selected_metadata.get("sha256") != _sha256_bytes(manifest_bytes):
        raise DatasetError("provenance manifest SHA-256 does not match selected manifest")
    if selected_metadata.get("size_bytes") != len(manifest_bytes):
        raise DatasetError("provenance manifest byte size does not match selected manifest")

    class_to_index = _parse_class_mapping(provenance.get("class_to_index"))
    rows: list[ManifestRow] = []
    seen_ids: set[str] = set()
    seen_locators: set[str] = set()
    seen_hashes: set[str] = set()
    previous_sample_id: str | None = None
    lines = manifest_bytes.splitlines(keepends=True)
    if not lines:
        raise DatasetError("selected manifest must contain at least one row")
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.endswith(b"\n"):
            raise DatasetError(f"manifest line {line_number} is missing its canonical newline")
        row_value = _decode_json_object(
            raw_line,
            description=f"{manifest_path.name} line {line_number}",
        )
        if _canonical_json(row_value) + b"\n" != raw_line:
            raise DatasetError(f"manifest line {line_number} is not canonical JSON")
        row = _parse_manifest_row(
            row_value,
            expected_split=expected_split,
            class_to_index=class_to_index,
        )
        if previous_sample_id is not None and row.sample_id <= previous_sample_id:
            raise DatasetError("manifest rows must be strictly sorted by stable sample_id")
        previous_sample_id = row.sample_id
        if row.sample_id in seen_ids:
            raise DatasetError(f"manifest contains duplicate sample_id: {row.sample_id}")
        if row.source_locator in seen_locators:
            raise DatasetError(f"manifest contains duplicate source_locator: {row.source_locator}")
        if row.raw_content_sha256 in seen_hashes:
            raise DatasetError("manifest contains an exact duplicate raw NPZ payload hash")
        seen_ids.add(row.sample_id)
        seen_locators.add(row.source_locator)
        seen_hashes.add(row.raw_content_sha256)
        rows.append(row)
    if selected_metadata.get("sample_count") != len(rows):
        raise DatasetError("provenance manifest sample count does not match selected manifest")
    project_counts = provenance["project_counts"]
    assert isinstance(project_counts, dict)
    if project_counts[expected_split] != len(rows):
        raise DatasetError("provenance project count does not match selected manifest")
    return tuple(rows)


def _validate_runtime_provenance(provenance: Mapping[str, Any]) -> None:
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
        raise DatasetError("provenance keys do not match the immutable supervised schema")
    if provenance["schema_version"] != PROVENANCE_SCHEMA_VERSION:
        raise DatasetError("provenance schema_version mismatch")
    if provenance["protocol_id"] != PROTOCOL_ID:
        raise DatasetError("provenance protocol_id does not match the immutable supervised protocol")
    if provenance["dataset_release"] != SPLIT_DATASET_RELEASE:
        raise DatasetError("provenance dataset release mismatch")
    for field in (
        "canonical_configuration_sha256",
        "generation_fingerprint_sha256",
        "source_catalog_sha256",
    ):
        if not _is_sha256(provenance[field]):
            raise DatasetError(f"provenance {field} is not a lowercase SHA-256 digest")

    generation = provenance["generation_parameters"]
    if not isinstance(generation, dict):
        raise DatasetError("provenance generation_parameters must be an object")
    if generation.get("source_to_project_mapping") != {
        "official_train": ["train", "validation"],
        "official_validation": ["test"],
    }:
        raise DatasetError("provenance source-to-project role mapping mismatch")
    expected_config_hash = _sha256_bytes(_canonical_json(generation))
    if provenance["canonical_configuration_sha256"] != expected_config_hash:
        raise DatasetError("provenance canonical configuration SHA-256 mismatch")
    fingerprint_payload = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "dataset_release": SPLIT_DATASET_RELEASE,
        "generation_parameters": generation,
        "class_to_index": provenance["class_to_index"],
        "source_files": provenance["source_files"],
        "source_counts": provenance["source_counts"],
        "source_per_class": provenance["source_per_class"],
        "source_catalog_sha256": provenance["source_catalog_sha256"],
        "release_contract": provenance["release_contract"],
    }
    expected_generation_fingerprint = _sha256_bytes(_canonical_json(fingerprint_payload))
    if provenance["generation_fingerprint_sha256"] != expected_generation_fingerprint:
        raise DatasetError("provenance generation fingerprint mismatch")

    manifest_files = provenance["manifest_files"]
    if not isinstance(manifest_files, dict) or set(manifest_files) != set(PROJECT_SPLITS):
        raise DatasetError("provenance manifest_files does not describe train, validation, and test")
    for split in PROJECT_SPLITS:
        metadata = manifest_files[split]
        if not isinstance(metadata, dict) or set(metadata) != {
            "filename",
            "sample_count",
            "size_bytes",
            "sha256",
        }:
            raise DatasetError(f"provenance manifest metadata is invalid for project split {split}")
        if metadata["filename"] != MANIFEST_FILENAMES[split]:
            raise DatasetError(f"provenance manifest filename mismatch for project split {split}")
        if (
            not _is_integer(metadata["sample_count"])
            or int(metadata["sample_count"]) <= 0
            or not _is_integer(metadata["size_bytes"])
            or int(metadata["size_bytes"]) <= 0
            or not _is_sha256(metadata["sha256"])
        ):
            raise DatasetError(f"provenance manifest checksum metadata is invalid for {split}")

    project_counts = provenance["project_counts"]
    if not isinstance(project_counts, dict) or set(project_counts) != set(PROJECT_SPLITS):
        raise DatasetError("provenance project_counts must describe train, validation, and test")
    if any(not _is_integer(value) or int(value) <= 0 for value in project_counts.values()):
        raise DatasetError("provenance project_counts values must be positive integers")
    if any(
        project_counts[split] != manifest_files[split]["sample_count"]
        for split in PROJECT_SPLITS
    ):
        raise DatasetError("provenance project counts do not match manifest metadata")

    expected_sources = {
        str(ARCHIVE_DIRECTORY / name): ("train_archive", True)
        for name in TRAIN_ARCHIVE_NAMES
    }
    expected_sources[str(ARCHIVE_DIRECTORY / VALIDATION_ARCHIVE_NAME)] = (
        "official_validation_archive",
        True,
    )
    expected_sources.update(
        {
            str(ARCHIVE_DIRECTORY / name): ("bundled_full_dataset_list", False)
            for name in BUNDLED_LIST_NAMES
        }
    )
    source_files = provenance["source_files"]
    if not isinstance(source_files, list) or len(source_files) != len(expected_sources):
        raise DatasetError("provenance source_files does not describe the fixed release")
    observed_sources: dict[str, tuple[object, object]] = {}
    previous_path: str | None = None
    for entry in source_files:
        if not isinstance(entry, dict) or set(entry) != {
            "relative_path",
            "role",
            "authoritative_for_membership",
            "size_bytes",
            "sha256",
        }:
            raise DatasetError("provenance source-file entry keys do not match the schema")
        relative_path = entry["relative_path"]
        if not isinstance(relative_path, str):
            raise DatasetError("provenance source-file path must be a string")
        _safe_posix_parts(relative_path, "provenance source-file path")
        if previous_path is not None and relative_path <= previous_path:
            raise DatasetError("provenance source files must be strictly path-sorted")
        previous_path = relative_path
        if (
            not _is_integer(entry["size_bytes"])
            or int(entry["size_bytes"]) <= 0
            or not _is_sha256(entry["sha256"])
            or not isinstance(entry["role"], str)
            or not isinstance(entry["authoritative_for_membership"], bool)
        ):
            raise DatasetError(f"invalid source-file provenance for {relative_path}")
        observed_sources[relative_path] = (
            entry["role"],
            entry["authoritative_for_membership"],
        )
    if observed_sources != expected_sources:
        raise DatasetError("provenance source-file roles or authority mismatch")


def _parse_manifest_row(
    value: Mapping[str, Any],
    *,
    expected_split: str,
    class_to_index: Mapping[str, int],
) -> ManifestRow:
    if set(value) != _ROW_KEYS:
        raise DatasetError("manifest row keys do not match the immutable schema")
    if (
        not _is_integer(value["schema_version"])
        or value["schema_version"] != MANIFEST_SCHEMA_VERSION
    ):
        raise DatasetError("manifest row schema_version does not match the immutable schema")
    for key in (
        "sample_id",
        "class_id",
        "split",
        "source_split",
        "source_archive",
        "source_locator",
        "raw_content_sha256",
    ):
        if not isinstance(value[key], str) or not value[key]:
            raise DatasetError(f"manifest row {key} must be a non-empty string")
    for key in ("class_index", "class_label", "raw_content_size_bytes"):
        if not _is_integer(value[key]):
            raise DatasetError(f"manifest row {key} must be an integer")
    if value["split"] != expected_split:
        raise DatasetError("manifest row project split does not match selected manifest")
    source_split = str(value["source_split"])
    if source_split not in SOURCE_SPLITS:
        raise DatasetError("manifest row source_split must be train or validation")
    if source_split == "validation" and expected_split != "test":
        raise DatasetError("official-validation source rows must have project split test")
    if source_split == "train" and expected_split not in {"train", "validation"}:
        raise DatasetError("source-train rows must have project split train or validation")

    class_id = str(value["class_id"])
    if CLASS_ID_PATTERN.fullmatch(class_id) is None:
        raise DatasetError("manifest row class_id must be a WordNet synset")
    if class_id not in class_to_index:
        raise DatasetError("manifest row class_id is absent from provenance class mapping")
    class_index = int(value["class_index"])
    class_label = int(value["class_label"])
    if class_index != class_to_index[class_id] or class_label != class_index:
        raise DatasetError("manifest row class index/label does not match its synset mapping")

    sample_id = str(value["sample_id"])
    sample_parts = _safe_posix_parts(sample_id, "sample_id")
    if len(sample_parts) != 3:
        raise DatasetError("manifest sample_id must be <source_split>/<synset>/<filename>.npz")
    if sample_parts[0] != source_split or sample_parts[1] != class_id:
        raise DatasetError("manifest sample_id does not agree with source_split and class_id")
    filename = sample_parts[2]
    if not filename.endswith(".npz") or PurePosixPath(filename).name != filename:
        raise DatasetError("manifest sample filename must be a single .npz name")

    source_archive = str(value["source_archive"])
    _safe_posix_parts(source_archive, "source_archive")
    raw_members = value["source_members"]
    if not isinstance(raw_members, list) or not raw_members or not all(
        isinstance(member, str) and member for member in raw_members
    ):
        raise DatasetError("manifest source_members must be a non-empty string list")
    source_members = tuple(str(member) for member in raw_members)
    for member in source_members:
        _safe_posix_parts(member, "source member")
        if "!" in member:
            raise DatasetError("manifest source member must not contain a locator separator")
    _validate_locator_semantics(
        source_split=source_split,
        class_id=class_id,
        filename=filename,
        source_archive=source_archive,
        source_members=source_members,
    )
    expected_locator = "!".join((source_archive, *source_members))
    if value["source_locator"] != expected_locator:
        raise DatasetError("manifest source_locator does not match source archive/member fields")

    raw_size = int(value["raw_content_size_bytes"])
    if raw_size <= 0:
        raise DatasetError("manifest raw_content_size_bytes must be positive")
    raw_hash = str(value["raw_content_sha256"])
    if len(raw_hash) != 64 or any(character not in _SHA256_CHARACTERS for character in raw_hash):
        raise DatasetError("manifest raw_content_sha256 must be a lowercase SHA-256 hex digest")
    return ManifestRow(
        sample_id=sample_id,
        class_id=class_id,
        class_index=class_index,
        class_label=class_label,
        split=expected_split,
        source_split=source_split,
        source_archive=source_archive,
        source_members=source_members,
        source_locator=expected_locator,
        raw_content_size_bytes=raw_size,
        raw_content_sha256=raw_hash,
    )


def _validate_locator_semantics(
    *,
    source_split: str,
    class_id: str,
    filename: str,
    source_archive: str,
    source_members: tuple[str, ...],
) -> None:
    if source_split == "train":
        matching = [
            archive_name
            for archive_name in TRAIN_ARCHIVE_NAMES
            if source_archive == str(ARCHIVE_DIRECTORY / archive_name)
        ]
        if len(matching) != 1 or len(source_members) != 2:
            raise DatasetError("source-train row has invalid archive/member locator semantics")
        part_number = matching[0].removeprefix("train_Part_").removesuffix(".zip")
        expected_members = (
            f"Part_{part_number}/{class_id}.tar.gz",
            f"{class_id}/{filename}",
        )
        if source_members != expected_members:
            raise DatasetError("source-train row locator semantics mismatch")
        return
    if source_split == "validation":
        expected_archive = str(ARCHIVE_DIRECTORY / VALIDATION_ARCHIVE_NAME)
        expected_members = (f"extracted_val/{class_id}/{filename}",)
        if source_archive != expected_archive or source_members != expected_members:
            raise DatasetError("official-validation row locator semantics mismatch")
        return
    raise AssertionError("source split was validated before locator semantics")


def _read_archive_payload(dataset_root: Path, row: ManifestRow) -> bytes:
    archive_path = _resolve_archive_path(dataset_root, row.source_archive)
    if not archive_path.is_file():
        raise DatasetError(f"declared source archive does not exist: {archive_path}")
    try:
        if row.source_split == "train":
            tar_member, npz_member = row.source_members
            with zipfile.ZipFile(archive_path) as zip_handle:
                with zip_handle.open(tar_member) as compressed_tar:
                    with tarfile.open(fileobj=compressed_tar, mode="r|gz") as tar_handle:
                        for member in tar_handle:
                            if not member.isfile() or member.name != npz_member:
                                continue
                            extracted = tar_handle.extractfile(member)
                            if extracted is None:
                                raise DatasetError(
                                    f"declared source member is not readable: {row.source_locator}"
                                )
                            return extracted.read()
            raise DatasetError(f"declared source member does not exist: {row.source_locator}")

        (npz_member,) = row.source_members
        with zipfile.ZipFile(archive_path) as zip_handle:
            info = zip_handle.getinfo(npz_member)
            if info.is_dir():
                raise DatasetError(f"declared source member is a directory: {row.source_locator}")
            return zip_handle.read(info)
    except DatasetError:
        raise
    except KeyError as exc:
        raise DatasetError(f"declared source member does not exist: {row.source_locator}") from exc
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise DatasetError(f"could not read declared source locator {row.source_locator}: {exc}") from exc


def _resolve_archive_path(dataset_root: Path, source_archive: str) -> Path:
    candidate = (dataset_root / PurePosixPath(source_archive)).resolve()
    try:
        candidate.relative_to(dataset_root)
    except ValueError as exc:
        raise DatasetError("declared source archive escapes dataset_root") from exc
    return candidate


def _validate_raw_payload(payload: bytes, row: ManifestRow) -> None:
    if len(payload) != row.raw_content_size_bytes:
        raise DatasetError(
            f"raw payload size mismatch for {row.sample_id}: expected "
            f"{row.raw_content_size_bytes}, observed {len(payload)}"
        )
    observed_hash = _sha256_bytes(payload)
    if observed_hash != row.raw_content_sha256:
        raise DatasetError(
            f"raw payload SHA-256 mismatch for {row.sample_id}: expected "
            f"{row.raw_content_sha256}, observed {observed_hash}"
        )


def _decode_event_fields(payload: bytes, *, sample_id: str) -> dict[str, np.ndarray]:
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            if archive.files != ["event_data"]:
                raise DatasetError(
                    f"{sample_id} NPZ keys must be exactly ['event_data']; observed {archive.files!r}"
                )
            event_data = archive["event_data"]
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"could not decode NPZ raw event payload for {sample_id}: {exc}") from exc
    if event_data.ndim != 1 or event_data.dtype != _EXPECTED_EVENT_DTYPE:
        raise DatasetError(
            f"{sample_id} event_data must be a one-dimensional packed x,y,t,p array; "
            f"observed shape={event_data.shape}, dtype={event_data.dtype.descr!r}"
        )
    fields = {name: event_data[name].copy() for name in ("x", "y", "t", "p")}
    _validate_event_fields(fields, sample_id=sample_id)
    return {name: _readonly_array(array) for name, array in fields.items()}


def _validate_event_fields(fields: Mapping[str, np.ndarray], *, sample_id: str) -> None:
    if set(fields) != {"x", "y", "t", "p"}:
        raise DatasetError("decoded raw fields must be exactly x,y,t,p")
    expected_dtypes = (np.dtype("<u2"), np.dtype("<u2"), np.dtype("<u2"), np.dtype("?"))
    arrays = tuple(np.asarray(fields[name]) for name in ("x", "y", "t", "p"))
    for name, array, dtype in zip(("x", "y", "t", "p"), arrays, expected_dtypes):
        if array.ndim != 1 or array.dtype != dtype:
            raise DatasetError(f"{sample_id} raw field {name} must be one-dimensional {dtype}")
    lengths = {array.shape[0] for array in arrays}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise DatasetError(f"{sample_id} raw x,y,t,p arrays must be non-empty and equal length")
    x, y, t, _ = arrays
    if np.any(x >= _SOURCE_WIDTH) or np.any(y >= _SOURCE_HEIGHT):
        raise DatasetError(f"{sample_id} has coordinates outside zero-based 480x640 bounds")
    if len(t) > 1 and not bool(np.all(t[1:] >= t[:-1])):
        raise DatasetError(f"{sample_id} timestamps must be nondecreasing")


def _source_identity(row: ManifestRow, fields: Mapping[str, np.ndarray]) -> SourceIdentity:
    timestamps = np.asarray(fields["t"])
    try:
        return SourceIdentity(
            sample_id=row.sample_id,
            split=row.source_split,
            event_subset_id=compute_event_fingerprint(fields),
            temporal_start=int(timestamps[0]),
            temporal_end=int(timestamps[-1]),
            interval_closure="[]",
            event_count=int(timestamps.shape[0]),
            raw_content_sha256=row.raw_content_sha256,
            project_split=row.split,
        )
    except RepresentationError as exc:
        raise DatasetError(f"invalid decoded raw-event identity for {row.sample_id}: {exc}") from exc


def _raw_event_record(
    row: ManifestRow,
    fields: Mapping[str, np.ndarray],
    source: SourceIdentity,
) -> RawEventRecord:
    return RawEventRecord(
        sample_id=row.sample_id,
        fields=MappingProxyType(dict(fields)),
        field_roles={"x": "x", "y": "y", "timestamp": "t", "polarity": "p"},
        storage_layout=(
            "NPZ key 'event_data': packed one-dimensional NumPy structured array, "
            "field order x,y,t,p, dtype [('x','<u2'),('y','<u2'),('t','<u2'),('p','|b1')], "
            "itemsize 7 bytes"
        ),
        timestamp_unit="microsecond",
        timestamp_ordering="nondecreasing",
        coordinate_convention="zero_based_xy",
        polarity_encoding="false/true",
        spatial_resolution=(_SOURCE_HEIGHT, _SOURCE_WIDTH),
        temporal_start=source.temporal_start,
        temporal_end=source.temporal_end,
        interval_closure=source.interval_closure,
        event_count_before_filter=source.event_count,
        event_count_after_filter=source.event_count,
        filtering="none; the complete stored event_data array is used",
    )


def _render(
    fields: Mapping[str, np.ndarray],
    source: SourceIdentity,
    config: RendererConfig,
) -> Any:
    try:
        return render_production_representations(fields, source=source, config=config)
    except RepresentationError as exc:
        raise DatasetError(f"could not render production representations: {exc}") from exc


def _parse_class_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise DatasetError("provenance class_to_index must be a non-empty object")
    mapping: dict[str, int] = {}
    for class_id, index in value.items():
        if not isinstance(class_id, str) or CLASS_ID_PATTERN.fullmatch(class_id) is None:
            raise DatasetError("provenance class_to_index has an invalid synset")
        if not _is_integer(index):
            raise DatasetError("provenance class_to_index values must be integers")
        mapping[class_id] = int(index)
    if set(mapping.values()) != set(range(len(mapping))):
        raise DatasetError("provenance class_to_index must be a contiguous zero-based bijection")
    if len(mapping) != EXPECTED_CLASS_COUNT:
        raise DatasetError(
            f"provenance class_to_index must contain {EXPECTED_CLASS_COUNT} N-ImageNet mini classes"
        )
    expected_mapping = {
        class_id: index for index, class_id in enumerate(sorted(mapping))
    }
    if mapping != expected_mapping:
        raise DatasetError("provenance class_to_index must use lexicographic synset order")
    return mapping


def _parse_checksum_file(payload: bytes) -> dict[str, str]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise DatasetError("SHA256SUMS must be ASCII") from exc
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split("  ")
        if len(parts) != 2 or not parts[1] or "/" in parts[1] or "\\" in parts[1]:
            raise DatasetError(f"SHA256SUMS line {line_number} is malformed")
        digest, name = parts
        if len(digest) != 64 or any(character not in _SHA256_CHARACTERS for character in digest):
            raise DatasetError(f"SHA256SUMS line {line_number} has an invalid SHA-256 digest")
        if name in checksums:
            raise DatasetError(f"SHA256SUMS names {name!r} more than once")
        checksums[name] = digest
    if not checksums:
        raise DatasetError("SHA256SUMS must not be empty")
    return checksums


def _canonical_checksum_file(checksums: Mapping[str, str]) -> bytes:
    return "".join(
        f"{checksums[name]}  {name}\n" for name in sorted(checksums)
    ).encode("ascii")


def _decode_json_object(payload: bytes, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                DatasetError(f"{description} contains non-standard numeric constant {token!r}")
            ),
        )
    except DatasetError:
        raise
    except UnicodeDecodeError as exc:
        raise DatasetError(f"{description} is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(
            f"{description} is invalid JSON at line {exc.lineno} column {exc.colno}"
        ) from exc
    if not isinstance(value, dict):
        raise DatasetError(f"{description} must contain a JSON object")
    return value


def _safe_posix_parts(value: str, description: str) -> tuple[str, ...]:
    if "\\" in value or "!" in value:
        raise DatasetError(f"{description} must use safe relative POSIX path syntax")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise DatasetError(f"{description} must be a safe relative POSIX path")
    return path.parts


def _readonly_array(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    result.setflags(write=False)
    return result


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise DatasetError(f"could not read {description} {path}: {exc}") from exc


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _canonical_pretty_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARACTERS for character in value)
    )


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "DATASET_NAME",
    "DEFAULT_N_IMAGENET_MINI_DATASET_ROOT",
    "ManifestBackedNImageNetMiniDataset",
    "ManifestRow",
    "ManifestSample",
    "ManifestSampleMetadata",
    "inspect_sample",
    "open_dataset",
    "renderer_provenance_fingerprint",
]
