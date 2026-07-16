"""Versioned production representation contract for the B0/B1 inputs.

This module is intentionally independent of the real-data probe renderer.  It
contains no model, fusion, pooling, classification, augmentation, or split
logic.  Labels are not accepted by any public function.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


CONTRACT_NAME = "n_imagenet_mini_b0_b1_representations_v1"
DATASET_RELEASE = "N-ImageNet mini 100-class original train/validation release"
# Schema 3 binds optional exact stored-NPZ payload and project-split provenance
# into SourceIdentity. This deliberately invalidates older entries, whose
# provenance could not distinguish alternate NPZ encodings of otherwise equal
# x/y/t/p arrays or a changed project role for the same source record.
CACHE_SCHEMA_VERSION = 3
FRAME_CACHE_SCHEMA_VERSION = 1
SOURCE_HEIGHT = 480
SOURCE_WIDTH = 640
POLARITY_ORDER = ("negative", "positive")
REPRESENTATION_NAMES = ("event_frame", "voxel_grid", "time_surface")


class RepresentationError(ValueError):
    """Raised when production representation input or provenance is invalid."""


@dataclass(frozen=True)
class RendererConfig:
    """All parameters that can affect production tensor values."""

    output_height: int = SOURCE_HEIGHT
    output_width: int = SOURCE_WIDTH
    voxel_bins: int = 5
    time_surface_tau: float = 0.3
    frame_normalization: str = "log1p"
    voxel_normalization: str = "log1p"
    spatial_mapping: str = "native_identity"
    voxel_binning: str = "linear_temporal_interpolation"
    time_surface_background: float = 0.0
    polarity_order: tuple[str, str] = POLARITY_ORDER
    dtype: str = "float32"
    clipping: str = "none"

    def __post_init__(self) -> None:
        if self.output_height <= 0 or self.output_width <= 0:
            raise RepresentationError("output spatial dimensions must be positive")
        if self.voxel_bins < 2:
            raise RepresentationError("voxel_bins must be at least 2")
        if not np.isfinite(self.time_surface_tau) or self.time_surface_tau <= 0:
            raise RepresentationError("time_surface_tau must be finite and positive")
        if self.frame_normalization != "log1p" or self.voxel_normalization != "log1p":
            raise RepresentationError("production count normalization is fixed to log1p")
        if self.spatial_mapping != "native_identity":
            raise RepresentationError("unsupported production spatial mapping")
        if self.voxel_binning != "linear_temporal_interpolation":
            raise RepresentationError("unsupported production voxel binning")
        if self.time_surface_background != 0.0:
            raise RepresentationError("production time-surface background is fixed to zero")
        if tuple(self.polarity_order) != POLARITY_ORDER:
            raise RepresentationError("production polarity order is fixed to negative, positive")
        if self.dtype != "float32" or self.clipping != "none":
            raise RepresentationError("production tensors are unclipped float32")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["polarity_order"] = list(self.polarity_order)
        return result


@dataclass(frozen=True)
class SourceIdentity:
    """Label-free identity shared by every representation of one raw sample."""

    sample_id: str
    split: str
    event_subset_id: str
    temporal_start: int
    temporal_end: int
    interval_closure: str
    event_count: int
    raw_content_sha256: str | None = None
    project_split: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise RepresentationError("sample_id must be non-empty")
        if self.split not in {"train", "validation"}:
            raise RepresentationError("split must be train or validation")
        if not self.sample_id.startswith(f"{self.split}/"):
            raise RepresentationError("sample_id must begin with the declared split")
        if not self.event_subset_id.startswith("sha256:"):
            raise RepresentationError("event_subset_id must be a canonical sha256 identity")
        if self.temporal_end < self.temporal_start:
            raise RepresentationError("temporal_end must be greater than or equal to temporal_start")
        if self.interval_closure != "[]":
            raise RepresentationError("production N-ImageNet intervals use closed endpoints []")
        if self.event_count <= 0:
            raise RepresentationError("production rendering requires at least one event")
        if self.raw_content_sha256 is not None:
            if (
                len(self.raw_content_sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.raw_content_sha256)
            ):
                raise RepresentationError(
                    "raw_content_sha256 must be a lowercase SHA-256 hex digest when supplied"
                )
        if self.project_split is not None and self.project_split not in {
            "train",
            "validation",
            "test",
        }:
            raise RepresentationError(
                "project_split must be train, validation, or test when supplied"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ProductionRepresentations:
    """The aligned B0/B1 tensors and their complete cache provenance."""

    event_frame: np.ndarray
    voxel_grid: np.ndarray
    time_surface: np.ndarray
    source: SourceIdentity
    config: RendererConfig
    cache_key: str

    def tensors(self) -> dict[str, np.ndarray]:
        return {
            "event_frame": self.event_frame,
            "voxel_grid": self.voxel_grid,
            "time_surface": self.time_surface,
        }


@dataclass(frozen=True)
class ProductionEventFrame:
    """The frame-only B0 tensor and its complete source provenance."""

    event_frame: np.ndarray
    source: SourceIdentity
    config: RendererConfig
    cache_key: str

    def tensors(self) -> dict[str, np.ndarray]:
        return {"event_frame": self.event_frame}


DEFAULT_RENDERER_CONFIG = RendererConfig()


def render_production_event_frame(
    fields: Mapping[str, np.ndarray],
    *,
    source: SourceIdentity,
    config: RendererConfig = DEFAULT_RENDERER_CONFIG,
) -> ProductionEventFrame:
    """Render only the production B0 frame; never allocate B1 representations."""

    _require_production_config(config)
    x, y, _t, p = _validated_fields(fields, source)
    event_frame = _render_event_frame(x=x, y=y, p=p, config=config)
    if event_frame.dtype != np.float32 or not bool(np.isfinite(event_frame).all()):
        raise RepresentationError("event_frame must be finite float32")
    return ProductionEventFrame(
        event_frame=event_frame,
        source=source,
        config=config,
        cache_key=representation_cache_key(source=source, config=config),
    )


def render_production_representations(
    fields: Mapping[str, np.ndarray],
    *,
    source: SourceIdentity,
    config: RendererConfig = DEFAULT_RENDERER_CONFIG,
) -> ProductionRepresentations:
    """Render aligned production B0/B1 inputs from one complete event subset.

    A zero-duration non-empty sample is valid: every event is at the interval end,
    hence voxel bin ``B-1``, while occupied time-surface pixels have value one.
    """

    _require_production_config(config)
    x, y, t, p = _validated_fields(fields, source)
    x_out = x.astype(np.int64, copy=False)
    y_out = y.astype(np.int64, copy=False)
    polarity = p.astype(np.int64, copy=False)  # False/negative=0, True/positive=1.

    event_frame = _render_event_frame(x=x, y=y, p=p, config=config)

    span = source.temporal_end - source.temporal_start
    if span == 0:
        normalized_t = np.ones(t.shape, dtype=np.float64)
    else:
        normalized_t = (t.astype(np.float64) - source.temporal_start) / span
        np.clip(normalized_t, 0.0, 1.0, out=normalized_t)

    continuous_bin = normalized_t * (config.voxel_bins - 1)
    lower = np.floor(continuous_bin).astype(np.int64)
    upper = np.minimum(lower + 1, config.voxel_bins - 1)
    upper_weight = (continuous_bin - lower).astype(np.float32)
    lower_weight = np.float32(1.0) - upper_weight
    voxel_mass = np.zeros(
        (2, config.voxel_bins, config.output_height, config.output_width),
        dtype=np.float32,
    )
    np.add.at(voxel_mass, (polarity, lower, y_out, x_out), lower_weight)
    distinct_upper = upper != lower
    if np.any(distinct_upper):
        np.add.at(
            voxel_mass,
            (
                polarity[distinct_upper],
                upper[distinct_upper],
                y_out[distinct_upper],
                x_out[distinct_upper],
            ),
            upper_weight[distinct_upper],
        )
    voxel_grid = np.log1p(voxel_mass).astype(np.float32, copy=False)

    latest = np.full(
        (2, config.output_height, config.output_width),
        -np.inf,
        dtype=np.float32,
    )
    np.maximum.at(latest, (polarity, y_out, x_out), normalized_t.astype(np.float32))
    occupied = np.isfinite(latest)
    time_surface = np.zeros_like(latest)
    time_surface[occupied] = np.exp(
        -(np.float32(1.0) - latest[occupied]) / np.float32(config.time_surface_tau)
    ).astype(np.float32, copy=False)

    tensors = {
        "event_frame": event_frame,
        "voxel_grid": voxel_grid,
        "time_surface": time_surface,
    }
    for name, tensor in tensors.items():
        if tensor.dtype != np.float32 or not bool(np.isfinite(tensor).all()):
            raise RepresentationError(f"{name} must be finite float32")
    cache_key = representation_cache_key(source=source, config=config)
    return ProductionRepresentations(source=source, config=config, cache_key=cache_key, **tensors)


def _render_event_frame(
    *,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    config: RendererConfig,
) -> np.ndarray:
    x_out = x.astype(np.int64, copy=False)
    y_out = y.astype(np.int64, copy=False)
    polarity = p.astype(np.int64, copy=False)
    frame_counts = np.zeros((2, config.output_height, config.output_width), dtype=np.uint32)
    np.add.at(frame_counts, (polarity, y_out, x_out), np.uint32(1))
    return np.log1p(frame_counts.astype(np.float32)).astype(np.float32, copy=False)


def representation_cache_key(*, source: SourceIdentity, config: RendererConfig) -> str:
    """Return the content-addressed key for raw identity plus every renderer parameter."""

    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "contract_name": CONTRACT_NAME,
        "dataset_release": DATASET_RELEASE,
        "raw_contract": {
            "spatial_resolution": [SOURCE_HEIGHT, SOURCE_WIDTH],
            "field_dtypes": {"x": "uint16", "y": "uint16", "t": "uint16", "p": "bool"},
        },
        "renderer": config.to_dict(),
        "source": source.to_dict(),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def cache_path(cache_root: str | Path, cache_key: str) -> Path:
    if len(cache_key) != 64 or any(char not in "0123456789abcdef" for char in cache_key):
        raise RepresentationError("cache_key must be a lowercase SHA-256 hex digest")
    return Path(cache_root) / cache_key[:2] / f"{cache_key}.npz"


def frame_cache_key(*, source: SourceIdentity, config: RendererConfig) -> str:
    """Key for the B0-only cache; it cannot collide with a B1 bundle entry."""

    return hashlib.sha256(
        _canonical_json(
            {
                "frame_cache_schema_version": FRAME_CACHE_SCHEMA_VERSION,
                "representation_cache_key": representation_cache_key(source=source, config=config),
                "tensor_names": ["event_frame"],
            }
        )
    ).hexdigest()


def write_frame_cache(
    path: str | Path, frame: ProductionEventFrame, *, cache_key: str | None = None
) -> None:
    """Atomically persist exactly one B0 event frame, never B1 tensors."""

    _require_production_config(frame.config)
    tensor = np.asarray(frame.event_frame)
    if tensor.shape != _expected_shapes(frame.config)["event_frame"] or tensor.dtype != np.float32:
        raise RepresentationError("frame cache tensor violates the B0 event-frame contract")
    if not bool(np.isfinite(tensor).all()):
        raise RepresentationError("frame cache tensor must be finite")
    key = frame_cache_key(source=frame.source, config=frame.config) if cache_key is None else cache_key
    if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
        raise RepresentationError("frame cache key must be a lowercase SHA-256 hex digest")
    manifest = {
        "frame_cache_schema_version": FRAME_CACHE_SCHEMA_VERSION,
        "cache_key": key,
        "source": frame.source.to_dict(),
        "renderer": frame.config.to_dict(),
        "tensor": {"shape": list(tensor.shape), "dtype": "float32", "sha256": _tensor_sha256(tensor)},
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, event_frame=tensor, manifest_utf8=np.frombuffer(_canonical_json(manifest), dtype=np.uint8))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def read_frame_cache(
    path: str | Path, *, expected_source: SourceIdentity, expected_config: RendererConfig,
    expected_cache_key: str | None = None,
) -> ProductionEventFrame:
    """Strictly validate a B0-only cache entry before returning its frame."""

    key = frame_cache_key(source=expected_source, config=expected_config) if expected_cache_key is None else expected_cache_key
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            if set(archive.files) != {"event_frame", "manifest_utf8"}:
                raise RepresentationError("frame cache contains non-B0 tensors")
            manifest = json.loads(bytes(archive["manifest_utf8"]).decode("utf-8"))
            tensor = archive["event_frame"].copy()
    except RepresentationError:
        raise
    except Exception as exc:
        raise RepresentationError(f"could not read frame cache: {exc}") from exc
    source_record = manifest.get("source")
    if not isinstance(source_record, dict):
        raise RepresentationError("frame cache source provenance is malformed")
    # A manifest-backed B0 cache hit deliberately validates only catalog facts:
    # the immutable sample ID and raw NPZ content digest.  Event-level facts are
    # retained in the entry for metadata but are not re-derived from an archive.
    catalog_match = (
        source_record.get("sample_id") == expected_source.sample_id
        and source_record.get("split") == expected_source.split
        and source_record.get("project_split") == expected_source.project_split
        and source_record.get("raw_content_sha256") == expected_source.raw_content_sha256
    )
    if (manifest.get("frame_cache_schema_version") != FRAME_CACHE_SCHEMA_VERSION
            or manifest.get("cache_key") != key
            or not catalog_match
            or manifest.get("renderer") != expected_config.to_dict()):
        raise RepresentationError("frame cache provenance does not match expected renderer/source")
    record = manifest.get("tensor", {})
    if (tensor.shape != _expected_shapes(expected_config)["event_frame"] or tensor.dtype != np.float32
            or record.get("shape") != list(tensor.shape) or record.get("dtype") != "float32"
            or record.get("sha256") != _tensor_sha256(tensor) or not bool(np.isfinite(tensor).all())):
        raise RepresentationError("frame cache tensor validation failed")
    try:
        stored_source = SourceIdentity(**source_record)
    except (TypeError, RepresentationError) as exc:
        raise RepresentationError(f"frame cache source provenance is invalid: {exc}") from exc
    return ProductionEventFrame(event_frame=tensor, source=stored_source, config=expected_config, cache_key=key)


def write_representation_cache(path: str | Path, bundle: ProductionRepresentations) -> None:
    """Atomically write an uncompressed, pickle-free NPZ with canonical provenance."""

    _require_production_config(bundle.config)
    _validate_tensor_contract(bundle.tensors(), bundle.config)
    target = Path(path)
    expected = representation_cache_key(source=bundle.source, config=bundle.config)
    if bundle.cache_key != expected:
        raise RepresentationError("bundle cache key does not match its provenance")
    manifest = _manifest(bundle)
    manifest_bytes = _canonical_json(manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                event_frame=bundle.event_frame,
                voxel_grid=bundle.voxel_grid,
                time_surface=bundle.time_surface,
                manifest_utf8=np.frombuffer(manifest_bytes, dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def read_representation_cache(
    path: str | Path,
    *,
    expected_source: SourceIdentity,
    expected_config: RendererConfig,
) -> ProductionRepresentations:
    """Load a cache only when source identity, parameters, shapes, and hashes match."""

    expected_key = representation_cache_key(source=expected_source, config=expected_config)
    _require_production_config(expected_config)
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            if set(archive.files) != {*REPRESENTATION_NAMES, "manifest_utf8"}:
                raise RepresentationError("cache entries do not match the production schema")
            manifest = json.loads(bytes(archive["manifest_utf8"]).decode("utf-8"))
            tensors = {name: archive[name].copy() for name in REPRESENTATION_NAMES}
    except RepresentationError:
        raise
    except Exception as exc:
        raise RepresentationError(f"could not read representation cache: {exc}") from exc
    if manifest.get("cache_key") != expected_key:
        raise RepresentationError("cache provenance does not match expected renderer parameters/source")
    if manifest.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise RepresentationError("cache schema version does not match production")
    if manifest.get("contract_name") != CONTRACT_NAME:
        raise RepresentationError("cache contract name does not match production")
    if manifest.get("dataset_release") != DATASET_RELEASE:
        raise RepresentationError("cache dataset release does not match production")
    if manifest.get("raw_contract") != _raw_contract():
        raise RepresentationError("cache raw contract does not match production")
    if manifest.get("source") != expected_source.to_dict():
        raise RepresentationError("cache source identity does not match expected source")
    if manifest.get("renderer") != expected_config.to_dict():
        raise RepresentationError("cache renderer parameters do not match expected parameters")
    _validate_tensor_contract(tensors, expected_config)
    for name, tensor in tensors.items():
        record = manifest.get("tensors", {}).get(name, {})
        if record.get("dtype") != "float32" or list(tensor.shape) != record.get("shape"):
            raise RepresentationError(f"cached {name} shape/dtype does not match its manifest")
        if _tensor_sha256(tensor) != record.get("sha256") or not np.isfinite(tensor).all():
            raise RepresentationError(f"cached {name} failed content or finite-value validation")
    return ProductionRepresentations(
        source=expected_source,
        config=expected_config,
        cache_key=expected_key,
        **tensors,
    )


def _validated_fields(
    fields: Mapping[str, np.ndarray], source: SourceIdentity
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if set(fields) != {"x", "y", "t", "p"}:
        raise RepresentationError("fields must be exactly x, y, t, p")
    arrays = tuple(np.asarray(fields[name]) for name in ("x", "y", "t", "p"))
    expected = (np.dtype("<u2"), np.dtype("<u2"), np.dtype("<u2"), np.dtype("?"))
    for name, array, dtype in zip(("x", "y", "t", "p"), arrays, expected):
        if array.ndim != 1 or array.dtype != dtype:
            raise RepresentationError(f"field {name} must be one-dimensional {dtype}")
        if len(array) != source.event_count:
            raise RepresentationError(f"field {name} length does not match source event_count")
    x, y, t, p = arrays
    if np.any(x >= SOURCE_WIDTH) or np.any(y >= SOURCE_HEIGHT):
        raise RepresentationError("event coordinates exceed native 480x640 bounds")
    if len(t) > 1 and not bool(np.all(t[1:] >= t[:-1])):
        raise RepresentationError("timestamps must be nondecreasing")
    if int(t[0]) != source.temporal_start or int(t[-1]) != source.temporal_end:
        raise RepresentationError("source interval must equal observed timestamp support")
    observed_subset_id = compute_event_fingerprint(fields)
    if observed_subset_id != source.event_subset_id:
        raise RepresentationError("source event_subset_id does not match the supplied raw fields")
    return x, y, t, p  # type: ignore[return-value]


def _manifest(bundle: ProductionRepresentations) -> dict[str, object]:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "contract_name": CONTRACT_NAME,
        "dataset_release": DATASET_RELEASE,
        "raw_contract": _raw_contract(),
        "cache_key": bundle.cache_key,
        "source": bundle.source.to_dict(),
        "renderer": bundle.config.to_dict(),
        "tensors": {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": _tensor_sha256(tensor),
            }
            for name, tensor in bundle.tensors().items()
        },
    }


def _tensor_sha256(tensor: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(tensor)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _require_production_config(config: RendererConfig) -> None:
    if config != DEFAULT_RENDERER_CONFIG:
        raise RepresentationError(
            "production rendering/cache accepts only the D012 native 480x640, B=5, tau=0.3 contract"
        )


def _expected_shapes(config: RendererConfig) -> dict[str, tuple[int, ...]]:
    return {
        "event_frame": (2, config.output_height, config.output_width),
        "voxel_grid": (2, config.voxel_bins, config.output_height, config.output_width),
        "time_surface": (2, config.output_height, config.output_width),
    }


def _validate_tensor_contract(
    tensors: Mapping[str, np.ndarray], config: RendererConfig
) -> None:
    expected_shapes = _expected_shapes(config)
    if set(tensors) != set(expected_shapes):
        raise RepresentationError("tensor names do not match the production contract")
    for name, expected_shape in expected_shapes.items():
        tensor = np.asarray(tensors[name])
        if tensor.shape != expected_shape or tensor.dtype != np.float32:
            raise RepresentationError(
                f"{name} must have production shape {expected_shape} and dtype float32"
            )
        if not bool(np.isfinite(tensor).all()):
            raise RepresentationError(f"{name} must contain only finite values")


def _raw_contract() -> dict[str, object]:
    return {
        "spatial_resolution": [SOURCE_HEIGHT, SOURCE_WIDTH],
        "field_dtypes": {"x": "uint16", "y": "uint16", "t": "uint16", "p": "bool"},
    }


def compute_event_fingerprint(fields: Mapping[str, np.ndarray]) -> str:
    """Match ``probe.compute_event_subset_id`` without a PyTorch conversion.

    The production raw dtypes are fixed, so their canonical PyTorch dtype names
    can be emitted directly while hashing the same contiguous bytes.
    """

    digest = hashlib.sha256()
    digest.update(b"ebackbone_v3:event_subset:v1\n")
    for role, field_name, torch_dtype in (
        ("x", "x", "torch.uint16"),
        ("y", "y", "torch.uint16"),
        ("timestamp", "t", "torch.uint16"),
        ("polarity", "p", "torch.bool"),
    ):
        array = np.ascontiguousarray(fields[field_name])
        digest.update(role.encode("utf-8"))
        digest.update(b"\0")
        digest.update(field_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(torch_dtype.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.view(np.uint8).tobytes())
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "FRAME_CACHE_SCHEMA_VERSION",
    "CONTRACT_NAME",
    "compute_event_fingerprint",
    "DATASET_RELEASE",
    "DEFAULT_RENDERER_CONFIG",
    "POLARITY_ORDER",
    "ProductionEventFrame",
    "ProductionRepresentations",
    "RendererConfig",
    "RepresentationError",
    "SourceIdentity",
    "cache_path",
    "frame_cache_key",
    "read_frame_cache",
    "read_representation_cache",
    "render_production_event_frame",
    "render_production_representations",
    "representation_cache_key",
    "write_representation_cache",
    "write_frame_cache",
]
