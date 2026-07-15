"""Archive-native N-ImageNet mini provider for the V3 real-data probe.

This module preserves the stored structured event fields and renders all three
probe representations from one validated in-memory event subset. Class labels
are never passed to the renderer and are used solely for deterministic
supervised bookkeeping.
"""

from __future__ import annotations

import errno
import math
import re
import tarfile
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Mapping

import numpy as np

from ebackbone_v3.contracts import (
    ClassificationRecord,
    ProbeSample,
    RawEventRecord,
    RepresentationRecord,
)
from ebackbone_v3.errors import ProbeError
from ebackbone_v3.probe import compute_event_subset_id


DATASET_NAME = "N-ImageNet mini (100-class original train/validation release)"
ARCHIVE_DIRECTORY = Path("mini_zenodo") / "archives"
VALIDATION_ARCHIVE = "mini_validation_split.zip"
TRAIN_ARCHIVE_NAMES = tuple(f"train_Part_{index}.zip" for index in range(1, 11))
SENSOR_HEIGHT = 480
SENSOR_WIDTH = 640
TIMESTAMP_UNIT = "microsecond"
TIME_SURFACE_TAU = 0.3
EXPECTED_CLASS_COUNT = 100
EXPECTED_EVENT_DTYPE = np.dtype(
    [("x", "<u2"), ("y", "<u2"), ("t", "<u2"), ("p", "?")],
    align=False,
)
CLASS_ID_PATTERN = re.compile(r"^n[0-9]{8}$")


@dataclass(frozen=True)
class _ArchiveCatalog:
    archive_dir: Path
    train_classes: Mapping[str, tuple[Path, str]]
    validation_classes: frozenset[str]
    class_to_index: Mapping[str, int]


def load_sample(
    *,
    dataset_root: str | Path,
    split: str,
    sample_id: str,
    voxel_bins: int,
) -> ProbeSample:
    """Load and render one complete archive-native N-ImageNet mini sample."""

    normalized_split = _normalize_split(split)
    if not isinstance(voxel_bins, int) or isinstance(voxel_bins, bool) or voxel_bins < 2:
        raise ProbeError("N-ImageNet mini probe voxel_bins must be an integer >= 2")
    canonical_sample_id, class_id, filename = _parse_sample_id(sample_id, normalized_split)
    root = Path(dataset_root).expanduser().resolve()
    archive_dir = root / ARCHIVE_DIRECTORY
    catalog = _build_archive_catalog(archive_dir)
    payload = _read_sample_payload(
        catalog,
        split=normalized_split,
        class_id=class_id,
        filename=filename,
    )
    fields, storage_layout = _decode_event_fields(payload, sample_id=canonical_sample_id)
    temporal_start, temporal_end = _validate_raw_fields(
        fields,
        sample_id=canonical_sample_id,
        height=SENSOR_HEIGHT,
        width=SENSOR_WIDTH,
    )

    # The renderer receives no sample label or class mapping. All three outputs
    # are built together from the same four arrays, before classification
    # bookkeeping is attached to the provider result.
    tensors = render_probe_representations(
        fields,
        spatial_resolution=(SENSOR_HEIGHT, SENSOR_WIDTH),
        voxel_bins=voxel_bins,
    )
    subset_id = compute_event_subset_id(
        fields,
        {"x": "x", "y": "y", "timestamp": "t", "polarity": "p"},
    )
    event_count = int(fields["t"].shape[0])
    if float(tensors["event_frame"].sum(dtype=np.float64)) != float(event_count):
        raise ProbeError("event-frame count sum does not equal the canonical raw-event count")
    if float(tensors["voxel_grid"].sum(dtype=np.float64)) != float(event_count):
        raise ProbeError("voxel-grid count sum does not equal the canonical raw-event count")
    raw = RawEventRecord(
        sample_id=canonical_sample_id,
        fields=fields,
        field_roles={"x": "x", "y": "y", "timestamp": "t", "polarity": "p"},
        storage_layout=storage_layout,
        timestamp_unit=TIMESTAMP_UNIT,
        timestamp_ordering="nondecreasing",
        coordinate_convention="zero_based_xy",
        polarity_encoding="false/true",
        spatial_resolution=(SENSOR_HEIGHT, SENSOR_WIDTH),
        temporal_start=temporal_start,
        temporal_end=temporal_end,
        interval_closure="[]",
        event_count_before_filter=event_count,
        event_count_after_filter=event_count,
        filtering=(
            "none in this provider: the complete stored event_data array is used; "
            "upstream acquisition/export filtering is not encoded in the archive"
        ),
    )
    common = {
        "source_sample_id": canonical_sample_id,
        "temporal_start": temporal_start,
        "temporal_end": temporal_end,
        "interval_closure": "[]",
        "event_subset_id": subset_id,
        "event_count_before_filter": event_count,
        "event_count_after_filter": event_count,
        "spatial_resolution": (SENSOR_HEIGHT, SENSOR_WIDTH),
        "padding_or_truncation": "none",
        "deterministic": True,
    }
    representations = {
        "event_frame": RepresentationRecord(
            tensor=tensors["event_frame"],
            channel_count=2,
            tensor_layout="C,H,W",
            polarity_handling=(
                "separate channels: channel 0=False/negative, channel 1=True/positive"
            ),
            normalization="none; values are per-pixel event counts",
            clipping="none",
            parameters={
                "definition": (
                    "one accumulated polarity-separated per-pixel count frame over the "
                    "complete stored event array"
                ),
                "rendered_frames": 1,
                "frame_axis": "none",
                "channel_order": ["False/negative", "True/positive"],
                "accumulation": "one unit per event",
                "empty_pixel_value": 0.0,
                "contract_scope": "V3 real-data probe; not a production B0/B1 model",
            },
            **common,
        ),
        "voxel_grid": RepresentationRecord(
            tensor=tensors["voxel_grid"],
            channel_count=voxel_bins,
            tensor_layout="C,H,W",
            polarity_handling="polarity validated, then both polarities accumulated together",
            normalization=(
                "timestamps normalized over observed event support for bin assignment only; "
                "voxel counts are not normalized"
            ),
            clipping="bin indices clamped to [0, bins-1]; count values are not clipped",
            parameters={
                "definition": (
                    "unpolarized per-pixel event counts in discrete temporal bins over the "
                    "complete stored event array"
                ),
                "bins": voxel_bins,
                "bin_axis": "C",
                "bin_formula": (
                    "floor(((t-t_min)/(t_max-t_min))*(bins-1)), then clamp to [0,bins-1]"
                ),
                "interpolation": "none",
                "last_bin_behavior": (
                    "under the adopted bins-1 formula, the final bin contains events at "
                    "normalized timestamp exactly 1"
                ),
                "empty_voxel_value": 0.0,
                "contract_scope": "V3 real-data probe; code-derived renderer choice",
            },
            **common,
        ),
        "time_surface": RepresentationRecord(
            tensor=tensors["time_surface"],
            channel_count=2,
            tensor_layout="C,H,W",
            polarity_handling=(
                "separate channels: channel 0=False/negative, channel 1=True/positive"
            ),
            normalization=(
                "latest timestamps first normalized per sample as "
                "(t-t_min)/(t_max-t_min), then exponentially decayed"
            ),
            clipping="none",
            parameters={
                "definition": (
                    "per-polarity exponential time surface from the latest normalized "
                    "timestamp at each pixel"
                ),
                "reference": "t_ref=t_max of the observed event-support interval",
                "decay": {
                    "type": "exponential",
                    "formula": "exp(-(1-latest_normalized_t)/tau)",
                    "tau": TIME_SURFACE_TAU,
                    "tau_unit": "normalized sample interval",
                },
                "formula": (
                    "latest[p,y,x]=max((t-t_min)/(t_max-t_min)); "
                    "S[p,y,x]=exp(-(1-latest[p,y,x])/0.3)"
                ),
                "duplicate_reduction": "maximum, independent of duplicate indexed-write order",
                "channel_order": ["False/negative", "True/positive"],
                "empty_pixel_value": math.exp(-1.0 / TIME_SURFACE_TAU),
                "contract_scope": (
                    "official N-ImageNet exponential formula at native resolution and full "
                    "sample support, with V3 deterministic maximum reduction"
                ),
            },
            **common,
        ),
    }

    class_to_index = dict(catalog.class_to_index)
    classification = ClassificationRecord(
        class_id=class_id,
        class_index=class_to_index[class_id],
        class_count=len(class_to_index),
        class_to_index=class_to_index,
        mapping_policy=(
            "zero-based lexicographic ordering of the 100 WordNet synset IDs present in both "
            "the train and validation archives"
        ),
        label_source=(
            "WordNet synset directory in the archive member path; used only for deterministic "
            "supervised bookkeeping"
        ),
    )
    return ProbeSample(
        dataset_name=DATASET_NAME,
        split=normalized_split,
        raw_events=raw,
        representations=representations,
        classification=classification,
    )


def render_probe_representations(
    fields: Mapping[str, np.ndarray],
    *,
    spatial_resolution: tuple[int, int],
    voxel_bins: int,
) -> dict[str, np.ndarray]:
    """Render the three deterministic probe tensors from one raw-event mapping."""

    if not isinstance(voxel_bins, int) or isinstance(voxel_bins, bool) or voxel_bins < 2:
        raise ProbeError("voxel_bins must be an integer >= 2")
    if len(spatial_resolution) != 2:
        raise ProbeError("spatial_resolution must contain (height, width)")
    height, width = spatial_resolution
    if height <= 0 or width <= 0:
        raise ProbeError("spatial_resolution values must be positive")
    x, y, t, p = _event_arrays(fields)
    _validate_array_values(x, y, t, p, height=height, width=width, sample_id="renderer input")

    x_index = x.astype(np.int64, copy=False)
    y_index = y.astype(np.int64, copy=False)
    polarity_index = p.astype(np.int64, copy=False)
    timestamp = t.astype(np.float64, copy=False)
    temporal_start = float(timestamp[0])
    temporal_end = float(timestamp[-1])
    span = temporal_end - temporal_start
    if span <= 0:
        raise ProbeError("renderer input timestamp span must be positive")
    normalized_t = (timestamp - temporal_start) / span

    frame = np.zeros((2, height, width), dtype=np.float32)
    np.add.at(frame, (polarity_index, y_index, x_index), np.float32(1.0))

    bin_index = np.floor(normalized_t * (voxel_bins - 1)).astype(np.int64)
    np.clip(bin_index, 0, voxel_bins - 1, out=bin_index)
    voxel = np.zeros((voxel_bins, height, width), dtype=np.float32)
    np.add.at(voxel, (bin_index, y_index, x_index), np.float32(1.0))

    latest_timestamp = np.zeros((2, height, width), dtype=np.float32)
    np.maximum.at(
        latest_timestamp,
        (polarity_index, y_index, x_index),
        normalized_t.astype(np.float32),
    )
    time_surface = np.exp(
        -(np.float32(1.0) - latest_timestamp) / np.float32(TIME_SURFACE_TAU)
    ).astype(np.float32, copy=False)
    return {
        "event_frame": frame,
        "voxel_grid": voxel,
        "time_surface": time_surface,
    }


def _build_archive_catalog(archive_dir: Path) -> _ArchiveCatalog:
    if not archive_dir.is_dir():
        _missing_path(archive_dir, "N-ImageNet mini archive directory does not exist")
    expected_paths = [archive_dir / name for name in TRAIN_ARCHIVE_NAMES]
    missing = [path for path in expected_paths if not path.is_file()]
    if missing:
        _missing_path(missing[0], "N-ImageNet mini train archive does not exist")
    unexpected_train = sorted(
        path.name for path in archive_dir.glob("train_Part_*.zip") if path.name not in TRAIN_ARCHIVE_NAMES
    )
    if unexpected_train:
        raise ProbeError(
            "unexpected N-ImageNet train archives would make the selected release ambiguous: "
            + ", ".join(unexpected_train)
        )

    train_classes: dict[str, tuple[Path, str]] = {}
    for zip_path in expected_paths:
        try:
            with zipfile.ZipFile(zip_path) as handle:
                tar_names = sorted(name for name in handle.namelist() if name.endswith(".tar.gz"))
        except zipfile.BadZipFile as exc:
            raise ProbeError(f"invalid N-ImageNet mini train ZIP archive: {zip_path}") from exc
        for tar_name in tar_names:
            class_id = PurePosixPath(tar_name).name.removesuffix(".tar.gz")
            _validate_class_id(class_id)
            if class_id in train_classes:
                raise ProbeError(f"duplicate train class payload for {class_id}")
            train_classes[class_id] = (zip_path, tar_name)

    validation_path = archive_dir / VALIDATION_ARCHIVE
    if not validation_path.is_file():
        _missing_path(validation_path, "N-ImageNet mini validation archive does not exist")
    validation_classes: set[str] = set()
    try:
        with zipfile.ZipFile(validation_path) as handle:
            for member_name in handle.namelist():
                if not member_name.endswith(".npz"):
                    continue
                parts = PurePosixPath(member_name).parts
                if len(parts) != 3 or parts[0] != "extracted_val":
                    raise ProbeError(
                        f"unexpected N-ImageNet mini validation member layout: {member_name}"
                    )
                _validate_class_id(parts[1])
                validation_classes.add(parts[1])
    except zipfile.BadZipFile as exc:
        raise ProbeError(f"invalid N-ImageNet mini validation ZIP archive: {validation_path}") from exc

    train_set = set(train_classes)
    if train_set != validation_classes:
        raise ProbeError(
            "N-ImageNet mini train/validation class sets differ: "
            f"train_only={sorted(train_set - validation_classes)}, "
            f"validation_only={sorted(validation_classes - train_set)}"
        )
    if len(train_set) != EXPECTED_CLASS_COUNT:
        raise ProbeError(
            f"selected N-ImageNet mini release must contain {EXPECTED_CLASS_COUNT} classes; "
            f"observed {len(train_set)}"
        )
    class_to_index = {class_id: index for index, class_id in enumerate(sorted(train_set))}
    return _ArchiveCatalog(
        archive_dir=archive_dir,
        train_classes=train_classes,
        validation_classes=frozenset(validation_classes),
        class_to_index=class_to_index,
    )


def _read_sample_payload(
    catalog: _ArchiveCatalog,
    *,
    split: str,
    class_id: str,
    filename: str,
) -> bytes:
    if class_id not in catalog.class_to_index:
        raise ProbeError(f"sample class {class_id!r} is not in the selected 100-class release")
    if split == "train":
        zip_path, tar_name = catalog.train_classes[class_id]
        target_member = f"{class_id}/{filename}"
        try:
            with zipfile.ZipFile(zip_path) as zip_handle:
                with zip_handle.open(tar_name) as zipped_tar:
                    with tarfile.open(fileobj=zipped_tar, mode="r|gz") as tar_handle:
                        for member in tar_handle:
                            if not member.isfile() or member.name != target_member:
                                continue
                            extracted = tar_handle.extractfile(member)
                            if extracted is None:
                                raise ProbeError(
                                    f"N-ImageNet train member is not readable: {target_member}"
                                )
                            return extracted.read()
        except (zipfile.BadZipFile, tarfile.TarError) as exc:
            raise ProbeError(
                f"could not read N-ImageNet train archive locator {zip_path}!{tar_name}: {exc}"
            ) from exc
        locator = f"{zip_path}!{tar_name}!{target_member}"
        _missing_path(locator, "N-ImageNet mini train sample does not exist")

    validation_path = catalog.archive_dir / VALIDATION_ARCHIVE
    target_member = f"extracted_val/{class_id}/{filename}"
    try:
        with zipfile.ZipFile(validation_path) as handle:
            try:
                return handle.read(target_member)
            except KeyError:
                locator = f"{validation_path}!{target_member}"
                _missing_path(locator, "N-ImageNet mini validation sample does not exist")
    except zipfile.BadZipFile as exc:
        raise ProbeError(f"invalid N-ImageNet mini validation ZIP archive: {validation_path}") from exc
    raise AssertionError("unreachable")


def _decode_event_fields(
    payload: bytes,
    *,
    sample_id: str,
) -> tuple[dict[str, np.ndarray], str]:
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as data:
            if data.files != ["event_data"]:
                raise ProbeError(
                    f"{sample_id} NPZ keys must be exactly ['event_data']; observed {data.files!r}"
                )
            event_data = data["event_data"]
    except ProbeError:
        raise
    except Exception as exc:
        raise ProbeError(f"could not decode N-ImageNet mini NPZ sample {sample_id}: {exc}") from exc
    if event_data.ndim != 1:
        raise ProbeError(f"{sample_id} event_data must be one-dimensional; got {event_data.shape}")
    if event_data.dtype != EXPECTED_EVENT_DTYPE:
        raise ProbeError(
            f"{sample_id} event_data dtype does not match the selected release: "
            f"expected {EXPECTED_EVENT_DTYPE.descr!r}, observed {event_data.dtype.descr!r}"
        )
    fields = {name: event_data[name].copy() for name in ("x", "y", "t", "p")}
    layout = (
        "NPZ key 'event_data': packed one-dimensional NumPy structured array, "
        "field order x,y,t,p, dtype [('x','<u2'),('y','<u2'),('t','<u2'),('p','|b1')], "
        "itemsize 7 bytes"
    )
    return fields, layout


def _validate_raw_fields(
    fields: Mapping[str, np.ndarray],
    *,
    sample_id: str,
    height: int,
    width: int,
) -> tuple[int, int]:
    x, y, t, p = _event_arrays(fields)
    _validate_array_values(x, y, t, p, height=height, width=width, sample_id=sample_id)
    return int(t[0]), int(t[-1])


def _event_arrays(
    fields: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if set(fields) != {"x", "y", "t", "p"}:
        raise ProbeError(f"raw event fields must be exactly x,y,t,p; observed {sorted(fields)}")
    arrays = tuple(np.asarray(fields[name]) for name in ("x", "y", "t", "p"))
    expected_dtypes = (np.dtype("<u2"), np.dtype("<u2"), np.dtype("<u2"), np.dtype("?"))
    for name, array, expected_dtype in zip(("x", "y", "t", "p"), arrays, expected_dtypes):
        if array.ndim != 1:
            raise ProbeError(f"raw event field {name} must be one-dimensional; got {array.shape}")
        if array.dtype != expected_dtype:
            raise ProbeError(
                f"raw event field {name} must have dtype {expected_dtype}; got {array.dtype}"
            )
    lengths = {int(array.shape[0]) for array in arrays}
    if len(lengths) != 1:
        raise ProbeError("raw x,y,t,p arrays must have identical lengths")
    if not lengths or next(iter(lengths)) == 0:
        raise ProbeError("raw event arrays must not be empty")
    return arrays  # type: ignore[return-value]


def _validate_array_values(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    *,
    height: int,
    width: int,
    sample_id: str,
) -> None:
    if np.any(x >= width) or np.any(y >= height):
        raise ProbeError(
            f"{sample_id} contains coordinates outside zero-based {height}x{width} bounds"
        )
    if p.dtype != np.bool_:
        raise ProbeError(f"{sample_id} polarity field must use bool storage")
    if t.shape[0] > 1 and not bool(np.all(t[1:] >= t[:-1])):
        raise ProbeError(f"{sample_id} timestamps are not nondecreasing")
    if int(t[-1]) <= int(t[0]):
        raise ProbeError(f"{sample_id} observed timestamp support must have positive duration")


def _parse_sample_id(sample_id: str, split: str) -> tuple[str, str, str]:
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ProbeError("N-ImageNet mini sample_id must be a non-empty string")
    parts = PurePosixPath(sample_id.strip()).parts
    if len(parts) != 3 or any(part in {"", ".", ".."} for part in parts):
        raise ProbeError(
            "N-ImageNet mini sample_id must have '<split>/<class_id>/<filename>.npz' format"
        )
    supplied_split = _normalize_split(parts[0])
    if supplied_split != split:
        raise ProbeError(
            f"sample_id split {supplied_split!r} does not match provider split {split!r}"
        )
    class_id, filename = parts[1], parts[2]
    _validate_class_id(class_id)
    if not filename.endswith(".npz") or PurePosixPath(filename).name != filename:
        raise ProbeError(f"N-ImageNet mini sample filename must end with .npz: {filename!r}")
    return f"{split}/{class_id}/{filename}", class_id, filename


def _normalize_split(split: str) -> str:
    if not isinstance(split, str):
        raise ProbeError("N-ImageNet mini split must be 'train' or 'validation'")
    normalized = split.strip().lower()
    if normalized == "val":
        normalized = "validation"
    if normalized == "test":
        raise ProbeError(
            "N-ImageNet mini archive release has no independent test split; do not alias validation"
        )
    if normalized not in {"train", "validation"}:
        raise ProbeError("N-ImageNet mini split must be 'train' or 'validation'")
    return normalized


def _validate_class_id(class_id: str) -> None:
    if not CLASS_ID_PATTERN.fullmatch(class_id):
        raise ProbeError(f"invalid N-ImageNet WordNet synset class id: {class_id!r}")


def _missing_path(path: str | Path, message: str) -> None:
    raise FileNotFoundError(errno.ENOENT, message, str(path))


__all__ = [
    "ARCHIVE_DIRECTORY",
    "DATASET_NAME",
    "EXPECTED_CLASS_COUNT",
    "SENSOR_HEIGHT",
    "SENSOR_WIDTH",
    "TIMESTAMP_UNIT",
    "TIME_SURFACE_TAU",
    "load_sample",
    "render_probe_representations",
]
