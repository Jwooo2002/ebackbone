"""Label-free ordered windows for the isolated hierarchy/CLIP comparisons.

Mini access is for engineering checks, not a choice of downstream benchmark.
The immutable Mini adapter supplies train/validation raw samples; this module
never permits final-test access, samples events, or changes the point MLP.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from .hierarchy_data import collate as packed_collate
from .representations import _validated_fields
from .v1_data import V1Dataset, raw


SEQUENCE_CONTRACT = {
    "version": "hierarchy-clip-equal-duration-windows-v1",
    "window_assignment": "floor(K*(t-start)/duration), clamped to K-1",
    "boundaries": "left-closed right-open, final window right-closed",
    "window_time_normalization": "relative to fixed equal-duration boundaries",
    "zero_duration": "all events in final window, normalized time 1, voxel bin 7",
    "empty_windows": "no fabricated events; mask false; encoded map zero",
    "voxel_bins_per_window": 8,
    "spatial_resolution": [480, 640],
    "point_channels": ["x/639", "y/479", "local_normalized_time", "2*p-1"],
    "event_filtering": "none", "sampling": "none", "augmentation": "none",
    "flattening": "sample-major, window-minor [B*K]",
    "allowed_splits": ["train", "validation"],
}


def sequence_contract(num_windows):
    """Return the exact contract and fingerprint, including the window count."""
    _validate_num_windows(num_windows)
    contract = {**SEQUENCE_CONTRACT, "num_windows": num_windows}
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return contract, fingerprint


def _validate_num_windows(num_windows):
    if isinstance(num_windows, bool) or not isinstance(num_windows, int) or num_windows < 1:
        raise ValueError("num_windows must be a positive integer")


def split_event_windows(fields, source, *, num_windows=4):
    """Split validated native raw fields without accepting labels.

    Integer arithmetic assigns boundary events exactly once; original ordering
    (including ties) is retained. Returned index arrays reference the original
    fields, making event conservation independently inspectable. Empty windows
    have zero-length arrays, never synthetic points or copied events.
    """
    _validate_num_windows(num_windows)
    x, y, t, p = _validated_fields(fields, source)
    elapsed = t.astype(np.int64) - source.temporal_start
    duration = source.temporal_end - source.temporal_start
    if num_windows > np.iinfo(np.int64).max // max(duration, 1):
        raise ValueError("num_windows is too large for exact integer window assignment")
    if duration:
        scaled = elapsed * num_windows
        assignment = np.minimum(scaled // duration, num_windows - 1)
        local_time = (scaled - assignment * duration).astype(np.float64) / duration
    else:
        assignment = np.full(len(t), num_windows - 1, dtype=np.int64)
        local_time = np.ones(len(t), dtype=np.float64)
    boundaries = np.linspace(source.temporal_start, source.temporal_end, num_windows + 1)
    result = []
    for index in range(num_windows):
        selected = np.flatnonzero(assignment == index)
        result.append({
            "index": index,
            "event_indices": selected,
            "fields": {name: values[selected] for name, values in zip(("x", "y", "t", "p"), (x, y, t, p))},
            "normalized_time": local_time[selected],
            "temporal_start": float(boundaries[index]),
            "temporal_end": float(boundaries[index + 1]),
            "interval_closure": "[]" if index == num_windows - 1 else "[)",
            "event_count": len(selected),
            "valid": bool(len(selected)),
        })
    return result


def prepare_sequence(fields, source, *, num_windows=4):
    """Prepare unchanged four-channel point inputs for each ordered window."""
    windows = split_event_windows(fields, source, num_windows=num_windows)
    packed_windows, metadata = [], []
    for window in windows:
        x, y, p = (window["fields"][name] for name in ("x", "y", "p"))
        normalized = window["normalized_time"]
        points = np.stack((x.astype(np.float32) / 639, y.astype(np.float32) / 479,
                           normalized.astype(np.float32), 2 * p.astype(np.float32) - 1), axis=1)
        position = normalized * 7
        lower = np.floor(position).astype(np.int64)
        upper = np.minimum(lower + 1, 7)
        spatial = (y.astype(np.int64) // 4) * 160 + x.astype(np.int64) // 4
        packed_windows.append({
            "points": torch.from_numpy(points),
            "voxel_lower": torch.from_numpy(lower * 120 * 160 + spatial),
            "voxel_upper": torch.from_numpy(upper * 120 * 160 + spatial),
            "alpha": torch.from_numpy((position - lower).astype(np.float32)),
            "event_counts": torch.tensor([window["event_count"]], dtype=torch.long),
        })
        metadata.append({key: window[key] for key in (
            "index", "temporal_start", "temporal_end", "interval_closure", "event_count", "valid")})
    return {"windows": packed_windows,
            "window_mask": torch.tensor([w["valid"] for w in windows], dtype=torch.bool),
            "window_metadata": metadata}


class HierarchySequenceDataset(V1Dataset):
    """Immutable Mini train/validation raw access for bounded compatibility checks."""

    def __init__(self, manifest_dir, dataset_root, split, *, num_windows=4,
                 raw_cache=None, limit=None, seed=20260908):
        self.sequence_contract, self.sequence_contract_sha256 = sequence_contract(num_windows)
        self.num_windows = num_windows
        # V1Dataset opens and validates the immutable manifests and denies test.
        super().__init__(manifest_dir, dataset_root, split, multiview=False,
                         raw_cache=raw_cache, limit=limit, seed=seed)

    def __getitem__(self, index):
        row = self.rows[index]
        start = time.perf_counter()
        fields = raw._decode_event_fields(self.payload(row), sample_id=row.sample_id)
        source = raw._source_identity(row, fields)
        decoded = time.perf_counter()
        sequence = prepare_sequence(fields, source, num_windows=self.num_windows)
        end = time.perf_counter()
        return {**sequence, "label": row.class_label, "sample_id": row.sample_id,
                "split": row.split, "source_split": row.source_split,
                "source": asdict(source), "decode_seconds": decoded - start,
                "render_seconds": end - decoded,
                "sequence_contract_sha256": self.sequence_contract_sha256}


def sequence_collate(samples, *, height=480, width=640):
    """Pack B*K windows with offsets, preserving original sample identities."""
    if not samples:
        raise ValueError("sequence batch must contain at least one sample")
    num_windows = len(samples[0]["windows"])
    _validate_num_windows(num_windows)
    if min(height, width) < 32 or height % 32 or width % 32:
        raise ValueError("height and width must be positive multiples of 32")
    flattened = []
    for sample in samples:
        mask = sample["window_mask"]
        if len(sample["windows"]) != num_windows or mask.shape != (num_windows,) or mask.dtype != torch.bool:
            raise ValueError("all samples must share num_windows and boolean window_mask[K]")
        counts = torch.tensor([int(w["event_counts"].sum()) for w in sample["windows"]])
        if not torch.equal(mask.cpu(), counts > 0):
            raise ValueError("window mask must agree with event counts")
        for window in sample["windows"]:
            if window["event_counts"].shape != (1,) or int(window["event_counts"].item()) != len(window["points"]):
                raise ValueError("each window event_count must equal its actual points")
            flattened.append({**sample, "inputs": window})
    packed = packed_collate(flattened, height=height, width=width)["inputs"]
    return {"inputs": {"packed": packed, "window_mask": torch.stack([s["window_mask"] for s in samples])},
            "labels": torch.tensor([s["label"] for s in samples], dtype=torch.long),
            "sample_ids": [s["sample_id"] for s in samples],
            "splits": [s["split"] for s in samples],
            "source_splits": [s["source_split"] for s in samples],
            "window_metadata": [s.get("window_metadata") for s in samples],
            "sources": [s.get("source") for s in samples],
            "sequence_contract_sha256": [s.get("sequence_contract_sha256") for s in samples],
            "decode_seconds": sum(s["decode_seconds"] for s in samples),
            "render_seconds": sum(s["render_seconds"] for s in samples)}
