"""All-event packed input; reuse immutable V1 raw access, never render images."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from .representations import _validated_fields
from .v1_data import V1Dataset, raw

POINT_CONTRACT = {
    "version": "hierarchy-points-native-all-events-1", "height": 480, "width": 640,
    "point_channels": ["x/639", "y/479", "(t-start)/duration", "2*p-1"],
    "dtype": "float32", "voxel_bins": 8, "spatial_cell_size": 4,
    "spatial_assignment": "floor(x/4), floor(y/4), native coordinates",
    "temporal_assignment": "linear interpolation across 8 bins; zero duration maps to bin 7",
    "voxel_values": "weighted mean of learned 32D points plus log1p interpolated event mass",
    "window": "whole stored array, observed support, closed endpoints",
    "event_filtering": "none", "sampling": "none", "augmentation": "none",
    "batching": "packed variable lengths; no padding or truncation",
    "raw_frame": False, "time_surface": False,
}
POINT_CONTRACT_SHA256 = hashlib.sha256(json.dumps(POINT_CONTRACT, sort_keys=True).encode()).hexdigest()


def prepare_points(fields, source):
    x, y, t, p = _validated_fields(fields, source)
    span = source.temporal_end - source.temporal_start
    normalized = ((t.astype(np.float64) - source.temporal_start) / span
                  if span else np.ones(len(t), np.float64))
    points = np.stack((x.astype(np.float32) / 639, y.astype(np.float32) / 479,
                       normalized.astype(np.float32), 2 * p.astype(np.float32) - 1), axis=1)
    position = normalized * 7
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, 7)
    spatial = (y.astype(np.int64) // 4) * 160 + x.astype(np.int64) // 4
    return {"points": torch.from_numpy(points),
            "voxel_lower": torch.from_numpy(lower * 120 * 160 + spatial),
            "voxel_upper": torch.from_numpy(upper * 120 * 160 + spatial),
            "alpha": torch.from_numpy((position - lower).astype(np.float32)),
            "event_counts": torch.tensor([len(t)], dtype=torch.long)}


class HierarchyDataset(V1Dataset):
    def __init__(self, manifest_dir, dataset_root, split, *, raw_cache=None, limit=None, seed=20260908):
        # No final-test override exists in this supervised training path.
        super().__init__(manifest_dir, dataset_root, split, multiview=False,
                         raw_cache=raw_cache, limit=limit, seed=seed)

    def __getitem__(self, index):
        row = self.rows[index]
        start = time.perf_counter()
        fields = raw._decode_event_fields(self.payload(row), sample_id=row.sample_id)
        source = raw._source_identity(row, fields)
        decoded = time.perf_counter()
        inputs = prepare_points(fields, source)
        end = time.perf_counter()
        return {"inputs": inputs, "label": row.class_label, "sample_id": row.sample_id,
                "split": row.split, "source_split": row.source_split,
                "decode_seconds": decoded - start, "render_seconds": end - decoded,
                "source": asdict(source), "point_contract_sha256": POINT_CONTRACT_SHA256}


def collate(samples, *, height=480, width=640):
    cells = 8 * (height // 4) * (width // 4)
    inputs = {}
    for key in ("points", "voxel_lower", "voxel_upper", "alpha", "event_counts"):
        values = [(s["inputs"][key] + i * cells if key.startswith("voxel_") else s["inputs"][key])
                  for i, s in enumerate(samples)]
        inputs[key] = torch.cat(values, dim=0)
    return {"inputs": inputs, "labels": torch.tensor([s["label"] for s in samples], dtype=torch.long),
            "sample_ids": [s["sample_id"] for s in samples], "splits": [s["split"] for s in samples],
            "source_splits": [s["source_split"] for s in samples],
            "render_seconds": sum(s["render_seconds"] for s in samples),
            "decode_seconds": sum(s["decode_seconds"] for s in samples)}
