"""One raw decode supplies packed points and a polarity-separated time surface."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from .hierarchy_data import HierarchyDataset, POINT_CONTRACT, prepare_points, collate as point_collate
from .representations import _validated_fields
from .v1_data import raw

TS_CONTRACT = {
    **POINT_CONTRACT, "version": "hierarchy-points-ts-native-all-events-1", "time_surface": True,
    "ts": {"channels": ["negative", "positive"], "shape": [2, 480, 640], "dtype": "float32",
           "tau_duration_ratio": 0.2, "background": 0, "zero_duration_occupied": 1,
           "formula": "exp(-(1-last_normalized_timestamp)/0.2)",
           "source": "exactly the same full raw event subset and interval as points",
           "availability": "train and validation and inference"},
}
TS_CONTRACT_SHA256 = hashlib.sha256(json.dumps(TS_CONTRACT, sort_keys=True).encode()).hexdigest()


def render_time_surface(fields, source):
    x, y, t, p = _validated_fields(fields, source)
    x, y, p = (a.astype(np.int64) for a in (x, y, p))
    span = source.temporal_end - source.temporal_start
    normalized = ((t.astype(np.float64) - source.temporal_start) / span
                  if span else np.ones(len(t), np.float64))
    latest = np.full((2, 480, 640), -np.inf, np.float32)
    np.maximum.at(latest, (p, y, x), normalized.astype(np.float32))
    surface = np.zeros_like(latest)
    occupied = np.isfinite(latest)
    surface[occupied] = np.exp(-(1 - latest[occupied]) / np.float32(0.2))
    return torch.from_numpy(surface)


class HierarchyTSDataset(HierarchyDataset):
    def __getitem__(self, index):
        row = self.rows[index]
        start = time.perf_counter()
        fields = raw._decode_event_fields(self.payload(row), sample_id=row.sample_id)
        source = raw._source_identity(row, fields)
        decoded = time.perf_counter()
        inputs = {**prepare_points(fields, source), "time_surface": render_time_surface(fields, source)}
        end = time.perf_counter()
        return {"inputs": inputs, "label": row.class_label, "sample_id": row.sample_id,
                "split": row.split, "source_split": row.source_split,
                "decode_seconds": decoded - start, "render_seconds": end - decoded,
                "source": asdict(source), "representation_contract_sha256": TS_CONTRACT_SHA256}


def collate(samples, *, height=480, width=640):
    batch = point_collate(samples, height=height, width=width)
    batch["inputs"]["time_surface"] = torch.stack([s["inputs"]["time_surface"] for s in samples])
    return batch
