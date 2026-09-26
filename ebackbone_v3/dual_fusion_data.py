"""Aligned full-event points and V1 representations for the dual-branch study."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

import torch

from .hierarchy_data import POINT_CONTRACT_SHA256, prepare_points
from .hierarchy_models import INPUT_KEYS as POINT_KEYS
from .v1_data import RENDERER_SHA256, V1Dataset, raw, render_v1

MODES = ("hierarchy_only", "latent_only", "dual")
VIEW_KEYS = {"event_frame", "voxel_grid", "time_surface"}
INPUT_CONTRACT = {
    "version": "dual-fusion-full-events-1",
    "point_contract_sha256": POINT_CONTRACT_SHA256,
    "renderer_sha256": RENDERER_SHA256,
    "alignment": "one decoded stored event array and one source identity for every branch",
    "window": "whole stored array, observed support, closed endpoints",
    "temporal_window_splitting": False,
    "event_filtering": "none", "sampling": "none", "augmentation": "none",
    "height": 480, "width": 640,
}
INPUT_CONTRACT_SHA256 = hashlib.sha256(json.dumps(INPUT_CONTRACT, sort_keys=True).encode()).hexdigest()


def input_keys(mode):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    return ((POINT_KEYS if mode != "latent_only" else {"event_counts"})
            | (VIEW_KEYS if mode != "hierarchy_only" else set()))


def prepare_inputs(fields, source, *, mode="dual"):
    """Use the unchanged native renderers; labels are not accepted by this API."""
    input_keys(mode)
    inputs = prepare_points(fields, source) if mode != "latent_only" else {
        "event_counts": torch.tensor([source.event_count], dtype=torch.long)}
    if mode != "hierarchy_only":
        inputs.update({key: torch.from_numpy(value)
                       for key, value in render_v1(fields, source, multiview=True).items()})
    return inputs


class DualFusionDataset(V1Dataset):
    def __init__(self, manifest_dir, dataset_root, split, *, mode="dual", raw_cache=None,
                 limit=None, seed=20260908):
        input_keys(mode)
        super().__init__(manifest_dir, dataset_root, split, multiview=mode != "hierarchy_only",
                         raw_cache=raw_cache, limit=limit, seed=seed)
        self.mode = mode

    def __getitem__(self, index):
        row = self.rows[index]
        start = time.perf_counter()
        fields = raw._decode_event_fields(self.payload(row), sample_id=row.sample_id)
        source = raw._source_identity(row, fields)
        decoded = time.perf_counter()
        inputs = prepare_inputs(fields, source, mode=self.mode)
        end = time.perf_counter()
        return {"inputs": inputs, "label": row.class_label, "sample_id": row.sample_id,
                "split": row.split, "source_split": row.source_split, "source": asdict(source),
                "decode_seconds": decoded - start, "render_seconds": end - decoded,
                "input_contract_sha256": INPUT_CONTRACT_SHA256}


def collate(samples, *, height=480, width=640):
    if not samples:
        raise ValueError("cannot collate an empty batch")
    keys = set(samples[0]["inputs"])
    if keys not in [input_keys(mode) for mode in MODES]:
        raise ValueError("invalid dual-fusion sample input keys")
    if any(set(sample["inputs"]) != keys for sample in samples):
        raise ValueError("cannot mix branch modes in one batch")
    cells = 8 * (height // 4) * (width // 4)
    inputs = {}
    for key in sorted(keys):
        values = [(sample["inputs"][key] + i * cells if key.startswith("voxel_") and key != "voxel_grid"
                   else sample["inputs"][key]) for i, sample in enumerate(samples)]
        inputs[key] = torch.stack(values) if key in VIEW_KEYS else torch.cat(values)
    return {"inputs": inputs, "labels": torch.tensor([s["label"] for s in samples], dtype=torch.long),
            "sample_ids": [s["sample_id"] for s in samples], "splits": [s["split"] for s in samples],
            "source_splits": [s["source_split"] for s in samples],
            "render_seconds": sum(s["render_seconds"] for s in samples),
            "decode_seconds": sum(s["decode_seconds"] for s in samples)}
