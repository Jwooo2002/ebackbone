"""Versioned Mini V1 rendering over the existing immutable split manifests.

Only the existing raw decoding/identity utilities are reused. D012 caches and
renderer defaults are not modified or accepted by this path.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import n_imagenet_mini_dataset as raw
from .representations import SourceIdentity, _validated_fields

RENDERER = {
    "version": "mini-v1-8bins-tau02-1", "height": 480, "width": 640,
    "voxel_bins": 8, "tau_duration_ratio": 0.2, "polarity_order": ["negative", "positive"],
    "frame_normalization": "log1p", "voxel_normalization": "log1p",
    "time_surface_background": 0, "window": "whole stored array, observed support, closed endpoints",
    "spatial_mapping": "native_identity", "augmentation": "none", "event_filtering": "none",
    "voxel_binning": "linear_temporal_interpolation", "dtype": "float32",
}
RENDERER_SHA256 = hashlib.sha256(json.dumps(RENDERER, sort_keys=True).encode()).hexdigest()


def render_v1(fields, source: SourceIdentity, *, multiview: bool):
    x, y, t, p = _validated_fields(fields, source)
    x, y, p = (a.astype(np.int64) for a in (x, y, p))
    counts = np.zeros((2, 480, 640), np.float32)
    np.add.at(counts, (p, y, x), 1)
    tensors = {"event_frame": np.log1p(counts)}
    if not multiview:
        return tensors
    span = source.temporal_end - source.temporal_start
    normalized = ((t.astype(np.float64) - source.temporal_start) / span
                  if span else np.ones(len(t), np.float64))
    position = normalized * 7
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, 7)
    alpha = (position - lower).astype(np.float32)
    voxels = np.zeros((2, 8, 480, 640), np.float32)
    np.add.at(voxels, (p, lower, y, x), 1 - alpha)
    np.add.at(voxels, (p, upper, y, x), alpha)
    latest = np.full((2, 480, 640), -np.inf, np.float32)
    np.maximum.at(latest, (p, y, x), normalized.astype(np.float32))
    surface = np.zeros_like(latest)
    occupied = np.isfinite(latest)
    surface[occupied] = np.exp(-(1 - latest[occupied]) / np.float32(0.2))
    tensors.update(voxel_grid=np.log1p(voxels), time_surface=surface)
    return tensors


class V1Dataset(Dataset):
    def __init__(self, manifest_dir, dataset_root, split, *, multiview, raw_cache=None,
                 allow_final_test=False, limit=None, seed=20260908):
        # open_dataset gates test access before reading any test manifest. It
        # verifies Mini schema, canonical manifest hashes and row membership.
        adapter = raw.open_dataset(manifest_dir=manifest_dir, dataset_root=dataset_root,
                                   split=split, baseline="b0", cache="off",
                                   allow_final_test=allow_final_test)
        self.rows = adapter._rows
        self.root = Path(dataset_root).resolve()
        self.split = split
        self.multiview = multiview
        self.raw_cache = Path(raw_cache).resolve() if raw_cache else None
        self.manifest_sha256 = hashlib.sha256(adapter.manifest_path.read_bytes()).hexdigest()
        if limit is not None:
            if limit <= 0 or limit > len(self.rows):
                raise ValueError("diagnostic limit must be positive and within split size")
            # Label-free selection; only diagnostic mode supplies a limit.
            self.rows = tuple(sorted(self.rows, key=lambda r: hashlib.sha256(
                f"v1-diagnostic\0{seed}\0{r.sample_id}".encode()).digest())[:limit])

    def __len__(self):
        return len(self.rows)

    def payload(self, row):
        path = self.raw_cache / f"{row.raw_content_sha256}.npz" if self.raw_cache else None
        if path is not None:
            if not path.is_file():
                raise FileNotFoundError(f"raw cache incomplete; run prepare-raw first: {path}")
            payload = path.read_bytes()
        else:
            payload = raw._read_archive_payload(self.root, row)
        raw._validate_raw_payload(payload, row)
        return payload

    def __getitem__(self, index):
        row = self.rows[index]
        start = time.perf_counter()
        fields = raw._decode_event_fields(self.payload(row), sample_id=row.sample_id)
        source = raw._source_identity(row, fields)
        decoded = time.perf_counter()
        tensors = render_v1(fields, source, multiview=self.multiview)
        end = time.perf_counter()
        return {"inputs": {k: torch.from_numpy(v) for k, v in tensors.items()},
                "label": row.class_label, "sample_id": row.sample_id,
                "split": row.split, "source_split": row.source_split,
                "decode_seconds": decoded - start, "render_seconds": end - decoded,
                "source": asdict(source), "renderer_sha256": RENDERER_SHA256}


def prepare_raw_cache(manifest_dir, dataset_root, cache_root):
    """Read each nested class TAR once; cache verified NPZ bytes, train/val only.

    A cache is content addressed by the immutable row's exact payload SHA-256.
    Existing bytes are rechecked; no final-test rows or representations are read.
    """
    cache = Path(cache_root).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    groups = {}
    for split in ("train", "validation"):
        dataset = V1Dataset(manifest_dir, dataset_root, split, multiview=False)
        for row in dataset.rows:
            groups.setdefault((row.source_archive, row.source_members[0]), {})[row.source_members[1]] = row
    written = existing = 0
    for (archive, member), wanted in sorted(groups.items()):
        missing = {}
        for name, row in wanted.items():
            path = cache / f"{row.raw_content_sha256}.npz"
            if path.exists():
                raw._validate_raw_payload(path.read_bytes(), row)
                existing += 1
            else:
                missing[name] = row
        if missing:
            with zipfile.ZipFile(Path(dataset_root) / archive) as z:
                with z.open(member) as stream, tarfile.open(fileobj=stream, mode="r|gz") as tar:
                    for info in tar:
                        if not info.isfile() or info.name not in missing:
                            continue
                        row = missing.pop(info.name)
                        payload = tar.extractfile(info).read()
                        raw._validate_raw_payload(payload, row)
                        path = cache / f"{row.raw_content_sha256}.npz"
                        temporary = path.with_suffix(f".{os.getpid()}.tmp")
                        temporary.write_bytes(payload)
                        os.replace(temporary, path)
                        written += 1
            if missing:
                raise ValueError(f"missing members in {archive}!{member}: {len(missing)}")
        print(json.dumps({"archive": archive, "class_member": member,
                          "written": written, "verified_existing": existing}), flush=True)
    return {"written": written, "verified_existing": existing, "test_accessed": False}
