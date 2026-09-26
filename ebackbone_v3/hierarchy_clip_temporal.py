"""Shared hierarchy sequences and ordered temporal baseline; no training launch."""

from __future__ import annotations

import torch
from torch import nn

from .hierarchy_models import HierarchyV1, INPUT_KEYS


class HierarchySequenceEncoder(nn.Module):
    """Run the original point -> voxel -> frame hierarchy for B*K windows."""

    feature_dim = 256

    def __init__(self, hierarchy: HierarchyV1, num_windows=4):
        super().__init__()
        if isinstance(num_windows, bool) or not isinstance(num_windows, int) or num_windows < 1:
            raise ValueError("num_windows must be a positive integer")
        self.hierarchy = hierarchy
        self.num_windows = num_windows

    def train(self, mode=True):
        super().train(mode)
        if mode:
            # A later stage can unfreeze just temporal_collapse/frame_stage.
            # Keep all remaining frozen hierarchy blocks in evaluation mode.
            if not any(p.requires_grad for p in self.hierarchy.parameters()):
                self.hierarchy.eval()
            else:
                for module in self.hierarchy.children():
                    if not any(p.requires_grad for p in module.parameters()):
                        module.eval()
        return self

    def forward(self, inputs):
        if set(inputs) != {"packed", "window_mask"}:
            raise ValueError("sequence inputs require exactly packed and window_mask")
        packed, mask = inputs["packed"], inputs["window_mask"]
        if mask.ndim != 2 or mask.dtype != torch.bool or mask.shape[1] != self.num_windows or mask.shape[0] == 0:
            raise ValueError("window_mask must be boolean [B,K] matching num_windows")
        if set(packed) != INPUT_KEYS:
            raise ValueError("packed windows must contain exactly the hierarchy input keys")
        if any(value.device != mask.device for value in packed.values()):
            raise ValueError("all packed tensors and window_mask must share a device")
        counts = packed["event_counts"]
        if counts.dtype != torch.long or counts.shape != (mask.numel(),) or bool((counts < 0).any()):
            raise ValueError("event_counts must be nonnegative int64 [B*K]")
        if not torch.equal(counts > 0, mask.flatten()):
            raise ValueError("window mask must agree with packed event counts")
        points = packed["points"]
        if points.ndim != 2 or points.shape[1] != 4 or points.dtype != torch.float32 or not bool(torch.isfinite(points).all()):
            raise ValueError("packed points must be finite float32 [N,4]")
        event_count = int(counts.sum())
        if event_count != points.shape[0]:
            raise ValueError("event counts must conserve all packed points")
        alpha = packed["alpha"]
        if alpha.shape != (event_count,) or alpha.dtype != torch.float32 or not bool(torch.isfinite(alpha).all()) or bool(((alpha < 0) | (alpha > 1)).any()):
            raise ValueError("alpha must be finite float32 [N] interpolation weights in [0,1]")
        cells = 8 * (self.hierarchy.height // 4) * (self.hierarchy.width // 4)
        owners = torch.repeat_interleave(torch.arange(mask.numel(), device=mask.device), counts)
        for key in ("voxel_lower", "voxel_upper"):
            route = packed[key]
            if route.dtype != torch.long or route.shape != (event_count,):
                raise ValueError(f"{key} must be int64 [N]")
            # Counts define sample-major/window-minor event ordering. No route
            # may accidentally deposit points into another sample or window.
            if not torch.equal(torch.div(route, cells, rounding_mode="floor"), owners):
                raise ValueError(f"{key} routes must stay inside their own sample/window")
        if event_count == 0:
            return points.new_zeros(mask.shape[0], self.num_windows, 256,
                                    self.hierarchy.height // 32, self.hierarchy.width // 32)
        maps = self.hierarchy.forward_features(packed)
        expected = (mask.numel(), 256, self.hierarchy.height // 32, self.hierarchy.width // 32)
        if tuple(maps.shape) != expected:
            raise ValueError(f"hierarchy map contract mismatch: expected {expected}, got {tuple(maps.shape)}")
        maps = maps.reshape(mask.shape[0], self.num_windows, *maps.shape[1:])
        return maps.masked_fill(~mask[:, :, None, None, None], 0)


class TemporalAggregator(nn.Module):
    """Small position-aware transformer with padding exclusion and masked mean.

    An entirely empty sequence returns exactly zero. Empty windows cannot affect
    valid queries, and arbitrary values at masked positions are ignored.
    """

    def __init__(self, dim, num_windows=4, *, heads=4, layers=1, dropout=0.0):
        super().__init__()
        if min(dim, num_windows, heads, layers) < 1 or dim % heads:
            raise ValueError("positive dims/layers required and dim must divide heads")
        self.dim, self.num_windows = dim, num_windows
        self.position = nn.Parameter(torch.empty(1, num_windows, dim))
        nn.init.normal_(self.position, std=0.02)
        block = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=dim * 2,
                                          dropout=dropout, activation="gelu",
                                          batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, features, window_mask):
        if features.ndim != 3 or features.shape[1:] != (self.num_windows, self.dim):
            raise ValueError("temporal features must be [B,K,D] matching configured dimensions")
        if window_mask.dtype != torch.bool or tuple(window_mask.shape) != tuple(features.shape[:2]):
            raise ValueError("window_mask must be boolean [B,K]")
        if window_mask.device != features.device:
            raise ValueError("features and window_mask must share a device")
        safe_mask = window_mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True  # avoid softmax over entirely masked keys
        values = features.masked_fill(~window_mask[..., None], 0) + self.position
        values = self.transformer(values, src_key_padding_mask=~safe_mask)
        values = self.norm(values).masked_fill(~window_mask[..., None], 0)
        return values.sum(dim=1) / window_mask.sum(dim=1, keepdim=True).clamp_min(1)


class HierarchyTemporalBaseline(nn.Module):
    """Comparison A: hierarchy spatial mean -> temporal module -> linear CE head."""

    def __init__(self, encoder: HierarchySequenceEncoder, *, num_classes=100, temporal=None):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        self.encoder = encoder
        self.temporal = temporal if temporal is not None else TemporalAggregator(256, encoder.num_windows)
        self.classifier = nn.Linear(256, num_classes)

    def forward_features(self, inputs):
        maps = self.encoder(inputs)
        return self.temporal(maps.mean(dim=(-2, -1)), inputs["window_mask"])

    def forward(self, inputs):
        return self.classifier(self.forward_features(inputs))
