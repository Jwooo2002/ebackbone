"""Independent hierarchy ablations: local events, self-only control, early skip."""
from __future__ import annotations

import torch
from torch import nn

from .hierarchy_models import HierarchyV1, INPUT_KEYS, profile_macs as base_profile

MODEL_VERSION = "hierarchy-local-skip-controls-1"
MODEL_NAMES = ("hierarchy_local", "hierarchy_early_skip", "hierarchy_self_control")


class SpatialMeanDownsample(nn.Module):
    """Disjoint 2x2 means with deterministic CUDA backward; preserve time."""
    def forward(self, x):
        return (x[..., 0::2, 0::2] + x[..., 0::2, 1::2] +
                x[..., 1::2, 0::2] + x[..., 1::2, 1::2]) * .25


class EventInteraction(nn.Module):
    """Four bounded temporal neighbors within each original 4x4/time bucket.

    Same-width self-only control executes four messages with the same network,
    using (q_i,q_i,x_i,y_i,t_i,p_i) instead of (q_i,q_j,relative coordinates).
    All events remain in the original weighted voxel aggregation.
    """
    offsets = (-2, -1, 1, 2)

    def __init__(self, *, local=True, height=480, width=640):
        super().__init__()
        self.local = local
        self.reduce = nn.Linear(32, 8)
        self.message = nn.Sequential(nn.Linear(20, 16), nn.LayerNorm(16), nn.SiLU())
        self.project = nn.Linear(16, 32)
        self.register_buffer("relative_scale", torch.tensor([(width - 1) / 4, (height - 1) / 4, 7., .5]))
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, features, points, bucket):
        q = self.reduce(features)
        if not self.local:
            edge = torch.cat((q, q, points.to(q.dtype)), dim=1)
            # Intentional repeated operations: match local neural MACs exactly.
            messages = sum(self.message(edge) for _ in self.offsets) / len(self.offsets)
            return features + self.project(messages)
        with torch.no_grad():
            chronological = torch.argsort(points[:, 2], stable=True)
            order = chronological[torch.argsort(bucket[chronological], stable=True)]
            group, coordinates = bucket[order], points[order]
            rank = torch.arange(len(points), device=points.device)
        q = q[order]
        messages = q.new_zeros((len(points), 16))
        count = q.new_zeros((len(points), 1))
        for offset in self.offsets:
            other_q = torch.roll(q, -offset, dims=0)
            relative = (torch.roll(coordinates, -offset, dims=0) - coordinates) * self.relative_scale
            valid = ((rank + offset >= 0) & (rank + offset < len(points)) &
                     (group == torch.roll(group, -offset, dims=0)))
            mask = valid[:, None].to(q.dtype)
            edge = torch.cat((q, other_q, relative.to(q.dtype)), dim=1)
            messages = messages + self.message(edge) * mask
            count = count + mask
        correction = self.project(messages / count.clamp_min(1)) * (count > 0)
        # Original event order and all-event contribution are preserved.
        correction = torch.zeros_like(correction).index_copy(0, order, correction)
        return features + correction


class HierarchyAblation(HierarchyV1):
    def __init__(self, name, num_classes=100, height=480, width=640):
        if name not in MODEL_NAMES:
            raise ValueError("unknown hierarchy ablation")
        super().__init__(num_classes, height, width)
        self.name = name
        with torch.random.fork_rng(devices=[]):
            if name == "hierarchy_early_skip":
                self.early_skip = nn.Sequential(SpatialMeanDownsample(), nn.Flatten(1, 2),
                                                nn.Conv2d(64 * 8, 128, 1))
                nn.init.zeros_(self.early_skip[-1].weight)
                nn.init.zeros_(self.early_skip[-1].bias)
            else:
                self.event_interaction = EventInteraction(local=name == "hierarchy_local", height=height, width=width)

    def forward_features(self, inputs):
        if set(inputs) != INPUT_KEYS:
            raise ValueError(f"hierarchy ablations require exactly {sorted(INPUT_KEYS)}")
        points = inputs["points"]
        if points.ndim != 2 or points.shape[1] != 4 or points.shape[0] == 0:
            raise ValueError("points must be nonempty packed [sum_N,4]")
        for key in ("voxel_lower", "voxel_upper", "alpha"):
            if inputs[key].shape != points.shape[:1]:
                raise ValueError(f"{key} must align with every point")
        if inputs["event_counts"].ndim != 1 or inputs["event_counts"].numel() == 0:
            raise ValueError("event_counts must be [B]")
        features = self.point(points)
        if hasattr(self, "event_interaction"):
            features = self.event_interaction(features, points, inputs["voxel_lower"])
        x = self.point_to_voxel(features, inputs)
        early = self.voxel_stage1(self.voxel_projection(x))
        x = self.voxel_stage2(early)
        x = self.temporal_collapse(x.flatten(1, 2))
        if hasattr(self, "early_skip"):
            x = x + self.early_skip(early)
        return self.frame_stage(x)


def make_model(name, num_classes=100, height=480, width=640):
    return HierarchyAblation(name, num_classes, height, width)


def profile_macs(model, event_count=116790):
    result = base_profile(model, event_count)
    result["model_version"] = MODEL_VERSION
    result["variant"] = model.name
    variable = result["macs_by_stage"].get("event_interaction", 0)
    result["fixed_macs"] -= variable
    result["point_macs_per_event"] += variable // event_count
    result["mac_scope"] += "; excludes local sorting, neighbor gather, masking and relative-coordinate arithmetic"
    return result
