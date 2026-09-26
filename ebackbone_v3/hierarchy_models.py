"""Hierarchy-only V1: every event -> learned voxels -> frame features -> CE head."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .v1_models import BasicBlock, TemporalBlock, conv, norm

MODEL_VERSION = "hierarchy-v1-all-events-1"
INPUT_KEYS = {"points", "voxel_lower", "voxel_upper", "alpha", "event_counts"}


class PointToVoxel(nn.Module):
    """FP32 weighted mean of learned points plus log1p interpolated event mass.

    Indices use hard 4x4 spatial cells and linear interpolation in time. Only
    routing is fixed; all 32 feature channels backpropagate into the point MLP.
    """

    def __init__(self, height=480, width=640):
        super().__init__()
        self.height, self.width = height // 4, width // 4

    def forward(self, features, inputs):
        batch = inputs["event_counts"].shape[0]
        cells = batch * 8 * self.height * self.width
        values = torch.cat((features.float(), torch.ones_like(features[:, :1], dtype=torch.float32)), 1)
        alpha = inputs["alpha"].float().unsqueeze(1)
        sums = values.new_zeros(cells, 33)
        sums.index_add_(0, inputs["voxel_lower"], values * (1 - alpha))
        sums.index_add_(0, inputs["voxel_upper"], values * alpha)
        mass = sums[:, 32:]
        values = torch.cat((sums[:, :32] / mass.clamp_min(1e-6), mass.log1p()), 1)
        return values.view(batch, 8, self.height, self.width, 33).permute(0, 4, 1, 2, 3).contiguous()


class HierarchyV1(nn.Module):
    name = "hierarchy"

    def __init__(self, num_classes=100, height=480, width=640):
        super().__init__()
        if num_classes < 2 or min(height, width) < 32 or height % 32 or width % 32:
            raise ValueError("require >=2 classes and H,W positive multiples of 32")
        self.num_classes, self.height, self.width = num_classes, height, width
        self.point = nn.Sequential(nn.Linear(4, 32), nn.LayerNorm(32), nn.SiLU(),
                                   nn.Linear(32, 32), nn.LayerNorm(32), nn.SiLU())
        self.point_to_voxel = PointToVoxel(height, width)
        self.voxel_projection = nn.Sequential(conv(33, 32, 1, temporal=True), norm(32), nn.SiLU())
        self.voxel_stage1 = TemporalBlock(32, 64, stride=2)
        self.voxel_stage2 = TemporalBlock(64, 128, stride=2)
        self.temporal_collapse = nn.Sequential(conv(128 * 8, 128, 1), norm(128), nn.SiLU())
        self.frame_stage = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward_features(self, inputs):
        if set(inputs) != INPUT_KEYS:
            raise ValueError(f"hierarchy requires exactly {sorted(INPUT_KEYS)}")
        points = inputs["points"]
        if points.ndim != 2 or points.shape[1] != 4 or points.shape[0] == 0:
            raise ValueError("points must be nonempty packed [sum_N,4]")
        for key in ("voxel_lower", "voxel_upper", "alpha"):
            if inputs[key].shape != points.shape[:1]:
                raise ValueError(f"{key} must align with every point")
        if inputs["event_counts"].ndim != 1 or inputs["event_counts"].numel() == 0:
            raise ValueError("event_counts must be [B]")
        x = self.point_to_voxel(self.point(points), inputs)
        x = self.voxel_stage2(self.voxel_stage1(self.voxel_projection(x)))
        # C-major, temporal-bin-minor: ordered learned collapse, not time mean.
        x = self.temporal_collapse(x.flatten(1, 2))
        return self.frame_stage(x)

    def forward(self, inputs):
        return self.classifier(self.pool(self.forward_features(inputs)).flatten(1))


def example_inputs(event_count=116790, *, height=480, width=640, device="meta"):
    if event_count <= 0:
        raise ValueError("event_count must be positive")
    return {"points": torch.zeros(event_count, 4, device=device),
            "voxel_lower": torch.zeros(event_count, dtype=torch.long, device=device),
            "voxel_upper": torch.zeros(event_count, dtype=torch.long, device=device),
            "alpha": torch.zeros(event_count, device=device),
            "event_counts": torch.tensor([event_count], device=device)}


def profile_macs(model, event_count=116790):
    """Executed meta shape propagation; point MACs depend on actual event count."""
    probe = copy.deepcopy(model).to("meta").eval()
    stages, macs, handles = {}, {}, []

    def count(name):
        def hook(module, args, output):
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                cost = output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
            else:
                cost = output.numel() * module.in_features
            stage = name.split(".")[0]
            macs[stage] = macs.get(stage, 0) + cost
        return hook

    def shape(name):
        def hook(module, args, output):
            stages[name] = list(output.shape)
        return hook

    for name, module in probe.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            handles.append(module.register_forward_hook(count(name)))
        if name and "." not in name:
            handles.append(module.register_forward_hook(shape(name)))
    try:
        with torch.no_grad():
            probe(example_inputs(event_count, height=model.height, width=model.width))
    finally:
        for handle in handles:
            handle.remove()
    total = sum(macs.values())
    return {"model_version": MODEL_VERSION, "parameters": sum(p.numel() for p in model.parameters()),
            "parameters_by_stage": {name: sum(p.numel() for p in module.parameters())
                                    for name, module in model.named_children()},
            "macs": total, "gmacs": total / 1e9, "macs_by_stage": macs,
            "fixed_macs": total - macs["point"], "point_macs_per_event": 4 * 32 + 32 * 32,
            "event_count": event_count, "input_hw": [model.height, model.width],
            "shapes_batch1": stages,
            "mac_scope": "Conv2d, Conv3d, Linear only; batch=1; excludes scatter/interpolation, normalization, activation, pooling, preprocessing and I/O"}
