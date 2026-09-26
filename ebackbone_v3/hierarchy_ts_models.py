"""Hierarchy V1 with identity-initialized, recency-conditioned frame gating."""
from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .hierarchy_models import HierarchyV1, INPUT_KEYS, example_inputs as point_example

MODEL_VERSION = "hierarchy-ts-gate-v1-1"
TS_INPUT_KEYS = INPUT_KEYS | {"time_surface"}


class HierarchyTSV1(HierarchyV1):
    name = "hierarchy_ts"

    def __init__(self, num_classes=100, height=480, width=640):
        # Construct the complete original backbone first: same seed -> same weights.
        super().__init__(num_classes, height, width)
        with torch.random.fork_rng(devices=[]):
            blocks = []
            for incoming, outgoing in ((2, 16), (16, 32), (32, 64), (64, 64)):
                blocks.append(nn.Sequential(
                    nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(8, outgoing), nn.SiLU()))
            self.ts_encoder = nn.Sequential(*blocks)
            self.ts_gate = nn.Conv2d(64, 128, 1, bias=True)
            nn.init.zeros_(self.ts_gate.weight)
            nn.init.zeros_(self.ts_gate.bias)
        self.capture_gate_stats = False
        self.last_gate_stats = None

    def forward_features(self, inputs):
        if set(inputs) != TS_INPUT_KEYS:
            raise ValueError(f"hierarchy TS requires exactly {sorted(TS_INPUT_KEYS)}")
        points = inputs["points"]
        if points.ndim != 2 or points.shape[1] != 4 or points.shape[0] == 0:
            raise ValueError("points must be nonempty packed [sum_N,4]")
        for key in ("voxel_lower", "voxel_upper", "alpha"):
            if inputs[key].shape != points.shape[:1]:
                raise ValueError(f"{key} must align with every point")
        counts = inputs["event_counts"]
        if counts.ndim != 1 or counts.numel() == 0:
            raise ValueError("event_counts must be [B]")
        surface = inputs["time_surface"]
        if surface.shape != (counts.numel(), 2, self.height, self.width):
            raise ValueError("time_surface must be [B,2,H,W] aligned with packed points")
        x = self.point_to_voxel(self.point(points), inputs)
        x = self.voxel_stage2(self.voxel_stage1(self.voxel_projection(x)))
        x = self.temporal_collapse(x.flatten(1, 2))
        logits = self.ts_gate(self.ts_encoder(surface))
        multiplier = 1 + 0.5 * torch.tanh(logits)
        gated = x * multiplier
        if self.capture_gate_stats:
            with torch.no_grad():
                m, f = multiplier.detach().float(), x.detach().float()
                self.last_gate_stats = {
                    "multiplier_mean": float(m.mean()), "multiplier_std": float(m.std(unbiased=False)),
                    "multiplier_min": float(m.min()), "multiplier_max": float(m.max()),
                    "near_bound_fraction": float(((m < .51) | (m > 1.49)).float().mean()),
                    "relative_feature_change": float((gated.detach().float() - f).norm() / f.norm().clamp_min(1e-12)),
                }
        return self.frame_stage(gated)


def example_inputs(event_count=116790, *, height=480, width=640, device="meta"):
    return {**point_example(event_count, height=height, width=width, device=device),
            "time_surface": torch.zeros(1, 2, height, width, device=device)}


def profile_macs(model, event_count=116790):
    probe = copy.deepcopy(model).to("meta").eval()
    probe.capture_gate_stats = False
    macs, shapes, handles = {}, {}, []

    def count(name):
        def hook(module, args, output):
            cost = output.numel() * (module.in_features if isinstance(module, nn.Linear) else
                                    module.in_channels // module.groups * math.prod(module.kernel_size))
            stage = name.split(".")[0]
            macs[stage] = macs.get(stage, 0) + cost
        return hook

    def shape(name):
        def hook(module, args, output):
            shapes[name] = list(output.shape)
        return hook

    for name, module in probe.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            handles.append(module.register_forward_hook(count(name)))
        if name and ("." not in name or name.startswith("ts_encoder.") and name.count(".") == 1):
            handles.append(module.register_forward_hook(shape(name)))
    try:
        with torch.no_grad():
            probe(example_inputs(event_count, height=model.height, width=model.width))
    finally:
        for handle in handles:
            handle.remove()
    total = sum(macs.values())
    return {"model_version": MODEL_VERSION, "parameters": sum(p.numel() for p in model.parameters()),
            "parameters_by_stage": {n: sum(p.numel() for p in m.parameters()) for n, m in model.named_children()},
            "macs": total, "gmacs": total / 1e9, "macs_by_stage": macs,
            "fixed_macs": total - macs["point"], "point_macs_per_event": 1152,
            "event_count": event_count, "input_hw": [model.height, model.width], "shapes_batch1": shapes,
            "mac_scope": "Conv2d/Conv3d/Linear only; batch=1; excludes rendering, scatter, normalization, activation, pooling, elementwise gating and I/O"}
