"""Content-aware residual TS fusion; the original hierarchy path stays intact."""
from __future__ import annotations

import torch
from torch import nn

from .hierarchy_ts_models import HierarchyTSV1, TS_INPUT_KEYS, profile_macs as gate_profile

MODEL_VERSION = "hierarchy-ts-content-residual-v2-1"


class TSResidualFusion(nn.Module):
    """F + sigmoid(q([F,T])) * r(T); one spatial weight for 128 correction channels.

    The zero residual projection gives exact identity initially. Its first update
    opens task gradients into both the TS encoder and the confidence network.
    The sigmoid is a learned fusion weight, not a calibrated probability.
    """

    def __init__(self):
        super().__init__()
        self.residual = nn.Conv2d(64, 128, 1, bias=True)
        self.confidence = nn.Sequential(
            nn.Conv2d(192, 32, 1, bias=False), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 1, 1, bias=True), nn.Sigmoid())
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.capture_stats = False
        self.last_stats = None

    def forward(self, features, temporal):
        weight = self.confidence(torch.cat((features, temporal), dim=1))
        correction = weight * self.residual(temporal)
        output = features + correction
        if self.capture_stats:
            with torch.no_grad():
                w, f, d = weight.detach().float(), features.detach().float(), correction.detach().float()
                self.last_stats = {
                    "weight_mean": float(w.mean()), "weight_std": float(w.std(unbiased=False)),
                    "weight_min": float(w.min()), "weight_max": float(w.max()),
                    "near_bound_fraction": float(((w < .01) | (w > .99)).float().mean()),
                    "relative_feature_change": float(d.norm() / f.norm().clamp_min(1e-12)),
                }
        return output


class HierarchyTSResidualV2(HierarchyTSV1):
    name = "hierarchy_ts_residual"

    def __init__(self, num_classes=100, height=480, width=640):
        super().__init__(num_classes, height, width)
        # Reuse the identical hierarchy and TS encoder initialization, then
        # remove V1's multiplicative head entirely (no unused parameters).
        del self.ts_gate
        with torch.random.fork_rng(devices=[]):
            self.ts_fusion = TSResidualFusion()

    def forward_features(self, inputs):
        if set(inputs) != TS_INPUT_KEYS:
            raise ValueError(f"hierarchy TS residual requires exactly {sorted(TS_INPUT_KEYS)}")
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
        self.ts_fusion.capture_stats = self.capture_gate_stats
        x = self.ts_fusion(x, self.ts_encoder(surface))
        if self.capture_gate_stats:
            self.last_gate_stats = self.ts_fusion.last_stats
        return self.frame_stage(x)


def profile_macs(model, event_count=116790):
    result = gate_profile(model, event_count)
    result["model_version"] = MODEL_VERSION
    return result
