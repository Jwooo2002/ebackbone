"""V3: training-learned confidence control of a spatial TS residual."""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

from .hierarchy_ts_models import TS_INPUT_KEYS, profile_macs as ts_profile
from .hierarchy_ts_residual_models import HierarchyTSResidualV2

MODEL_VERSION = "hierarchy-ts-confidence-residual-v3-1"


class ConfidenceControl(nn.Module):
    """Monotone confidence control; the threshold learns only from final CE.

    No held-out routing threshold is reused. Initial threshold=.5, fixed
    temperature=.1. Detaching logits prevents confidence manipulation via this
    path; the same hierarchy/head still learn via the corrected final output.
    """

    def __init__(self):
        super().__init__()
        self.threshold_logit = nn.Parameter(torch.zeros(()))
        self.register_buffer("temperature", torch.tensor(.1))

    def forward(self, base_logits):
        probabilities = base_logits.detach().float().softmax(dim=1)
        top = probabilities.topk(2, dim=1).values
        margin = top[:, 0] - top[:, 1]
        strength = torch.sigmoid((self.threshold_logit.sigmoid() - margin) / self.temperature)
        return strength[:, None, None, None]


class HierarchyTSConfidenceV3(HierarchyTSResidualV2):
    name = "hierarchy_ts_confidence"

    def __init__(self, num_classes=100, height=480, width=640):
        super().__init__(num_classes, height, width)
        self.confidence_control = ConfidenceControl()

    def forward_features(self, inputs):
        if set(inputs) != TS_INPUT_KEYS:
            raise ValueError(f"hierarchy TS confidence requires exactly {sorted(TS_INPUT_KEYS)}")
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
        # Shared tail/head evaluated twice, not a separately parameterized head.
        # GroupNorm has no moving statistics. No auxiliary CE or base-path graph.
        # A no-grad autocast weight cache must not be reused by the trainable
        # second pass: that would silently detach the shared tail parameters.
        confidence_autocast = (nullcontext() if x.device.type == "meta" else
                               torch.autocast(x.device.type,
                                              enabled=torch.is_autocast_enabled(x.device.type),
                                              cache_enabled=False))
        with torch.no_grad(), confidence_autocast:
            base_logits = self.classifier(self.pool(self.frame_stage(x)).flatten(1))
        strength = self.confidence_control(base_logits)
        temporal = self.ts_encoder(surface)
        weight = self.ts_fusion.confidence(torch.cat((x, temporal), dim=1))
        correction = strength * weight * self.ts_fusion.residual(temporal)
        if self.capture_gate_stats:
            with torch.no_grad():
                w, f, d = weight.detach().float(), x.detach().float(), correction.detach().float()
                self.last_gate_stats = {
                    "threshold": float(self.confidence_control.threshold_logit.sigmoid()),
                    "strength_mean": float(strength.mean()),
                    "strength_min": float(strength.min()), "strength_max": float(strength.max()),
                    "weight_mean": float(w.mean()), "weight_std": float(w.std(unbiased=False)),
                    "weight_min": float(w.min()), "weight_max": float(w.max()),
                    "near_bound_fraction": float(((w < .01) | (w > .99)).float().mean()),
                    "relative_feature_change": float(d.norm() / f.norm().clamp_min(1e-12)),
                }
        return self.frame_stage(x + correction)


def profile_macs(model, event_count=116790):
    result = ts_profile(model, event_count)
    result["model_version"] = MODEL_VERSION
    result["shared_tail_forward_calls"] = 2
    result["confidence_control_shape"] = [1, 1, 1, 1]
    return result
