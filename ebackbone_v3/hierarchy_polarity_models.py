"""Polarity-separated voxel summaries; unchanged point MLP and downstream paths."""
from __future__ import annotations

import torch
from torch import nn

from .hierarchy_models import HierarchyV1, profile_macs as hierarchy_profile
from .hierarchy_ts_models import HierarchyTSV1, profile_macs as ts_profile
from .hierarchy_ts_residual_models import HierarchyTSResidualV2
from .hierarchy_ts_confidence_models import HierarchyTSConfidenceV3
from .v1_models import conv

MODEL_VERSION = "hierarchy-polarity-separated-voxels-1"
BASE_MODELS = {
    "hierarchy": HierarchyV1,
    "hierarchy_ts": HierarchyTSV1,
    "hierarchy_ts_residual": HierarchyTSResidualV2,
    "hierarchy_ts_confidence": HierarchyTSConfidenceV3,
}
POLARITY_CONTRACT = {
    "version": MODEL_VERSION,
    "order": ["negative", "positive"],
    "point_mlp": "unchanged shared 4->32->32 MLP; signed polarity remains an input",
    "channels_per_polarity": 33,
    "summary": "32D weighted feature mean plus log1p interpolated event mass, separately per polarity",
    "empty_polarity_cell": "all 33 channels zero",
    "routing": "original 4x4 spatial cells and eight-bin linear temporal interpolation",
    "fusion": "concatenate negative33 and positive33; Conv3d66->32 kernel1 biasFalse, existing GroupNorm and SiLU",
    "point_sampling": "none",
    "initialization": "fresh widened fusion convolution; original point/downstream/TS seed weights and RNG preserved",
}


class PolarityPointToVoxel(nn.Module):
    """FP32 accumulation into separate negative/positive banks before any fusion."""
    def __init__(self, height=480, width=640):
        super().__init__()
        self.height, self.width = height // 4, width // 4

    def forward(self, features, inputs):
        points = inputs["points"]
        if features.ndim != 2 or features.shape[1] != 32 or points.shape != (features.shape[0], 4):
            raise ValueError("require aligned [M,32] features and [M,4] points")
        polarity = points[:, 3]
        if points.device.type != "meta" and not bool(((polarity == -1) | (polarity == 1)).all()):
            raise ValueError("polarity must be signed -1 or +1 as in prepare_points")
        batch = inputs["event_counts"].shape[0]
        cells = batch * 8 * self.height * self.width
        offset = (polarity > 0).long() * cells
        values = torch.cat((features.float(), torch.ones_like(features[:, :1], dtype=torch.float32)), dim=1)
        alpha = inputs["alpha"].float().unsqueeze(1)
        sums = values.new_zeros(2 * cells, 33)
        sums.index_add_(0, inputs["voxel_lower"] + offset, values * (1 - alpha))
        sums.index_add_(0, inputs["voxel_upper"] + offset, values * alpha)
        mass = sums[:, 32:]
        values = torch.cat((sums[:, :32] / mass.clamp_min(1e-6), mass.log1p()), dim=1)
        # [polarity,B,T,H,W,C] -> [B,polarity,C,T,H,W] -> polarity-major channels.
        return (values.view(2, batch, 8, self.height, self.width, 33)
                .permute(1, 0, 5, 2, 3, 4).reshape(batch, 66, 8, self.height, self.width))


def make_model(base="hierarchy", num_classes=100, height=480, width=640):
    """Construct a NEW model, altering only voxel aggregation and its input fusion.

    Reuse the selected base forward implementation verbatim, including its TS
    integration. Existing model instances, source files, and checkpoints are not
    changed. The widened projection is deliberately incompatible with a strict
    unmodified baseline checkpoint load.
    """
    if base not in BASE_MODELS:
        raise ValueError(f"base must be one of {tuple(BASE_MODELS)}")
    model = BASE_MODELS[base](num_classes, height, width)
    model.point_to_voxel = PolarityPointToVoxel(height, width)
    with torch.random.fork_rng(devices=[]):
        model.voxel_projection[0] = conv(66, 32, 1, temporal=True)
    model.polarity_base = base
    model.name = base + "_polarity"
    return model


def profile_macs(model, event_count=116790):
    profile = hierarchy_profile if model.polarity_base == "hierarchy" else ts_profile
    result = profile(model, event_count)
    result["model_version"] = MODEL_VERSION
    result["polarity_base"] = model.polarity_base
    result["polarity_contract"] = POLARITY_CONTRACT
    result["added_parameters_vs_selected_base"] = 33 * 32
    result["added_macs_vs_selected_base"] = 33 * 32 * 8 * (model.height // 4) * (model.width // 4)
    result["mac_scope"] += "; polarity scatter and additional activation memory excluded"
    return result
