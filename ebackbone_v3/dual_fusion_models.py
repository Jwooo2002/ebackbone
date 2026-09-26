"""Two independently learned embeddings, convex scalar fusion, one CE classifier."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch import nn

from .dual_fusion_data import INPUT_CONTRACT_SHA256, VIEW_KEYS, input_keys
from .hierarchy_models import HierarchyV1, INPUT_KEYS as POINT_KEYS, example_inputs as point_example
from .v1_models import Branch

MODEL_VERSION = "hierarchy-latent-dual-fusion-1"
EXPORT_VERSION = "dual-fusion-backbone-export-1"


def _seeded(seed, factory):
    # Seed only CPU initialization and restore its RNG; construction must not
    # advance caller RNG or alter any already initialized CUDA generators.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return factory()


class LatentBranch(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.frame = Branch("frame", width=width)
        self.voxel = Branch("voxel", width=width)
        self.surface = Branch("surface", width=width)
        self.fusion = nn.Sequential(nn.Linear(12 * width, 256), nn.SiLU(), nn.Linear(256, 256))

    def forward(self, inputs):
        features = [self.frame(inputs["event_frame"]), self.voxel(inputs["voxel_grid"]),
                    self.surface(inputs["time_surface"])]
        return self.fusion(torch.cat([x.mean(dim=(-2, -1)) for x in features], dim=1))


class DualFusionBackbone(nn.Module):
    """Ablations remove inactive modules; identical components share initial weights.

    LayerNorm after each independent projection controls branch scale while
    retaining a conventional learned CE classifier. It is feature normalization,
    not an L2-unit-vector constraint or cosine classifier.
    """

    def __init__(self, mode="dual", num_classes=100, height=480, width=640,
                 dimension=256, seed=20260908, latent_width=8, *, include_classifier=True):
        super().__init__()
        input_keys(mode)
        if (num_classes < 2 or dimension < 1 or min(height, width) < 32 or height % 32 or width % 32
                or latent_width < 8 or latent_width % 8):
            raise ValueError("require >=2 classes, positive dimension, H,W positive multiples of 32")
        self.mode, self.num_classes, self.dimension = mode, num_classes, dimension
        self.height, self.width, self.seed = height, width, seed
        self.latent_width = latent_width
        if mode != "latent_only":
            self.hierarchy = _seeded(seed, lambda: HierarchyV1(num_classes, height, width))
            del self.hierarchy.classifier
            self.hierarchy_projection = _seeded(seed + 1, lambda: nn.Sequential(
                nn.Linear(256, dimension), nn.LayerNorm(dimension)))
        if mode != "hierarchy_only":
            self.latent = _seeded(seed + 2, lambda: LatentBranch(latent_width))
            self.latent_projection = _seeded(seed + 3, lambda: nn.Sequential(
                nn.Linear(256, dimension), nn.LayerNorm(dimension)))
        if mode == "dual":
            self.a = nn.Parameter(torch.tensor(0.))
        if include_classifier:
            self.classifier = _seeded(seed + 4, lambda: nn.Linear(dimension, num_classes))

    def construction_config(self):
        return {"mode": self.mode, "num_classes": self.num_classes, "height": self.height,
                "width": self.width, "dimension": self.dimension, "seed": self.seed,
                "latent_width": self.latent_width,
                "include_classifier": hasattr(self, "classifier")}

    def _validate(self, inputs):
        if set(inputs) != input_keys(self.mode):
            raise ValueError(f"{self.mode} requires exactly {sorted(input_keys(self.mode))}")
        counts = inputs["event_counts"]
        if counts.ndim != 1 or counts.numel() == 0 or counts.dtype != torch.int64:
            raise ValueError("event_counts must be nonempty int64 [B]")
        batch = counts.shape[0]
        if self.mode != "hierarchy_only":
            for key in VIEW_KEYS:
                shape = (batch, 2, 8, self.height, self.width) if key == "voxel_grid" else (
                    batch, 2, self.height, self.width)
                if inputs[key].shape != shape or not inputs[key].is_floating_point():
                    raise ValueError(f"{key} must be floating point {shape}")
        if self.mode != "latent_only":
            for key in ("voxel_lower", "voxel_upper"):
                if inputs[key].dtype != torch.int64:
                    raise ValueError(f"{key} must be int64")
            if not inputs["points"].is_floating_point() or not inputs["alpha"].is_floating_point():
                raise ValueError("points and alpha must be floating point")

    def branch_embeddings(self, inputs):
        self._validate(inputs)
        branches = {}
        if self.mode != "latent_only":
            features = self.hierarchy.forward_features({k: inputs[k] for k in POINT_KEYS})
            branches["hierarchy"] = self.hierarchy_projection(features.mean(dim=(-2, -1)))
        if self.mode != "hierarchy_only":
            branches["latent"] = self.latent_projection(self.latent(inputs))
        return branches

    def forward_embedding(self, inputs):
        branches = self.branch_embeddings(inputs)
        if self.mode == "hierarchy_only":
            return branches["hierarchy"]
        if self.mode == "latent_only":
            return branches["latent"]
        weight = self.a.sigmoid()
        return (1 - weight) * branches["hierarchy"] + weight * branches["latent"]

    def forward(self, inputs):
        embedding = self.forward_embedding(inputs)
        return self.classifier(embedding) if hasattr(self, "classifier") else embedding


def example_inputs(mode="dual", event_count=116790, *, height=480, width=640, device="meta"):
    input_keys(mode)
    inputs = point_example(event_count, height=height, width=width, device=device)
    if mode == "latent_only":
        inputs = {"event_counts": inputs["event_counts"]}
    if mode != "hierarchy_only":
        inputs.update(event_frame=torch.zeros(1, 2, height, width, device=device),
                      voxel_grid=torch.zeros(1, 2, 8, height, width, device=device),
                      time_surface=torch.zeros(1, 2, height, width, device=device))
    return inputs


def profile_macs(model, event_count=116790):
    """Execute actual meta forward and count Conv/Linear multiply-accumulates."""
    probe = copy.deepcopy(model).to("meta").eval()
    macs, handles = {}, []

    def count(name):
        def hook(module, args, output):
            factor = ((module.in_channels // module.groups) * math.prod(module.kernel_size)
                      if isinstance(module, (nn.Conv2d, nn.Conv3d)) else module.in_features)
            stage = ".".join(name.split(".")[:2]) if name.startswith(("hierarchy.", "latent.")) else name.split(".")[0]
            macs[stage] = macs.get(stage, 0) + output.numel() * factor
        return hook

    for name, module in probe.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            handles.append(module.register_forward_hook(count(name)))
    try:
        with torch.no_grad():
            probe(example_inputs(model.mode, event_count, height=model.height, width=model.width))
    finally:
        for handle in handles:
            handle.remove()
    total = sum(macs.values())
    point_rate = 1152 if model.mode != "latent_only" else 0
    return {"model_version": MODEL_VERSION, "mode": model.mode,
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameters_by_stage": {**{name: sum(p.numel() for p in module.parameters())
                                       for name, module in model.named_children()},
                                    **({"a": 1} if model.mode == "dual" else {})},
            "macs": total, "gmacs": total / 1e9, "macs_by_stage": macs,
            "fixed_macs": total - event_count * point_rate, "point_macs_per_event": point_rate,
            "event_count": event_count, "input_hw": [model.height, model.width],
            "mac_scope": "Conv2d, Conv3d, Linear only; batch=1; excludes scatter/interpolation, normalization, activation, pooling, scalar fusion, preprocessing and I/O"}


def export_backbone(model, path):
    """Portable CPU weights with strict construction/input metadata and no CE head."""
    config = {**model.construction_config(), "include_classifier": False}
    payload = {"export_version": EXPORT_VERSION, "model_version": MODEL_VERSION,
               "input_contract_sha256": INPUT_CONTRACT_SHA256, "model_config": config,
               "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                              if not key.startswith("classifier.")}}
    path = Path(path)
    # Exclusive creation protects previously exported artifacts.
    with path.open("xb") as handle:
        torch.save(payload, handle)
    return path


def load_backbone_export(path, *, map_location="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (set(payload) != {"export_version", "model_version", "input_contract_sha256", "model_config", "state_dict"}
            or payload["export_version"] != EXPORT_VERSION or payload["model_version"] != MODEL_VERSION
            or payload["input_contract_sha256"] != INPUT_CONTRACT_SHA256):
        raise ValueError("backbone export version or input contract mismatch")
    config = payload["model_config"]
    if set(config) != {"mode", "num_classes", "height", "width", "dimension", "seed", "latent_width", "include_classifier"} or config["include_classifier"] is not False:
        raise ValueError("invalid classifier-free construction metadata")
    model = DualFusionBackbone(**config)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(map_location).eval()
