"""V1 heterogeneous encoder and its three controlled baselines. No pretraining."""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn

MODEL_NAMES = ("frame", "same_arch", "heterogeneous", "frame_matched")
MODEL_VERSION = "v1-specialized-concat-1"


def norm(channels: int) -> nn.Module:
    return nn.GroupNorm(math.gcd(8, channels), channels)


def conv(cin: int, cout: int, kernel=3, stride=1, *, temporal=False, groups=1):
    cls = nn.Conv3d if temporal else nn.Conv2d
    padding = tuple(k // 2 for k in kernel) if isinstance(kernel, tuple) else kernel // 2
    return cls(cin, cout, kernel, stride, padding, groups=groups, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(conv(cin, cout, stride=stride), norm(cout), nn.SiLU(),
                                  conv(cout, cout), norm(cout))
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            conv(cin, cout, 1, stride), norm(cout))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class SurfaceBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(conv(cin, cin, 5, stride, groups=cin), norm(cin), nn.SiLU(),
                                  conv(cin, cout, 1), norm(cout))
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            conv(cin, cout, 1, stride), norm(cout))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class TemporalBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        layers = []
        for pair, (input_width, spatial_stride) in enumerate(((cin, stride), (cout, 1))):
            layers += [conv(input_width, cout, (1, 3, 3), (1, spatial_stride, spatial_stride),
                            temporal=True), norm(cout), nn.SiLU(),
                       conv(cout, cout, (3, 1, 1), temporal=True), norm(cout)]
            if pair == 0:
                layers.append(nn.SiLU())
        self.body = nn.Sequential(*layers)
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            conv(cin, cout, (1, 1, 1), (1, stride, stride), temporal=True), norm(cout))
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class Branch(nn.Module):
    def __init__(self, kind: str, cin: int = 2, width: int = 32):
        super().__init__()
        temporal = kind == "voxel"
        block = {"frame": BasicBlock, "surface": SurfaceBlock, "voxel": TemporalBlock}[kind]
        stem = conv(cin, width, (1, 3, 3), (1, 2, 2), temporal=True) if temporal else conv(cin, width, stride=2)
        layers = [stem, norm(width), nn.SiLU()]
        previous = width
        for i, channels in enumerate((width, 2 * width, 4 * width)):
            layers += [block(previous, channels, 1 if i == 0 else 2), block(channels, channels)]
            previous = channels
        self.body = nn.Sequential(*layers)
        self.collapse = nn.Sequential(conv(8 * 4 * width, 4 * width, 1), norm(4 * width), nn.SiLU()) if temporal else nn.Identity()
        self.temporal = temporal

    def forward(self, x):
        x = self.body(x)
        if self.temporal:
            x = x.flatten(1, 2)  # C-major, then ordered temporal bins; never average time.
        return self.collapse(x)


class V1Backbone(nn.Module):
    def __init__(self, name: str, num_classes: int = 100, frame_width: int = 32):
        super().__init__()
        if name not in MODEL_NAMES or num_classes < 2 or frame_width < 8 or frame_width % 8:
            raise ValueError("invalid V1 architecture, class count, or frame width")
        if name != "frame_matched" and frame_width != 32:
            raise ValueError("only frame_matched may change branch width")
        self.name = name
        self.frame_width = frame_width
        self.multiview = name in {"same_arch", "heterogeneous"}
        self.frame = Branch("frame", width=frame_width)
        if self.multiview:
            self.voxel = Branch("voxel") if name == "heterogeneous" else Branch("frame", cin=16)
            self.surface = Branch("surface" if name == "heterogeneous" else "frame")
        incoming = 384 if self.multiview else 4 * frame_width
        self.fusion = nn.Sequential(conv(incoming, 256, 1), norm(256), nn.SiLU())
        self.trunk = nn.Sequential(BasicBlock(256, 256), BasicBlock(256, 256))
        self.classifier = nn.Linear(256, num_classes)

    def forward_features(self, inputs: dict[str, Tensor]) -> Tensor:
        expected = {"event_frame", "voxel_grid", "time_surface"} if self.multiview else {"event_frame"}
        if set(inputs) != expected:
            raise ValueError(f"{self.name} requires exactly {sorted(expected)}")
        frame = inputs["event_frame"]
        if frame.ndim != 4 or frame.shape[1] != 2:
            raise ValueError("event frame must be [B,2,H,W]")
        features = [self.frame(frame)]
        if self.multiview:
            voxel, surface = inputs["voxel_grid"], inputs["time_surface"]
            if voxel.shape != (frame.shape[0], 2, 8, *frame.shape[2:]) or surface.shape != frame.shape:
                raise ValueError("V1 requires spatially aligned frame, 8-bin voxel and surface")
            if self.name == "same_arch":
                voxel = voxel.flatten(1, 2)
            features += [self.voxel(voxel), self.surface(surface)]
        return self.trunk(self.fusion(torch.cat(features, dim=1)))

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        return self.classifier(self.forward_features(inputs).mean(dim=(-2, -1)))


def example_inputs(name: str, height=480, width=640, *, device="cpu"):
    inputs = {"event_frame": torch.zeros(1, 2, height, width, device=device)}
    if name in {"same_arch", "heterogeneous"}:
        inputs.update(voxel_grid=torch.zeros(1, 2, 8, height, width, device=device),
                      time_surface=torch.zeros(1, 2, height, width, device=device))
    return inputs


def profile_macs(model: V1Backbone, height=480, width=640) -> dict:
    """Count convolution/linear MACs on meta tensors, excluding norms/activations.

    One multiply-accumulate is one MAC, approximately two FLOPs. No timing or
    parameter matching is implied. Meta execution avoids allocating activations.
    """
    probe = copy.deepcopy(model).to("meta").eval()
    macs = 0

    def count(module, inputs, output):
        nonlocal macs
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            macs += output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features

    hooks = [m.register_forward_hook(count) for m in probe.modules()
             if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear))]
    try:
        with torch.no_grad():
            probe(example_inputs(model.name, height, width, device="meta"))
    finally:
        for hook in hooks:
            hook.remove()
    return {"parameters": sum(p.numel() for p in model.parameters()), "macs": macs,
            "input_hw": [height, width], "frame_width": model.frame_width,
            "mac_scope": "Conv2d, Conv3d and Linear only; batch=1; rendering excluded"}


def match_frame_width(num_classes=100, height=480, width=640, tolerance=0.05):
    """Select width by MACs only, before training and without reading labels."""
    with torch.random.fork_rng(devices=[]):
        target = profile_macs(V1Backbone("heterogeneous", num_classes), height, width)["macs"]
        candidates = []
        for width_base in range(32, 161, 8):
            stats = profile_macs(V1Backbone("frame_matched", num_classes, width_base), height, width)
            candidates.append((abs(stats["macs"] / target - 1), width_base, stats))
    error, width_base, stats = min(candidates, key=lambda entry: entry[0])
    if error > tolerance:
        raise ValueError(f"no frame width matches target MACs within {tolerance:.1%}: {error:.1%}")
    return {**stats, "target_macs": target, "relative_mac_error": error, "tolerance": tolerance}
