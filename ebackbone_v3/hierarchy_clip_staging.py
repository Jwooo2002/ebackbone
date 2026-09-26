"""Explicit fine-tuning stages and strict transfer/checkpoint contracts.

These helpers do not start a training schedule. Rebuild the optimizer when
changing stage so newly unfrozen parameters are actually optimized.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from .hierarchy_data import POINT_CONTRACT_SHA256


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hierarchy_backbone(hierarchy, path):
    """Load a trusted local hierarchy checkpoint/export; drop only its CE head."""
    path = Path(path).resolve()
    state = torch.load(path, map_location="cpu", weights_only=False)
    identity = state.get("identity", {})
    variant = state.get("variant", identity.get("architecture", identity.get("variant")))
    if variant not in (None, "hierarchy"):
        raise ValueError(f"expected hierarchy-only pretrained backbone, got {variant}")
    contract = identity.get("point_contract_sha256")
    if contract is not None and contract != POINT_CONTRACT_SHA256:
        raise ValueError("hierarchy point contract mismatch")
    source = state.get("backbone", state.get("model"))
    if not isinstance(source, dict) or not source:
        raise ValueError("requires a hierarchy model checkpoint or backbone export")
    prefixed = [k.startswith("module.") for k in source]
    if any(prefixed) and not all(prefixed):
        raise ValueError("mixed DDP prefixes")
    source = {k.removeprefix("module."): v for k, v in source.items()}
    source = {k: v for k, v in source.items() if not k.startswith("classifier.")}
    expected = {k: v for k, v in hierarchy.state_dict().items() if not k.startswith("classifier.")}
    if set(source) != set(expected):
        raise ValueError(f"backbone keys mismatch: missing={sorted(set(expected)-set(source))}, "
                         f"unexpected={sorted(set(source)-set(expected))}")
    for key, tensor in source.items():
        if tensor.shape != expected[key].shape or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid backbone tensor: {key}")
    # A complete state is assembled only after validating every backbone key.
    hierarchy.load_state_dict({**hierarchy.state_dict(), **source}, strict=True)
    return {"path": str(path), "sha256": file_sha256(path), "epoch": state.get("epoch"),
            "variant": "hierarchy", "backbone_keys": len(source), "strict_backbone": True,
            "classifier_transferred": False}


def configure_stage(model, stage, *, learning_rate=1e-3, backbone_lr_ratio=0.1,
                    visual_blocks=2):
    """Return disjoint optimizer groups for adapters-only or selective tuning."""
    if stage not in ("connectors", "selective"):
        raise ValueError("stage must be connectors or selective")
    if learning_rate <= 0 or not 0 < backbone_lr_ratio <= 1 or visual_blocks < 0:
        raise ValueError("invalid learning rate or visual block count")
    model.requires_grad_(False)
    # Discard stale gradients from a previous stage.
    model.zero_grad(set_to_none=True)
    for name in ("adapter", "temporal", "projection", "classifier"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            module.requires_grad_(True)
    lower_rate_ids = set()
    if stage == "selective":
        hierarchy = model.encoder.hierarchy
        for name in ("temporal_collapse", "frame_stage"):
            module = getattr(hierarchy, name)
            module.requires_grad_(True)
            lower_rate_ids.update(id(p) for p in module.parameters())
        visual = getattr(model, "visual", None)
        if visual is not None and visual_blocks:
            blocks = visual.transformer.resblocks
            if visual_blocks > len(blocks):
                raise ValueError("requested more visual blocks than the pretrained ViT has")
            for block in list(blocks)[-visual_blocks:]:
                block.requires_grad_(True)
                lower_rate_ids.update(id(p) for p in block.parameters())
    groups = []
    for low, name, lr in ((False, "connectors", learning_rate),
                           (True, "pretrained_upper", learning_rate * backbone_lr_ratio)):
        params = [p for p in model.parameters() if p.requires_grad and (id(p) in lower_rate_ids) == low]
        if params:
            groups.append({"params": params, "lr": lr, "name": name})
    if not groups:
        raise ValueError("stage has no trainable parameters")
    model._fine_tuning_stage = stage
    set_staged_train(model)
    return groups


def set_staged_train(model):
    """Keep frozen modules in eval while retaining autograd through frozen ViT."""
    model.train()
    for module in model.modules():
        parameters = list(module.parameters())
        if parameters and not any(p.requires_grad for p in parameters):
            module.eval()
    return model


def gradient_report(model):
    result = {}
    for name, module in model.named_children():
        parameters = list(module.parameters())
        grads = [p.grad.detach() for p in parameters if p.grad is not None]
        if any(not torch.isfinite(g).all() for g in grads):
            raise FloatingPointError(f"nonfinite gradient in {name}")
        if any(p.grad is not None for p in parameters if not p.requires_grad):
            raise ValueError(f"frozen parameter acquired gradient in {name}")
        result[name] = {"trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad),
                        "gradient_l1": sum(float(g.abs().sum()) for g in grads)}
    return result


def save_comparison_checkpoint(path, model, optimizer, identity):
    """Save bounded-run state with its immutable data/prompt/model identity."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    torch.save({"schema": "hierarchy-clip-comparison-1", "identity": identity,
                "stage": model._fine_tuning_stage,
                "trainable_names": [n for n, p in model.named_parameters() if p.requires_grad],
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict()}, path)


def load_comparison_checkpoint(path, model, optimizer, identity):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != "hierarchy-clip-comparison-1" or state.get("identity") != identity:
        raise ValueError("comparison checkpoint identity mismatch")
    if state["stage"] != getattr(model, "_fine_tuning_stage", None):
        raise ValueError("configure the checkpoint stage before loading")
    if state["trainable_names"] != [n for n, p in model.named_parameters() if p.requires_grad]:
        raise ValueError("checkpoint trainable parameter policy mismatch")
    model.load_state_dict(state["model"], strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value.detach().cpu(), state["model"][name]):
            raise ValueError(f"loaded comparison tensor differs: {name}")
    optimizer.load_state_dict(state["optimizer"])
    return state
