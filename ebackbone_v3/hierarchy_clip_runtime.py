"""Memory-controlled production execution of the verified event/CLIP models.

Chunking changes execution batch sizes, never events, windows, parameters, or
optimizer batches. Original hierarchy and CLIP parameter/state names are kept
so strict transfer and staged optimizer groups retain their existing meaning.
This module builds models on CPU and never initializes CUDA or launches jobs.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .hierarchy_models import HierarchyV1, INPUT_KEYS
from .hierarchy_clip_models import (HierarchyTextAlignment, HierarchyCLIPViTAlignment,
    MINI_PROMPT_TEMPLATE, fixed_prompts, load_pretrained_clip, _visual_contract)
from .hierarchy_clip_temporal import HierarchySequenceEncoder, HierarchyTemporalBaseline
from .hierarchy_clip_staging import file_sha256, load_hierarchy_backbone

VARIANTS = ("hierarchy_temporal", "hierarchy_text", "hierarchy_clip_vit")


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _average_matrix(input_size, output_size):
    """Exact adaptive-average bin membership; coefficients are fixed constants."""
    result = torch.zeros(output_size, input_size, dtype=torch.float64)
    for index in range(output_size):
        start = index * input_size // output_size
        end = ((index + 1) * input_size + output_size - 1) // output_size
        result[index, start:end] = 1.0 / (end - start)
    return result


class DeterministicAdaptiveAverage2d(nn.Module):
    """Adaptive-average math using separable fixed-bin matrices.

    CUDA adaptive_avg_pool2d backward is not deterministic for overlapping bins.
    These matmuls implement the same rectangular averages and gradients without
    that kernel. FP32 accumulation is kept under BF16 autocast; outputs retain
    the input dtype. CUDA callers must configure deterministic CuBLAS before
    initializing CUDA (CUBLAS_WORKSPACE_CONFIG=:4096:8).
    """

    def __init__(self, input_hw, output_hw):
        super().__init__()
        self.input_hw = tuple(_positive_integer(int(v), "input size") for v in input_hw)
        self.output_hw = tuple(_positive_integer(int(v), "output size") for v in output_hw)
        if len(self.input_hw) != 2 or len(self.output_hw) != 2:
            raise ValueError("adaptive-average dimensions must be pairs")
        # Constants can be rebuilt; omit them from the original model state.
        self.register_buffer("height_average", _average_matrix(self.input_hw[0], self.output_hw[0]), persistent=False)
        self.register_buffer("width_average", _average_matrix(self.input_hw[1], self.output_hw[1]), persistent=False)

    def forward(self, features):
        if features.ndim != 4 or tuple(features.shape[-2:]) != self.input_hw or not features.is_floating_point():
            raise ValueError(f"pooling input must be floating [N,C,{self.input_hw[0]},{self.input_hw[1]}]")
        dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=features.device.type, enabled=False):
            values = features.to(dtype=dtype)
            rows = self.height_average.to(device=features.device, dtype=dtype)
            columns = self.width_average.to(device=features.device, dtype=dtype)
            output = torch.matmul(torch.matmul(rows, values), columns.t())
        return output.to(dtype=features.dtype)


class DeterministicSpatialTokenAdapter(nn.Module):
    """Keep verified adapter weights/names and replace only its pooling kernel."""

    def __init__(self, adapter, input_hw):
        super().__init__()
        self.grid_size, self.input_channels = adapter.grid_size, adapter.input_channels
        self.projection, self.norm = adapter.projection, adapter.norm
        self.pool = DeterministicAdaptiveAverage2d(input_hw, (self.grid_size, self.grid_size))

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.input_channels:
            raise ValueError(f"spatial features must be [B*K,{self.input_channels},H,W]")
        return self.norm(self.projection(self.pool(features)).flatten(2).transpose(1, 2))


class RuntimeHierarchySequenceEncoder(HierarchySequenceEncoder):
    """Same point -> voxel -> frame hierarchy, evaluated in packed window chunks."""

    def __init__(self, hierarchy, num_windows=4, *, hierarchy_chunk_windows=4, activation_checkpointing=True):
        super().__init__(hierarchy, num_windows)
        self.hierarchy_chunk_windows = _positive_integer(hierarchy_chunk_windows, "hierarchy_chunk_windows")
        if not isinstance(activation_checkpointing, bool):
            raise ValueError("activation_checkpointing must be boolean")
        self.activation_checkpointing = activation_checkpointing

    def _validate(self, inputs):
        # The same validated packed contract as HierarchySequenceEncoder, applied
        # once to the full optimizer microbatch before extracting any chunks.
        if set(inputs) != {"packed", "window_mask"}:
            raise ValueError("sequence inputs require exactly packed and window_mask")
        packed, mask = inputs["packed"], inputs["window_mask"]
        if mask.ndim != 2 or mask.dtype != torch.bool or mask.shape[1] != self.num_windows or mask.shape[0] == 0:
            raise ValueError("window_mask must be boolean [B,K] matching num_windows")
        if set(packed) != INPUT_KEYS or any(value.device != mask.device for value in packed.values()):
            raise ValueError("all exact hierarchy packed tensors must share the window mask device")
        counts, points, alpha = packed["event_counts"], packed["points"], packed["alpha"]
        if counts.dtype != torch.long or counts.shape != (mask.numel(),) or bool((counts < 0).any()):
            raise ValueError("event_counts must be nonnegative int64 [B*K]")
        if not torch.equal(counts > 0, mask.flatten()):
            raise ValueError("window mask must agree with packed event counts")
        if points.ndim != 2 or points.shape[1] != 4 or points.dtype != torch.float32 or not bool(torch.isfinite(points).all()):
            raise ValueError("packed points must be finite float32 [N,4]")
        count_values = counts.tolist()
        total = sum(count_values)
        if total != len(points):
            raise ValueError("event counts must conserve all packed points")
        if alpha.shape != (total,) or alpha.dtype != torch.float32 or not bool(torch.isfinite(alpha).all()) or bool(((alpha < 0) | (alpha > 1)).any()):
            raise ValueError("alpha must be finite float32 [N] interpolation weights in [0,1]")
        cells = 8 * (self.hierarchy.height // 4) * (self.hierarchy.width // 4)
        owners = torch.repeat_interleave(torch.arange(mask.numel(), device=mask.device), counts)
        for key in ("voxel_lower", "voxel_upper"):
            route = packed[key]
            if route.dtype != torch.long or route.shape != (total,):
                raise ValueError(f"{key} must be int64 [N]")
            if not torch.equal(torch.div(route, cells, rounding_mode="floor"), owners):
                raise ValueError(f"{key} routes must stay inside their own sample/window")
        return count_values, cells

    def _prefix(self, packed):
        hierarchy = self.hierarchy
        values = hierarchy.point_to_voxel(hierarchy.point(packed["points"]), packed)
        values = hierarchy.voxel_stage2(hierarchy.voxel_stage1(hierarchy.voxel_projection(values)))
        return values.flatten(1, 2)

    def _upper(self, values):
        return self.hierarchy.frame_stage(self.hierarchy.temporal_collapse(values))

    def _features(self, packed):
        prefix_modules = (self.hierarchy.point, self.hierarchy.voxel_projection,
                          self.hierarchy.voxel_stage1, self.hierarchy.voxel_stage2)
        prefix_grad = packed["points"].requires_grad or any(p.requires_grad for m in prefix_modules for p in m.parameters())
        use_checkpoint = self.activation_checkpointing and torch.is_grad_enabled()
        if use_checkpoint and prefix_grad:
            values = checkpoint(self._prefix, packed, use_reentrant=False)
        else:
            values = self._prefix(packed)
        upper_grad = values.requires_grad or any(p.requires_grad for m in (
            self.hierarchy.temporal_collapse, self.hierarchy.frame_stage) for p in m.parameters())
        if use_checkpoint and upper_grad:
            return checkpoint(self._upper, values, use_reentrant=False)
        return self._upper(values)

    def forward(self, inputs):
        counts, cells = self._validate(inputs)
        packed, mask = inputs["packed"], inputs["window_mask"]
        output_hw = (self.hierarchy.height // 32, self.hierarchy.width // 32)
        chunks, event_start = [], 0
        for start in range(0, len(counts), self.hierarchy_chunk_windows):
            end = min(start + self.hierarchy_chunk_windows, len(counts))
            event_end = event_start + sum(counts[start:end])
            if event_start == event_end:
                output = packed["points"].new_zeros(end - start, 256, *output_hw)
            else:
                local = {key: value[event_start:event_end] for key, value in packed.items() if key != "event_counts"}
                local["event_counts"] = packed["event_counts"][start:end]
                for key in ("voxel_lower", "voxel_upper"):
                    local[key] = local[key] - start * cells
                output = self._features(local)
                if tuple(output.shape) != (end - start, 256, *output_hw):
                    raise ValueError("hierarchy chunk map contract mismatch")
            chunks.append(output)
            event_start = event_end
        # Empty chunks must not promote BF16 outputs back to FP32 under autocast.
        dtype = next((chunk.dtype for chunk in chunks if chunk.requires_grad), chunks[0].dtype)
        if torch.is_autocast_enabled(packed["points"].device.type):
            dtype = torch.get_autocast_dtype(packed["points"].device.type)
        maps = torch.cat([chunk.to(dtype=dtype) for chunk in chunks], dim=0)
        maps = maps.reshape(mask.shape[0], self.num_windows, 256, *output_hw)
        return maps.masked_fill(~mask[:, :, None, None, None], 0)


class RuntimeHierarchyCLIPViTAlignment(HierarchyCLIPViTAlignment):
    """Verified spatial-token architecture with bounded visual chunk/checkpoints."""

    def __init__(self, encoder, pretrained, class_names, *, prompt_template=MINI_PROMPT_TEMPLATE,
                 temporal=None, visual_chunk_windows=8, activation_checkpointing=True, spatial_hw=None):
        super().__init__(encoder, pretrained, class_names, prompt_template=prompt_template, temporal=temporal)
        self.visual_chunk_windows = _positive_integer(visual_chunk_windows, "visual_chunk_windows")
        if not isinstance(activation_checkpointing, bool):
            raise ValueError("activation_checkpointing must be boolean")
        self.activation_checkpointing = activation_checkpointing
        if spatial_hw is None:
            spatial_hw = (encoder.hierarchy.height // 32, encoder.hierarchy.width // 32)
        self.adapter = DeterministicSpatialTokenAdapter(self.adapter, spatial_hw)

    def _visual_tokens(self, tokens):
        grid, width, _ = _visual_contract(self.visual)
        if tokens.ndim != 3 or tokens.shape[1:] != (grid * grid, width):
            raise ValueError("visual adapter token contract mismatch")
        values = tokens.to(dtype=self.visual.class_embedding.dtype)
        cls = self.visual.class_embedding.to(values.dtype).expand(values.shape[0], 1, -1)
        values = self.visual.ln_pre(torch.cat((cls, values), dim=1) + self.visual.positional_embedding.to(values.dtype))
        values = values.permute(1, 0, 2)
        for block in self.visual.transformer.resblocks:
            needs_grad = values.requires_grad or any(p.requires_grad for p in block.parameters())
            if self.activation_checkpointing and torch.is_grad_enabled() and needs_grad:
                values = checkpoint(block, values, use_reentrant=False)
            else:
                values = block(values)
        return self.visual.ln_post(values.permute(1, 0, 2)[:, 0]) @ self.visual.proj

    def forward_embeddings(self, inputs):
        maps = self.encoder(inputs)
        batch, windows = maps.shape[:2]
        flat = maps.flatten(0, 1)
        embeddings = []
        for start in range(0, len(flat), self.visual_chunk_windows):
            tokens = self.adapter(flat[start:start + self.visual_chunk_windows])
            embeddings.append(self._visual_tokens(tokens))
        encoded = torch.cat(embeddings).reshape(batch, windows, -1)
        return F.normalize(self.projection(self.temporal(encoded, inputs["window_mask"])).float(), dim=-1)


def _class_contract(config):
    path = Path(config["class_names_path"])
    labels = json.loads(path.read_text())
    manifest = Path(config["manifest_dir"]) / "provenance.json"
    provenance = json.loads(manifest.read_text())
    mapping = provenance["class_to_index"]
    if len(mapping) != 100 or sorted(mapping.values()) != list(range(100)):
        raise ValueError("production Mini runtime requires exactly 100 contiguous class labels")
    class_ids = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    names = labels["class_names"]
    if labels["class_ids"] != class_ids or len(names) != len(class_ids):
        raise ValueError("text class order differs from immutable Mini manifest labels")
    template = config.get("prompt_template", MINI_PROMPT_TEMPLATE)
    prompts = fixed_prompts(names, template)
    return names, {"class_ids": class_ids, "class_names": names, "prompts": list(prompts),
                   "prompt_template": template, "class_names_sha256": file_sha256(path),
                   "manifest_provenance_sha256": file_sha256(manifest)}


def build_model(variant, config):
    """Construct one independent CPU comparison and its transfer/runtime identity."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown hierarchy/CLIP comparison: {variant}")
    names, classes = _class_contract(config)
    execution = config.get("execution", {})
    windows = config.get("num_windows", 4)
    if windows != 4:
        raise ValueError("production Mini comparison protocol requires exactly four windows")
    hierarchy = HierarchyV1(len(names))
    hierarchy_provenance = load_hierarchy_backbone(hierarchy, config["hierarchy_checkpoint"])
    encoder = RuntimeHierarchySequenceEncoder(hierarchy, windows,
        hierarchy_chunk_windows=execution.get("hierarchy_chunk_windows", 4),
        activation_checkpointing=execution.get("activation_checkpointing", True))
    clip_provenance = None
    if variant == "hierarchy_temporal":
        model = HierarchyTemporalBaseline(encoder, num_classes=len(names))
    else:
        # Reconstructing a fully loaded checkpoint must not consume the RNG used
        # to initialize the shared hierarchy-temporal and hierarchy-text heads.
        with torch.random.fork_rng(devices=[]):
            pretrained = load_pretrained_clip(config["clip_checkpoint"],
                source_dir=config.get("clip_source_dir"), model_name=config.get("clip_model_name", "ViT-B/16"))
        clip_provenance = pretrained.provenance
        if variant == "hierarchy_text":
            model = HierarchyTextAlignment(encoder, pretrained, names, prompt_template=classes["prompt_template"])
        else:
            model = RuntimeHierarchyCLIPViTAlignment(encoder, pretrained, names,
                prompt_template=classes["prompt_template"],
                visual_chunk_windows=execution.get("visual_chunk_windows", 8),
                activation_checkpointing=encoder.activation_checkpointing)
        del pretrained
    provenance = {"runtime_version": "hierarchy-clip-chunked-deterministic-1", "variant": variant,
        "hierarchy": hierarchy_provenance, "clip": clip_provenance, "classes": classes,
        "num_windows": windows, "native_resolution": [480, 640], "raw_rgb_used": False,
        "execution": {"hierarchy_chunk_windows": encoder.hierarchy_chunk_windows,
                      "visual_chunk_windows": execution.get("visual_chunk_windows", 8),
                      "activation_checkpointing": encoder.activation_checkpointing,
                      "adapter_pooling": "separable fixed adaptive-average bins, FP32 accumulation"},
        "runtime_source_sha256": file_sha256(__file__)}
    return model, provenance
