"""Fixed-prompt hierarchy/CLIP comparisons; no training or download side effects.

The production loader accepts only checksum-verified OpenAI ViT checkpoints.
Small injected models used by unit tests are architectural fixtures, not
pretrained CLIP results. The RGB patch convolution is intentionally bypassed
only after loading the complete original checkpoint strictly.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
from string import Formatter
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from .hierarchy_clip_temporal import TemporalAggregator

MINI_PROMPT_TEMPLATE = "a photo of a {class_name}."
# Public OpenAI CLIP download identifiers (also the full-file SHA256).
# https://github.com/openai/CLIP/blob/main/clip/clip.py
OPENAI_CLIP_SHA256 = {
    "ViT-B/16": "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
    "ViT-B/32": "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
    "ViT-L/14": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
    "ViT-L/14@336px": "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_prompts(class_names, template=MINI_PROMPT_TEMPLATE):
    """Preserve caller-provided label order and reject ambiguous templates."""
    names = tuple(class_names)
    if len(names) < 2 or any(not isinstance(n, str) or not n.strip() or n != n.strip() for n in names):
        raise ValueError("require >=2 nonempty, trimmed human-readable class names in label order")
    if len(set(names)) != len(names):
        raise ValueError("class names must be unique")
    fields = [(name, spec, conversion) for _, name, spec, conversion in Formatter().parse(template)
              if name is not None]
    if fields != [("class_name", "", None)]:
        raise ValueError("fixed template must contain exactly one plain {class_name} field")
    return tuple(template.format(class_name=name) for name in names)


def _source_module(path, suffix):
    spec = importlib.util.spec_from_file_location(f"_hierarchy_openai_clip_{suffix}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load CLIP source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_clip_source(source_dir):
    """Lazy source import avoids torchvision/image preprocessing and CUDA probes."""
    if source_dir is None:
        spec = importlib.util.find_spec("clip")
        if spec is None or spec.origin is None:
            raise ImportError("OpenAI CLIP is optional; supply source_dir containing model.py and simple_tokenizer.py")
        source_dir = Path(spec.origin).parent
    source_dir = Path(source_dir).resolve()
    model_path = source_dir / "model.py"
    tokenizer_path = source_dir / "simple_tokenizer.py"
    bpe_path = source_dir / "bpe_simple_vocab_16e6.txt.gz"
    for path in (model_path, tokenizer_path, bpe_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    architecture = _source_module(model_path, "model")
    try:
        tokenizer = _source_module(tokenizer_path, "tokenizer").SimpleTokenizer(str(bpe_path))
    except ModuleNotFoundError as error:
        raise ImportError("OpenAI CLIP text tokenization requires optional ftfy and regex dependencies") from error

    def tokenize(texts, context_length=77):
        result = torch.zeros(len(texts), context_length, dtype=torch.long)
        for row, text in enumerate(texts):
            ids = [tokenizer.encoder["<|startoftext|>"]] + tokenizer.encode(text) + [tokenizer.encoder["<|endoftext|>"]]
            if len(ids) > context_length:
                raise ValueError(f"fixed prompt exceeds CLIP context length {context_length}: {text!r}")
            result[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        return result

    source = {"source_dir": str(source_dir), "model_source_sha256": _sha256(model_path),
              "tokenizer_source_sha256": _sha256(tokenizer_path), "bpe_sha256": _sha256(bpe_path)}
    return architecture.build_model, tokenize, source


@dataclass
class PretrainedCLIP:
    model: nn.Module
    tokenize: Callable
    provenance: dict


def load_pretrained_clip(checkpoint_path, *, source_dir=None, model_name="ViT-B/16"):
    """Load an explicit local official checkpoint strictly on CPU, without fallback.

    Raw state dictionaries from unrelated or randomly initialized models do not
    match the official file digest and are refused before deserialization.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    if model_name not in OPENAI_CLIP_SHA256:
        raise ValueError(f"unsupported OpenAI ViT model: {model_name}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual = _sha256(checkpoint_path)
    if actual != OPENAI_CLIP_SHA256[model_name]:
        raise ValueError(f"OpenAI CLIP checkpoint SHA256 mismatch for {model_name}: {actual}")
    build_model, tokenize, source = _load_clip_source(source_dir)
    try:
        archive = torch.jit.load(str(checkpoint_path), map_location="cpu")
        state = archive.state_dict()
        del archive
    except RuntimeError:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "visual.proj" not in state:
        raise ValueError("checkpoint must contain a full OpenAI CLIP ViT state dictionary")
    state = {key: value for key, value in state.items()
             if key not in {"input_resolution", "context_length", "vocab_size"}}
    model = build_model(dict(state))
    # Enforce strictness even if a supplied source implementation relaxes it.
    model.load_state_dict(state, strict=True)
    model.float().eval().requires_grad_(False)
    _visual_contract(model.visual)
    return PretrainedCLIP(model, tokenize, {
        "model_name": model_name, "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": actual, "official_openai_sha256_verified": True,
        "strict_load": True, "load_device": "cpu", **source,
    })


def _visual_contract(visual):
    required = ("conv1", "class_embedding", "positional_embedding", "ln_pre",
                "transformer", "ln_post", "proj")
    if any(not hasattr(visual, key) for key in required):
        raise ValueError("visual must implement the OpenAI CLIP ViT token contract")
    position = visual.positional_embedding
    if position.ndim != 2:
        raise ValueError("CLIP positional embedding must be [1+grid*grid,width]")
    token_count, width = position.shape[0] - 1, position.shape[1]
    grid = math.isqrt(token_count)
    if grid < 1 or grid * grid != token_count or visual.class_embedding.shape != (width,):
        raise ValueError("CLIP positional/CLS shape mismatch or nonsquare patch grid")
    if visual.proj is None or visual.proj.ndim != 2 or visual.proj.shape[0] != width:
        raise ValueError("CLIP ViT must have a pretrained width-to-embedding projection")
    return grid, width, visual.proj.shape[1]


class FixedPromptTextBank(nn.Module):
    """Frozen normalized class text vectors; no learnable prompt or text module."""

    def __init__(self, pretrained, class_names, prompt_template=MINI_PROMPT_TEMPLATE):
        super().__init__()
        self.class_names = tuple(class_names)
        self.prompt_template = prompt_template
        self.prompts = fixed_prompts(self.class_names, prompt_template)
        self.provenance = dict(pretrained.provenance)
        model = pretrained.model.eval().requires_grad_(False)
        tokens = pretrained.tokenize(self.prompts, context_length=model.context_length)
        device = next(model.parameters()).device
        with torch.no_grad():
            embeddings = model.encode_text(tokens.to(device)).float()
        if embeddings.ndim != 2 or embeddings.shape[0] != len(self.class_names):
            raise ValueError("CLIP text encoder must return one vector per class")
        if not torch.isfinite(embeddings).all() or torch.any(embeddings.norm(dim=-1) == 0):
            raise ValueError("CLIP text features must be finite and nonzero")
        self.register_buffer("embeddings", F.normalize(embeddings, dim=-1))
        self.register_buffer("logit_scale", model.logit_scale.detach().float().exp().clamp(max=100))

    @property
    def embedding_dim(self):
        return self.embeddings.shape[1]

    def forward(self, event_embeddings):
        if event_embeddings.ndim != 2 or event_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"event embeddings must be [B,{self.embedding_dim}]")
        return self.logit_scale * F.normalize(event_embeddings.float(), dim=-1) @ self.embeddings.float().t()

    def contract(self):
        return {"class_names": list(self.class_names), "prompt_template": self.prompt_template,
                "prompts": list(self.prompts), "text_encoder_frozen": True,
                "text_features_cached": True, "logit_scale_trainable": False,
                "embedding_dim": self.embedding_dim, "clip": self.provenance}


class SpatialTokenAdapter(nn.Module):
    """Small learned 256->CLIP-width adapter on the pretrained positional grid."""

    def __init__(self, grid_size, width, input_channels=256):
        super().__init__()
        if min(grid_size, width, input_channels) < 1:
            raise ValueError("adapter dimensions must be positive")
        self.grid_size, self.input_channels = grid_size, input_channels
        self.projection = nn.Conv2d(input_channels, width, 1)
        self.norm = nn.LayerNorm(width)

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.input_channels or min(features.shape[2:]) < 1:
            raise ValueError(f"spatial features must be [B*K,{self.input_channels},H,W]")
        x = F.adaptive_avg_pool2d(features, (self.grid_size, self.grid_size))
        return self.norm(self.projection(x).flatten(2).transpose(1, 2))


def clip_visual_from_tokens(visual, patch_tokens):
    """Reuse the exact pretrained CLS/positions/ViT path after RGB patchification.

    Do not wrap this in no_grad: even a frozen visual tower must propagate
    gradients back into the event adapter and (in later stages) hierarchy.
    """
    grid, width, _ = _visual_contract(visual)
    if patch_tokens.ndim != 3 or patch_tokens.shape[1:] != (grid * grid, width):
        raise ValueError(f"CLIP patch tokens must be [B*K,{grid * grid},{width}]")
    x = patch_tokens.to(dtype=visual.class_embedding.dtype)
    cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
    x = torch.cat((cls, x), dim=1) + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)
    x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    return visual.ln_post(x[:, 0]) @ visual.proj


class HierarchyTextAlignment(nn.Module):
    """Shared hierarchy -> pooled windows -> temporal -> fixed class text CE."""

    name = "hierarchy_text_alignment"

    def __init__(self, encoder, pretrained, class_names, *, prompt_template=MINI_PROMPT_TEMPLATE, temporal=None):
        super().__init__()
        self.encoder = encoder
        self.text_bank = FixedPromptTextBank(pretrained, class_names, prompt_template)
        self.temporal = temporal if temporal is not None else TemporalAggregator(256, num_windows=encoder.num_windows)
        self.projection = nn.Linear(256, self.text_bank.embedding_dim)
        self.num_classes = len(self.text_bank.class_names)

    def forward_embeddings(self, inputs):
        maps = self.encoder(inputs)
        windows = maps.mean(dim=(-2, -1))
        return F.normalize(self.projection(self.temporal(windows, inputs["window_mask"])).float(), dim=-1)

    def forward(self, inputs):
        return self.text_bank(self.forward_embeddings(inputs))


class HierarchyCLIPViTAlignment(nn.Module):
    """Shared hierarchy -> spatial adapter -> pretrained ViT -> temporal -> text."""

    name = "hierarchy_clip_vit_alignment"

    def __init__(self, encoder, pretrained, class_names, *, prompt_template=MINI_PROMPT_TEMPLATE, temporal=None):
        super().__init__()
        self.encoder = encoder
        self.text_bank = FixedPromptTextBank(pretrained, class_names, prompt_template)
        # Each comparison owns its fine-tunable tower. Sharing the source tower
        # would leak updates between experiments and let another text-bank
        # construction silently freeze this model's selective tuning stage.
        self.visual = copy.deepcopy(pretrained.model.visual)
        grid, width, output_dim = _visual_contract(self.visual)
        if output_dim != self.text_bank.embedding_dim:
            raise ValueError("CLIP visual and text embedding dimensions must match")
        self.adapter = SpatialTokenAdapter(grid, width)
        self.temporal = temporal if temporal is not None else TemporalAggregator(output_dim, num_windows=encoder.num_windows)
        self.projection = nn.Identity()  # The pretrained visual.proj supplies CLIP's projection.
        self.num_classes = len(self.text_bank.class_names)

    def forward_embeddings(self, inputs):
        maps = self.encoder(inputs)
        batch, windows = maps.shape[:2]
        tokens = self.adapter(maps.flatten(0, 1))
        encoded = clip_visual_from_tokens(self.visual, tokens).reshape(batch, windows, -1)
        return F.normalize(self.projection(self.temporal(encoded, inputs["window_mask"])).float(), dim=-1)

    def forward(self, inputs):
        return self.text_bank(self.forward_embeddings(inputs))
