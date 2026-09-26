"""Tiny architectural fixtures only: these are not pretrained CLIP results."""

import copy
import hashlib

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ebackbone_v3 import hierarchy_clip_models as models


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(8)
        self.attn = nn.MultiheadAttention(8, 2)
        self.ln_2 = nn.LayerNorm(8)
        self.mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))

    def forward(self, x):
        normalized = self.ln_1(x)
        x = x + self.attn(normalized, normalized, normalized, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblocks = nn.Sequential(TinyBlock(), TinyBlock())

    def forward(self, x):
        return self.resblocks(x)


class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 2, stride=2, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(8))
        self.positional_embedding = nn.Parameter(torch.randn(5, 8))
        self.ln_pre = nn.LayerNorm(8)
        self.transformer = TinyTransformer()
        self.ln_post = nn.LayerNorm(8)
        self.proj = nn.Parameter(torch.randn(8, 8))

    def forward(self, image):
        patches = self.conv1(image).flatten(2).transpose(1, 2)
        cls = self.class_embedding.expand(image.shape[0], 1, -1)
        x = self.ln_pre(torch.cat((cls, patches), 1) + self.positional_embedding)
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        return self.ln_post(x[:, 0]) @ self.proj


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = TinyVisual()
        self.context_length = 6
        self.token_embedding = nn.Embedding(32, 8)
        self.text_projection = nn.Parameter(torch.randn(8, 8))
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def encode_text(self, tokens):
        return self.token_embedding(tokens).mean(1) @ self.text_projection


def fixture_tokenize(texts, context_length=6):
    return torch.stack([torch.full((context_length,), i + 1, dtype=torch.long) for i, _ in enumerate(texts)])


def fixture_pretrained():
    return models.PretrainedCLIP(TinyCLIP(), fixture_tokenize,
                                 {"fixture_only": True, "official_openai_sha256_verified": False})


class TinySequenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_windows = 4
        self.hierarchy = nn.Conv2d(4, 256, 1)

    def forward(self, inputs):
        x = inputs["fixture_maps"]
        b, k = x.shape[:2]
        return self.hierarchy(x.flatten(0, 1)).reshape(b, k, 256, *x.shape[-2:]) * inputs["window_mask"][:, :, None, None, None]


def fixture_inputs():
    return {"fixture_maps": torch.randn(2, 4, 4, 3, 5),
            "window_mask": torch.tensor([[True, True, False, False], [True, True, True, True]])}


def test_fixed_prompt_label_order_and_validation():
    assert models.fixed_prompts(["dog", "cat"]) == ("a photo of a dog.", "a photo of a cat.")
    assert models.fixed_prompts(["walking", "running"], "a person {class_name}.") == (
        "a person walking.", "a person running.")
    for names in (["dog"], ["dog", "dog"], ["", "cat"], [" dog", "cat"]):
        with pytest.raises(ValueError):
            models.fixed_prompts(names)
    for template in ("{label}", "{class_name} {class_name}", "no field", "{class_name!r}"):
        with pytest.raises(ValueError):
            models.fixed_prompts(["dog", "cat"], template)


def test_text_bank_is_fixed_normalized_and_label_ordered():
    pretrained = fixture_pretrained()
    bank = models.FixedPromptTextBank(pretrained, ["dog", "cat"])
    assert not list(bank.parameters())
    assert all(not p.requires_grad for p in pretrained.model.parameters())
    torch.testing.assert_close(bank.embeddings.norm(dim=-1), torch.ones(2))
    event = torch.randn(3, 8, requires_grad=True)
    expected = bank.logit_scale * F.normalize(event, dim=-1) @ bank.embeddings.t()
    torch.testing.assert_close(bank(event), expected)
    bank(event).sum().backward()
    assert event.grad is not None and event.grad.abs().sum() > 0
    assert all(p.grad is None for p in pretrained.model.parameters())
    assert bank.contract()["prompts"] == ["a photo of a dog.", "a photo of a cat."]
    with pytest.raises(ValueError, match="event embeddings"):
        bank(torch.zeros(2, 7))


def test_spatial_adapter_contract_and_normalization():
    adapter = models.SpatialTokenAdapter(2, 8)
    features = torch.randn(3, 256, 3, 5, requires_grad=True)
    tokens = adapter(features)
    assert tokens.shape == (3, 4, 8)
    torch.testing.assert_close(tokens.mean(-1), torch.zeros(3, 4), atol=1e-6, rtol=0)
    tokens.square().sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    with pytest.raises(ValueError, match="spatial features"):
        adapter(torch.randn(2, 255, 3, 5))


def test_visual_path_matches_pretrained_cls_positions_transformer_and_projection():
    visual = TinyVisual().eval()
    image = torch.randn(2, 3, 4, 4)
    patches = visual.conv1(image).flatten(2).transpose(1, 2)
    expected = visual(image)
    seen = []
    handle = visual.ln_pre.register_forward_pre_hook(lambda _module, args: seen.append(args[0].detach()))
    conv_calls = []
    conv_handle = visual.conv1.register_forward_hook(lambda *_args: conv_calls.append(1))
    try:
        actual = models.clip_visual_from_tokens(visual, patches)
    finally:
        handle.remove()
        conv_handle.remove()
    torch.testing.assert_close(actual, expected)
    assert not conv_calls  # bypass RGB patch convolution after adapter tokens
    torch.testing.assert_close(seen[0][:, 0], (visual.class_embedding + visual.positional_embedding[0]).expand(2, -1))
    torch.testing.assert_close(seen[0][:, 1:], patches + visual.positional_embedding[1:])
    with pytest.raises(ValueError, match="CLIP patch tokens"):
        models.clip_visual_from_tokens(visual, torch.randn(2, 5, 8))


@pytest.mark.parametrize("model_type", [models.HierarchyTextAlignment, models.HierarchyCLIPViTAlignment])
def test_comparison_forward_gradients_and_exact_reload(model_type):
    torch.manual_seed(13)
    pretrained = fixture_pretrained()
    model = model_type(TinySequenceEncoder(), pretrained, ["dog", "cat"])
    inputs = fixture_inputs()
    embeddings = model.forward_embeddings(inputs)
    assert embeddings.shape == (2, 8)
    torch.testing.assert_close(embeddings.norm(dim=-1), torch.ones(2))
    logits = model(inputs)
    assert logits.shape == (2, 2) and torch.isfinite(logits).all()
    loss = F.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert model.encoder.hierarchy.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.temporal.parameters())
    assert all(p.grad is None for p in pretrained.model.parameters())
    if model_type is models.HierarchyCLIPViTAlignment:
        assert model.adapter.projection.weight.grad.abs().sum() > 0
        assert all(not p.requires_grad and p.grad is None for p in model.visual.parameters())
    else:
        assert model.projection.weight.grad.abs().sum() > 0
    restored = model_type(TinySequenceEncoder(), fixture_pretrained(), ["dog", "cat"])
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
    state = dict(model.state_dict())
    state.pop(next(iter(state)))
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(state, strict=True)


def test_comparisons_own_visual_weights_and_preserve_other_model_stage():
    pretrained = fixture_pretrained()
    first = models.HierarchyCLIPViTAlignment(TinySequenceEncoder(), pretrained, ["dog", "cat"])
    first.visual.train()
    first.visual.transformer.resblocks[-1].requires_grad_(True)
    expected_trainable = {name for name, parameter in first.visual.named_parameters() if parameter.requires_grad}
    models.HierarchyTextAlignment(TinySequenceEncoder(), pretrained, ["dog", "cat"])
    second = models.HierarchyCLIPViTAlignment(TinySequenceEncoder(), pretrained, ["dog", "cat"])
    assert first.visual is not second.visual and first.visual is not pretrained.model.visual
    assert first.visual.training
    assert {name for name, parameter in first.visual.named_parameters() if parameter.requires_grad} == expected_trainable
    assert all(not parameter.requires_grad for parameter in second.visual.parameters())
    before = pretrained.model.visual.proj.detach().clone()
    with torch.no_grad():
        first.visual.proj.add_(1)
    torch.testing.assert_close(second.visual.proj, before, rtol=0, atol=0)
    torch.testing.assert_close(pretrained.model.visual.proj, before, rtol=0, atol=0)
    assert not torch.equal(first.visual.proj, before)


def test_loader_refuses_missing_unknown_and_nonpretrained_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        models.load_pretrained_clip(tmp_path / "absent.pt")
    path = tmp_path / "random.pt"
    torch.save(TinyCLIP().state_dict(), path)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        models.load_pretrained_clip(path)
    with pytest.raises(ValueError, match="unsupported"):
        models.load_pretrained_clip(path, model_name="random")


def test_loader_enforces_strict_state_keys_on_explicit_tiny_fixture(tmp_path, monkeypatch):
    """Patch digest allowlist ONLY in this fixture; production rejects this file."""
    state = TinyCLIP().state_dict()
    path = tmp_path / "tiny_fixture.pt"
    torch.save(state, path)
    monkeypatch.setattr(models, "OPENAI_CLIP_SHA256", {"test-fixture": hashlib.sha256(path.read_bytes()).hexdigest()})
    monkeypatch.setattr(models, "_load_clip_source", lambda _: (
        lambda _state: TinyCLIP(), fixture_tokenize, {"fixture_only": True}))
    loaded = models.load_pretrained_clip(path, model_name="test-fixture")
    for key, value in state.items():
        torch.testing.assert_close(loaded.model.state_dict()[key], value, atol=0, rtol=0)
    assert loaded.provenance["strict_load"]
    assert all(not p.requires_grad and p.device.type == "cpu" for p in loaded.model.parameters())
    state.pop("token_embedding.weight")
    torch.save(state, path)
    monkeypatch.setitem(models.OPENAI_CLIP_SHA256, "test-fixture", hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(RuntimeError, match="Missing key"):
        models.load_pretrained_clip(path, model_name="test-fixture")
