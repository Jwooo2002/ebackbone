"""CPU equivalence checks for memory-controlled execution, not accuracy tests."""
import copy
import json

import pytest
import torch
from torch.nn import functional as F

from ebackbone_v3 import hierarchy_clip_runtime as runtime
from ebackbone_v3.hierarchy_models import HierarchyV1
from ebackbone_v3.hierarchy_clip_models import HierarchyCLIPViTAlignment
from ebackbone_v3.hierarchy_clip_temporal import HierarchySequenceEncoder
from ebackbone_v3.hierarchy_clip_data import sequence_collate
from tests.test_hierarchy_clip_temporal import _tiny_sample
from tests.test_hierarchy_clip_models import fixture_pretrained, TinySequenceEncoder, fixture_inputs


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("shape,target", [((15, 20), (14, 14)), ((3, 5), (7, 7)), ((8, 8), (2, 2))])
def test_deterministic_average_matches_adaptive_forward_and_backward(shape, target):
    torch.manual_seed(5)
    original = torch.randn(2, 3, *shape, dtype=torch.float64, requires_grad=True)
    actual_input = original.detach().clone().requires_grad_()
    expected = F.adaptive_avg_pool2d(original, target)
    layer = runtime.DeterministicAdaptiveAverage2d(shape, target)
    actual = layer(actual_input)
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(actual_input.grad, original.grad, rtol=2e-14, atol=2e-14)
    assert not layer.state_dict() and not list(layer.parameters())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        bf16 = layer(original.detach().float().bfloat16())
    assert bf16.dtype == torch.bfloat16
    torch.testing.assert_close(bf16.float(), F.adaptive_avg_pool2d(original.detach().float().bfloat16(), target).float(),
                               rtol=.015, atol=.015)


@pytest.mark.parametrize("activation_checkpointing", [False, True])
def test_hierarchy_window_chunks_preserve_maps_and_selective_gradients(activation_checkpointing):
    torch.manual_seed(23)
    hierarchy = HierarchyV1(3, 32, 32).requires_grad_(False)
    hierarchy.temporal_collapse.requires_grad_(True)
    hierarchy.frame_stage.requires_grad_(True)
    expected = HierarchySequenceEncoder(hierarchy, num_windows=4).train()
    actual = runtime.RuntimeHierarchySequenceEncoder(copy.deepcopy(hierarchy), num_windows=4,
        hierarchy_chunk_windows=2, activation_checkpointing=activation_checkpointing).train()
    samples = [_tiny_sample(0, counts=(0, 0, 2, 3)), _tiny_sample(1, counts=(2, 0, 1, 4))]
    inputs = sequence_collate(samples, height=32, width=32)["inputs"]
    sizes = []
    handle = actual.hierarchy.voxel_projection.register_forward_hook(lambda _m, _args, value: sizes.append(value.shape[0]))
    original_maps, chunk_maps = expected(inputs), actual(inputs)
    handle.remove()
    assert sizes and max(sizes) <= 2
    torch.testing.assert_close(chunk_maps, original_maps, rtol=2e-5, atol=3e-6)
    assert torch.count_nonzero(chunk_maps[0, :2]) == 0
    weights = torch.randn_like(original_maps)
    (original_maps * weights).sum().backward()
    (chunk_maps * weights).sum().backward()
    original_params = dict(expected.named_parameters())
    for name, parameter in actual.named_parameters():
        other = original_params[name]
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-3, atol=2e-4)
        else:
            assert parameter.grad is None
    assert set(expected.state_dict()) == set(actual.state_dict())
    malformed = copy.deepcopy(inputs)
    malformed["packed"]["voxel_lower"][0] = 0
    with pytest.raises(ValueError, match="own sample/window"):
        actual(malformed)


def test_checkpointed_visual_chunks_keep_frozen_tower_input_and_selective_block_gradients():
    torch.manual_seed(31)
    pretrained = fixture_pretrained()
    expected = HierarchyCLIPViTAlignment(TinySequenceEncoder(), pretrained, ["dog", "cat"])
    actual = runtime.RuntimeHierarchyCLIPViTAlignment(TinySequenceEncoder(), pretrained, ["dog", "cat"],
        visual_chunk_windows=3, activation_checkpointing=True, spatial_hw=(3, 5))
    actual.load_state_dict(expected.state_dict(), strict=True)
    expected.visual.transformer.resblocks[-1].requires_grad_(True)
    actual.visual.transformer.resblocks[-1].requires_grad_(True)
    inputs = fixture_inputs()
    wanted, observed = expected(inputs), actual(inputs)
    torch.testing.assert_close(observed, wanted, rtol=2e-5, atol=3e-6)
    weights = torch.randn_like(wanted)
    (wanted * weights).sum().backward()
    (observed * weights).sum().backward()
    original_params = dict(expected.named_parameters())
    for name, parameter in actual.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, original_params[name].grad, rtol=4e-3, atol=4e-5)
        else:
            assert parameter.grad is None, name
    assert actual.adapter.projection.weight.grad.abs().sum() > 0
    assert actual.encoder.hierarchy.weight.grad.abs().sum() > 0
    assert actual.visual.transformer.resblocks[-1].attn.in_proj_weight.grad.abs().sum() > 0
    assert set(expected.state_dict()) == set(actual.state_dict())


def test_builder_validates_minis_class_order_and_keeps_baseline_clip_optional(tmp_path, monkeypatch):
    ids = [f"n{index:08d}" for index in range(100)]
    names = [f"class {index}" for index in range(100)]
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps({"class_to_index": dict(zip(ids, range(100)))}))
    names_path = tmp_path / "names.json"
    names_path.write_text(json.dumps({"class_ids": ids, "class_names": names}))
    config = {"manifest_dir": str(tmp_path), "class_names_path": str(names_path),
              "hierarchy_checkpoint": "fixture-loader-only", "execution": {"hierarchy_chunk_windows": 2}}
    monkeypatch.setattr(runtime, "load_hierarchy_backbone", lambda model, path: {"fixture_only": True})
    monkeypatch.setattr(runtime, "load_pretrained_clip", lambda *args, **kwargs: pytest.fail("baseline must not load CLIP"))
    model, identity = runtime.build_model("hierarchy_temporal", config)
    assert model.classifier.out_features == 100
    assert identity["clip"] is None and identity["raw_rgb_used"] is False
    assert model.encoder.hierarchy_chunk_windows == 2
    names_path.write_text(json.dumps({"class_ids": ids[::-1], "class_names": names}))
    with pytest.raises(ValueError, match="class order"):
        runtime.build_model("hierarchy_temporal", config)
    with pytest.raises(ValueError, match="unknown"):
        runtime.build_model("rgb_model", config)
