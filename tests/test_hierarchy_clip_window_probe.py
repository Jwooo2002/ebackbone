"""CPU checks for isolating window partitioning from time normalization."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from ebackbone_v3.hierarchy_clip_data import prepare_sequence, sequence_collate
from ebackbone_v3.hierarchy_clip_runtime import RuntimeHierarchySequenceEncoder
from ebackbone_v3.hierarchy_clip_window_probe import FrozenWindowReadout, prepare_window_control
from ebackbone_v3.hierarchy_data import prepare_points
from ebackbone_v3.hierarchy_models import HierarchyV1
from ebackbone_v3.representations import SourceIdentity, compute_event_fingerprint
from tests.test_hierarchy_clip_preservation import trained_hierarchy
from tests.test_hierarchy_clip_temporal import _tiny_sample


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _native_events(times):
    count = len(times)
    fields = {"x": np.resize(np.array([0, 3, 4, 639], dtype=np.uint16), count),
              "y": np.resize(np.array([0, 3, 4, 479], dtype=np.uint16), count),
              "t": np.asarray(times, dtype=np.uint16),
              "p": np.arange(count) % 2 == 1}
    source = SourceIdentity("train/n00000001/control.npz", "train",
                            compute_event_fingerprint(fields), times[0], times[-1], "[]", count)
    return fields, source


def _assert_same_sequence(actual, expected):
    assert torch.equal(actual["window_mask"], expected["window_mask"])
    assert actual["window_metadata"] == expected["window_metadata"]
    assert len(actual["windows"]) == len(expected["windows"])
    for left, right in zip(actual["windows"], expected["windows"]):
        assert left.keys() == right.keys()
        for key in left:
            assert torch.equal(left[key], right[key]), key


def test_local_control_is_exactly_existing_preprocessing():
    fields, source = _native_events((0, 1, 25, 25, 49, 50, 75, 99, 100))
    actual = prepare_window_control(fields, source, num_windows=4, time_mode="local")
    _assert_same_sequence(actual, prepare_sequence(fields, source, num_windows=4))
    assert actual["time_mode"] == "local"


def test_global_control_preserves_all_original_time_routes_with_same_window_membership():
    fields, source = _native_events((0, 1, 25, 25, 49, 50, 75, 99, 100))
    original = prepare_points(fields, source)
    local = prepare_window_control(fields, source, num_windows=4, time_mode="local")
    global_time = prepare_window_control(fields, source, num_windows=4, time_mode="global")
    assert global_time["time_mode"] == "global"
    assert global_time["window_mask"].tolist() == [True, True, True, True]
    assert torch.equal(global_time["window_mask"], local["window_mask"])
    assert global_time["window_metadata"] == local["window_metadata"]
    # Explicit membership checks catch tie/boundary duplication and omissions.
    indices_by_window = [[0, 1], [2, 3, 4], [5], [6, 7, 8]]
    for window, indices in zip(global_time["windows"], indices_by_window):
        assert window["event_counts"].tolist() == [len(indices)]
        for key in ("points", "voxel_lower", "voxel_upper", "alpha"):
            assert torch.equal(window[key], original[key][indices]), key
    for key in ("points", "voxel_lower", "voxel_upper", "alpha"):
        assert torch.equal(torch.cat([w[key] for w in global_time["windows"]]), original[key]), key
    assert global_time["windows"][1]["points"][0, 2].item() == 0.25
    assert local["windows"][1]["points"][0, 2].item() == 0.0
    assert (global_time["windows"][1]["voxel_lower"][0] // (120 * 160)).item() == 1
    assert global_time["windows"][1]["alpha"][0].item() == 0.75
    # Window division changes time only; it never changes native x/y/p fields.
    for local_window, global_window in zip(local["windows"], global_time["windows"]):
        assert torch.equal(local_window["points"][:, [0, 1, 3]],
                           global_window["points"][:, [0, 1, 3]])


def test_zero_duration_keeps_all_events_in_last_window_and_last_global_bin():
    fields, source = _native_events((10, 10, 10, 10))
    local = prepare_window_control(fields, source, num_windows=4, time_mode="local")
    global_time = prepare_window_control(fields, source, num_windows=4, time_mode="global")
    _assert_same_sequence(global_time, local)
    assert global_time["window_mask"].tolist() == [False, False, False, True]
    for window in global_time["windows"][:-1]:
        assert window["points"].shape == (0, 4)
        assert window["event_counts"].tolist() == [0]
    last = global_time["windows"][-1]
    assert last["event_counts"].tolist() == [4]
    assert last["points"][:, 2].tolist() == [1.0] * 4
    assert (last["voxel_lower"] // (120 * 160)).tolist() == [7] * 4
    assert torch.equal(last["voxel_lower"], last["voxel_upper"])
    assert last["alpha"].tolist() == [0.0] * 4


@pytest.mark.parametrize("mode", ["local", "global"])
@pytest.mark.parametrize("times", [(0, 25, 25, 100), (60000, 62500, 62500, 65535), (10, 10, 10, 10)])
def test_k1_control_has_exact_original_whole_event_inputs(mode, times):
    fields, source = _native_events(times)
    original = prepare_points(fields, source)
    actual = prepare_window_control(fields, source, num_windows=1, time_mode=mode)
    assert actual["window_mask"].tolist() == [True]
    assert len(actual["windows"]) == 1
    for key in original:
        assert torch.equal(actual["windows"][0][key], original[key]), key


@pytest.mark.parametrize("mode", ["unsupported", "LOCAL", None])
def test_control_rejects_unknown_time_modes(mode):
    fields, source = _native_events((0, 25, 50, 100))
    with pytest.raises(ValueError):
        prepare_window_control(fields, source, time_mode=mode)


class _MapEncoder(nn.Module):
    def __init__(self, hierarchy, maps):
        super().__init__()
        self.hierarchy = hierarchy
        self.num_windows = maps.shape[1]
        self.register_buffer("maps", maps, persistent=False)

    def forward(self, inputs):
        return self.maps


def test_equal_mean_excludes_masked_windows_and_is_permutation_invariant():
    hierarchy = HierarchyV1(num_classes=3, height=32, width=32).eval()
    model = FrozenWindowReadout(hierarchy, num_windows=4).eval()
    values = torch.tensor([[1.0, 10000.0, 3.0, 8.0], [-20000.0, 2.0, 40000.0, 6.0]])
    maps = values[:, :, None, None, None].expand(2, 4, 256, 2, 2).clone()
    mask = torch.tensor([[True, False, True, True], [False, True, False, True]])
    model.encoder = _MapEncoder(hierarchy, maps)
    inputs = {"window_mask": mask}
    features = model.forward_features(inputs)
    torch.testing.assert_close(features, torch.full((2, 256), 4.0), atol=0, rtol=0)
    assert model.window_features(inputs).shape == (2, 4, 256)
    permutation = torch.tensor([2, 0, 3, 1])
    model.encoder = _MapEncoder(hierarchy, maps[:, permutation])
    permuted_inputs = {"window_mask": mask[:, permutation]}
    assert torch.equal(model.forward_features(permuted_inputs), features)
    assert torch.equal(model(permuted_inputs), hierarchy.classifier(features))


def test_real_readout_preserves_trained_head_and_all_model_weights(trained_hierarchy):
    reference = copy.deepcopy(trained_hierarchy)
    model = FrozenWindowReadout(trained_hierarchy, num_windows=4).eval()
    samples = [_tiny_sample(0, counts=(2, 0, 3, 1)), _tiny_sample(1, counts=(0, 2, 0, 4))]
    inputs = sequence_collate(samples, height=32, width=32)["inputs"]
    mask = inputs["window_mask"]
    before = {key: value.clone() for key, value in trained_hierarchy.state_dict().items()}
    assert {id(p) for p in model.parameters()} == {id(p) for p in trained_hierarchy.parameters()}
    with torch.no_grad():
        raw_features = reference.pool(reference.forward_features(inputs["packed"])).flatten(1)
        per_window = raw_features.reshape(2, 4, 256).masked_fill(~mask[..., None], 0)
        expected_features = per_window.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        actual_features = model.forward_features(inputs)
        actual_logits = model(inputs)
        assert torch.equal(model.window_features(inputs), per_window)
        assert torch.equal(actual_features, expected_features)
        assert torch.equal(actual_logits, reference.classifier(expected_features))
        # Averaging original per-window classifier logits independently checks
        # preservation of both the trained weight and the bias.
        logits = reference(inputs["packed"]).reshape(2, 4, 3)
        expected_logits = logits.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)
        torch.testing.assert_close(actual_logits, expected_logits, atol=2e-6, rtol=1e-6)
    assert all(torch.equal(trained_hierarchy.state_dict()[key], value) for key, value in before.items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_readout_rejects_an_all_empty_sample_even_with_other_valid_samples(trained_hierarchy):
    model = FrozenWindowReadout(trained_hierarchy, num_windows=4).eval()
    samples = [_tiny_sample(0, counts=(2, 0, 3, 1)), _tiny_sample(1, counts=(0, 0, 0, 0))]
    inputs = sequence_collate(samples, height=32, width=32)["inputs"]
    with pytest.raises(ValueError):
        model(inputs)


def test_runtime_single_valid_window_matches_original_logits_and_skips_empty_windows(trained_hierarchy):
    reference = copy.deepcopy(trained_hierarchy).eval().requires_grad_(False)
    trained_hierarchy.eval().requires_grad_(False)
    encoder = RuntimeHierarchySequenceEncoder(trained_hierarchy, num_windows=4,
                                               hierarchy_chunk_windows=1,
                                               activation_checkpointing=False)
    model = FrozenWindowReadout(trained_hierarchy, num_windows=4, encoder=encoder).eval()
    sample = _tiny_sample(1, counts=(0, 0, 5, 0))
    inputs = sequence_collate([sample], height=32, width=32)["inputs"]
    point_calls = []
    handle = trained_hierarchy.point.register_forward_hook(
        lambda module, args, output: point_calls.append(output.shape[0]))
    try:
        with torch.no_grad():
            actual = model(inputs)
            expected = reference(sample["windows"][2])
    finally:
        handle.remove()
    assert point_calls == [5]
    assert torch.equal(actual, expected)
