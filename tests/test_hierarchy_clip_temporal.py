from dataclasses import replace

import numpy as np
import pytest
import torch

from ebackbone_v3.errors import DatasetError
from ebackbone_v3.hierarchy_clip_data import (
    HierarchySequenceDataset, prepare_sequence, sequence_collate,
    sequence_contract, split_event_windows,
)
from ebackbone_v3.hierarchy_clip_temporal import (
    HierarchySequenceEncoder, HierarchyTemporalBaseline, TemporalAggregator,
)
from ebackbone_v3.hierarchy_models import HierarchyV1
from ebackbone_v3.representations import SourceIdentity, compute_event_fingerprint
from tests.test_manifest_dataset import fixture_release
from tests.test_v1_models_data import fields_source


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def _fields(times):
    n = len(times)
    fields = {"x": np.arange(n, dtype=np.uint16), "y": np.zeros(n, dtype=np.uint16),
              "t": np.array(times, dtype=np.uint16), "p": np.arange(n) % 2 == 1}
    source = SourceIdentity("train/n00000001/a.npz", "train", compute_event_fingerprint(fields),
                            times[0], times[-1], "[]", n)
    return fields, source


def _tiny_sample(index=0, counts=(2, 0, 3, 1)):
    generator = torch.Generator().manual_seed(32 + index)
    windows = []
    for count in counts:
        lower = torch.randint(0, 7 * 8 * 8, (count,), generator=generator)
        windows.append({"points": torch.rand(count, 4, generator=generator),
                        "voxel_lower": lower, "voxel_upper": lower + 64,
                        "alpha": torch.rand(count, generator=generator),
                        "event_counts": torch.tensor([count])})
    return {"windows": windows, "window_mask": torch.tensor(counts) > 0,
            "label": index % 3, "sample_id": f"train/{index}", "split": "train",
            "source_split": "train", "render_seconds": .01, "decode_seconds": .02}


def test_equal_duration_boundaries_order_conservation_and_fixed_normalization():
    fields, source = _fields((0, 1, 25, 25, 49, 50, 75, 99, 100))
    windows = split_event_windows(fields, source, num_windows=4)
    assert [w["event_count"] for w in windows] == [2, 3, 1, 3]
    np.testing.assert_array_equal(np.concatenate([w["event_indices"] for w in windows]), np.arange(9))
    for name in fields:
        np.testing.assert_array_equal(np.concatenate([w["fields"][name] for w in windows]), fields[name])
    np.testing.assert_array_equal(windows[1]["normalized_time"], [0, 0, .96])
    np.testing.assert_array_equal(windows[-1]["normalized_time"], [0, .96, 1])
    assert [w["interval_closure"] for w in windows] == ["[)", "[)", "[)", "[]"]
    prepared = prepare_sequence(fields, source, num_windows=4)
    assert sum(int(w["event_counts"].sum()) for w in prepared["windows"]) == len(fields["t"])
    assert all(w["points"].dtype == torch.float32 for w in prepared["windows"])
    assert prepared["windows"][-1]["points"][-1, 2] == 1
    assert prepared["windows"][1]["points"][0, 2] == 0
    # Uneven timestamp intervals still use equal real-duration boundaries.
    unequal_fields, unequal_source = _fields((0, 1, 2, 5))
    unequal = split_event_windows(unequal_fields, unequal_source, num_windows=3)
    assert [w["event_indices"].tolist() for w in unequal] == [[0, 1], [2], [3]]
    assert unequal[0]["normalized_time"][1] == pytest.approx(.6)


def test_zero_duration_empty_windows_and_source_validation():
    fields, source = fields_source((10, 10, 10, 10))
    prepared = prepare_sequence(fields, source, num_windows=4)
    assert prepared["window_mask"].tolist() == [False, False, False, True]
    for window in prepared["windows"][:-1]:
        assert window["points"].shape == (0, 4)
        assert window["event_counts"].tolist() == [0]
    assert torch.all(prepared["windows"][-1]["points"][:, 2] == 1)
    assert torch.all(prepared["windows"][-1]["voxel_lower"] // (120 * 160) == 7)
    assert sequence_contract(3)[1] != sequence_contract(4)[1]
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            split_event_windows(fields, source, num_windows=invalid)
    fields["x"][0] = 4
    with pytest.raises(ValueError, match="identity|subset_id"):
        prepare_sequence(fields, source)


def test_large_uint16_timestamps_do_not_overflow_when_scaling_windows():
    fields, source = _fields((60000, 62500, 65535))
    windows = split_event_windows(fields, source, num_windows=32)
    assert [w["index"] for w in windows if w["valid"]] == [0, 14, 31]
    assert sum(w["event_count"] for w in windows) == 3
    assert windows[-1]["normalized_time"].tolist() == [1.0]
    assert windows[14]["normalized_time"][0] == pytest.approx((2500 * 32 - 14 * 5535) / 5535)
    with pytest.raises(ValueError, match="too large"):
        split_event_windows(fields, source, num_windows=2 ** 63)


def test_collation_groups_and_isolates_all_windows():
    a, b = _tiny_sample(), _tiny_sample(1, counts=(0, 2, 0, 4))
    batch = sequence_collate([a, b], height=32, width=32)
    packed = batch["inputs"]["packed"]
    assert batch["sample_ids"] == ["train/0", "train/1"]
    assert batch["labels"].tolist() == [0, 1]
    assert packed["event_counts"].tolist() == [2, 0, 3, 1, 0, 2, 0, 4]
    assert batch["inputs"]["window_mask"].shape == (2, 4)
    event = 0
    for sequence_index, sample in enumerate((a, b)):
        for window_index, original in enumerate(sample["windows"]):
            n = len(original["points"])
            offset = (sequence_index * 4 + window_index) * 8 * 8 * 8
            torch.testing.assert_close(packed["voxel_lower"][event:event+n], original["voxel_lower"] + offset)
            torch.testing.assert_close(packed["points"][event:event+n], original["points"])
            event += n
    assert batch["render_seconds"] == .02
    assert batch["decode_seconds"] == .04
    a["window_mask"][1] = True
    with pytest.raises(ValueError, match="mask"):
        sequence_collate([a], height=32, width=32)


def test_immutable_adapter_label_independence_and_final_test_guard(fixture_release, monkeypatch, tmp_path):
    from ebackbone_v3 import v1_data

    def forbidden(*args, **kwargs):
        raise AssertionError("sequence path must not render raw images")

    monkeypatch.setattr(v1_data, "render_v1", forbidden)
    locations = dict(manifest_dir=fixture_release.manifest_dir, dataset_root=fixture_release.dataset_root)
    for split in ("train", "validation"):
        dataset = HierarchySequenceDataset(**locations, split=split, num_windows=4)
        sample = dataset[0]
        assert sample["split"] == split
        assert sum(int(w["event_counts"].sum()) for w in sample["windows"]) == sample["source"]["event_count"]
        row = dataset.rows[0]
        dataset.rows = (replace(row, class_label=(row.class_label + 1) % 100), *dataset.rows[1:])
        relabeled = dataset[0]
        assert sample["label"] != relabeled["label"]
        assert sample["window_metadata"] == relabeled["window_metadata"]
        for first, second in zip(sample["windows"], relabeled["windows"]):
            for key in first:
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    with pytest.raises(DatasetError, match="test|final"):
        HierarchySequenceDataset(tmp_path, tmp_path, "test")


def test_temporal_order_padding_isolation_all_empty_and_gradient():
    torch.manual_seed(3)
    temporal = TemporalAggregator(16, 4).eval()
    features = torch.randn(2, 4, 16, requires_grad=True)
    mask = torch.tensor([[True, False, True, True], [False, False, False, False]])
    output = temporal(features, mask)
    assert output.shape == (2, 16) and torch.isfinite(output).all()
    assert torch.count_nonzero(output[1]) == 0
    changed = features.detach().clone()
    changed[~mask] = float("nan")
    torch.testing.assert_close(temporal(changed, mask), output, atol=0, rtol=0)
    # Permuting content while holding positions/mask fixed changes the sequence.
    shuffled = features.detach().clone()
    shuffled[0, [0, 2]] = shuffled[0, [2, 0]]
    assert not torch.allclose(temporal(shuffled, mask)[0], output[0], atol=1e-6, rtol=1e-6)
    output[0].square().sum().backward()
    assert features.grad[mask].abs().sum() > 0
    assert torch.count_nonzero(features.grad[~mask]) == 0
    assert temporal.position.grad.abs().sum() > 0


def test_sequence_shapes_empty_maps_staged_gradients_and_strict_reload(tmp_path):
    torch.manual_seed(6)
    encoder = HierarchySequenceEncoder(HierarchyV1(num_classes=3, height=32, width=32), 4)
    model = HierarchyTemporalBaseline(encoder, num_classes=3)
    inputs = sequence_collate([_tiny_sample(), _tiny_sample(1)], height=32, width=32)["inputs"]
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    model.train()
    assert not encoder.hierarchy.training
    maps = encoder(inputs)
    assert maps.shape == (2, 4, 256, 1, 1)
    assert torch.count_nonzero(maps[:, 1]) == 0
    loss = torch.nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1]))
    loss.backward()
    assert all(p.grad is None for p in encoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.temporal.parameters())
    assert model.classifier.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    for module in (encoder.hierarchy.temporal_collapse, encoder.hierarchy.frame_stage):
        module.requires_grad_(True)
    model.train()
    assert not encoder.hierarchy.point.training
    assert encoder.hierarchy.frame_stage.training
    torch.nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1])).backward()
    for name, parameter in encoder.hierarchy.named_parameters():
        if name.startswith(("temporal_collapse.", "frame_stage.")):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    assert encoder.hierarchy.frame_stage[0].body[0].weight.grad.abs().sum() > 0
    path = tmp_path / "comparison_a.pt"
    torch.save(model.state_dict(), path)
    restored = HierarchyTemporalBaseline(
        HierarchySequenceEncoder(HierarchyV1(num_classes=3, height=32, width=32), 4), num_classes=3)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    model.eval(), restored.eval()
    torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
    # Packed execution agrees with per-sequence execution; no cross-sample mixing.
    separate = torch.cat([model(sequence_collate([_tiny_sample(i)], height=32, width=32)["inputs"])
                          for i in range(2)])
    torch.testing.assert_close(model(inputs), separate, atol=3e-6, rtol=1e-5)
    malformed = {**inputs, "window_mask": ~inputs["window_mask"]}
    with pytest.raises(ValueError, match="mask"):
        model(malformed)


def test_direct_packed_contract_rejects_cross_window_routes_and_invalid_values():
    encoder = HierarchySequenceEncoder(HierarchyV1(num_classes=3, height=32, width=32), 4)
    for key, value, message in (("voxel_lower", 512, "own sample/window"),
                                ("voxel_upper", -1, "own sample/window"),
                                ("alpha", 1.1, "alpha"),
                                ("points", float("nan"), "finite float32")):
        inputs = sequence_collate([_tiny_sample()], height=32, width=32)["inputs"]
        inputs["packed"][key][0] = value
        with pytest.raises(ValueError, match=message):
            encoder(inputs)
    inputs = sequence_collate([_tiny_sample(counts=(0, 0, 0, 0))], height=32, width=32)["inputs"]
    maps = encoder(inputs)
    assert maps.shape == (1, 4, 256, 1, 1) and torch.count_nonzero(maps) == 0
    model = HierarchyTemporalBaseline(encoder, num_classes=3)
    assert torch.isfinite(model(inputs)).all()
