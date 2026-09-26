"""Bounded CPU regression checks for preserving a trained hierarchy classifier."""

import copy
import hashlib
from dataclasses import replace

import numpy as np
import pytest
import torch

from ebackbone_v3.hierarchy_clip_data import prepare_sequence, sequence_collate
from ebackbone_v3.hierarchy_clip_preservation import (
    PreservedHierarchyReadout,
    load_full_hierarchy_checkpoint,
)
from ebackbone_v3.hierarchy_clip_temporal import HierarchySequenceEncoder
from ebackbone_v3.hierarchy_data import POINT_CONTRACT_SHA256, collate, prepare_points
from ebackbone_v3.hierarchy_models import HierarchyV1
from ebackbone_v3.representations import compute_event_fingerprint
from tests.test_hierarchy import sample as tiny_sample
from tests.test_v1_models_data import fields_source


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _native_fields(times):
    fields, source = fields_source(times)
    # Exercise both native coordinate extrema and the 4-pixel cell boundary.
    fields["x"] = np.array([0, 3, 4, 639], dtype=np.uint16)
    fields["y"] = np.array([0, 3, 4, 479], dtype=np.uint16)
    return fields, replace(source, event_subset_id=compute_event_fingerprint(fields))


@pytest.mark.parametrize("times", [
    (0, 25, 50, 100),
    (0, 0, 100, 100),
    (0, 1, 1, 7),
    (60000, 62500, 62500, 65535),
    (10, 10, 10, 10),
])
def test_one_window_preserves_native_points_routes_and_interpolation(times):
    fields, source = _native_fields(times)
    original = prepare_points(fields, source)
    sequence = prepare_sequence(fields, source, num_windows=1)
    assert sequence["window_mask"].tolist() == [True]
    assert len(sequence["windows"]) == 1
    actual = sequence["windows"][0]
    assert actual.keys() == original.keys()
    for key in original:
        assert torch.equal(actual[key], original[key]), key
    assert actual["event_counts"].tolist() == [4]
    assert actual["points"][:, :2].tolist() == original["points"][:, :2].tolist()
    assert actual["points"][0, :2].tolist() == [0.0, 0.0]
    assert actual["points"][-1, :2].tolist() == [1.0, 1.0]
    assert (actual["voxel_lower"] % (120 * 160)).tolist() == [0, 0, 161, 19199]
    assert actual["voxel_upper"][-1].item() == 8 * 120 * 160 - 1
    if times[0] == times[-1]:
        assert actual["points"][:, 2].tolist() == [1.0] * 4
        assert (actual["voxel_lower"] // (120 * 160)).tolist() == [7] * 4
    else:
        assert actual["points"][0, 2].item() == 0.0
        assert actual["points"][-1, 2].item() == 1.0


def test_one_window_collation_preserves_native_batch_offsets_and_sample_order():
    originals, sequences = [], []
    for index, times in enumerate(((0, 25, 25, 100), (10, 10, 10, 10))):
        fields, source = _native_fields(times)
        metadata = {"label": index, "sample_id": f"train/{index}", "split": "train",
                    "source_split": "train", "decode_seconds": 0.0, "render_seconds": 0.0}
        originals.append({**metadata, "inputs": prepare_points(fields, source)})
        sequences.append({**metadata, **prepare_sequence(fields, source, num_windows=1)})
    original = collate(originals)
    sequence = sequence_collate(sequences)
    assert sequence["sample_ids"] == original["sample_ids"]
    assert torch.equal(sequence["labels"], original["labels"])
    assert sequence["inputs"]["window_mask"].tolist() == [[True], [True]]
    for key, expected in original["inputs"].items():
        assert torch.equal(sequence["inputs"]["packed"][key], expected), key


def _tiny_inputs():
    samples = [tiny_sample(0), tiny_sample(3)]
    sequence_samples = [{**sample, "windows": [sample["inputs"]],
                         "window_mask": torch.tensor([True])} for sample in samples]
    original = collate(samples, height=32, width=32)["inputs"]
    sequence = sequence_collate(sequence_samples, height=32, width=32)["inputs"]
    return original, sequence


@pytest.fixture
def trained_hierarchy():
    torch.manual_seed(219)
    model = HierarchyV1(num_classes=3, height=32, width=32).eval()
    original, _ = _tiny_inputs()
    with torch.no_grad():
        features = model.pool(model.forward_features(original)).flatten(1)
    previous = {key: value.clone() for key, value in model.classifier.state_dict().items()}
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.2)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model.classifier(features), torch.tensor([1, 2]))
    loss.backward()
    optimizer.step()
    assert all(not torch.equal(previous[key], value)
               for key, value in model.classifier.state_dict().items())
    model.zero_grad(set_to_none=True)
    return model


def test_preserved_readout_matches_independent_trained_model_maps_logits_and_predictions(trained_hierarchy):
    original = trained_hierarchy
    # Independent objects catch accidental reliance on shared mutated state.
    clone = copy.deepcopy(original)
    wrapper = PreservedHierarchyReadout(clone).eval()
    original_inputs, sequence_inputs = _tiny_inputs()
    assert {id(p) for p in wrapper.parameters()} == {id(p) for p in clone.parameters()}
    assert not ({id(p) for p in wrapper.parameters()} & {id(p) for p in original.parameters()})
    with torch.no_grad():
        original_maps = original.forward_features(original_inputs)
        wrapped_maps = wrapper.forward_features(sequence_inputs)
        original_logits = original(original_inputs)
        wrapped_logits = wrapper(sequence_inputs)
    assert original_maps.shape == (2, 256, 1, 1)
    assert original_logits.shape == (2, 3)
    assert torch.equal(wrapped_maps, original_maps)
    assert torch.equal(wrapped_logits, original_logits)
    assert torch.equal(wrapped_logits.argmax(1), original_logits.argmax(1))
    # A replacement head must fail the same preservation criterion.
    random_head = torch.nn.Linear(256, 3)
    assert not torch.equal(random_head(original.pool(original_maps).flatten(1)), original_logits)


def test_constructor_rejects_multiple_windows_and_foreign_hierarchy():
    model = HierarchyV1(num_classes=3, height=32, width=32)
    valid = HierarchySequenceEncoder(model, num_windows=1)
    assert PreservedHierarchyReadout(model, encoder=valid).encoder is valid
    with pytest.raises(ValueError):
        PreservedHierarchyReadout(model, encoder=HierarchySequenceEncoder(model, num_windows=4))
    foreign = HierarchyV1(num_classes=3, height=32, width=32)
    with pytest.raises(ValueError):
        PreservedHierarchyReadout(model, encoder=HierarchySequenceEncoder(foreign, num_windows=1))


def test_preserved_readout_rejects_empty_samples_and_malformed_sequence_inputs(trained_hierarchy):
    wrapper = PreservedHierarchyReadout(trained_hierarchy).eval()
    _, sequence = _tiny_inputs()
    first_count = int(sequence["packed"]["event_counts"][0])
    # This is a consistent packed batch with one genuinely empty sample;
    # sequence encoding permits it, but the original classifier contract does not.
    packed = {key: value[:first_count] for key, value in sequence["packed"].items()
              if key != "event_counts"}
    packed["event_counts"] = torch.tensor([first_count, 0])
    invalid = {"packed": packed, "window_mask": torch.tensor([[True], [False]])}
    with pytest.raises(ValueError):
        wrapper.forward_features(invalid)
    with pytest.raises(ValueError):
        wrapper({**sequence, "window_mask": torch.ones(2, 2, dtype=torch.bool)})
    with pytest.raises(ValueError):
        wrapper({**sequence, "labels": torch.tensor([0, 1])})


def _checkpoint(model):
    return {"model": model.state_dict(), "epoch": 17,
            "identity": {"variant": "hierarchy", "point_contract_sha256": POINT_CONTRACT_SHA256}}


def test_full_checkpoint_restores_trained_classifier_and_bit_exact_predictions(tmp_path, trained_hierarchy):
    path = tmp_path / "checkpoint_best.pt"
    torch.save(_checkpoint(trained_hierarchy), path)
    restored = HierarchyV1(num_classes=3, height=32, width=32).eval()
    assert not torch.equal(restored.classifier.weight, trained_hierarchy.classifier.weight)
    provenance = load_full_hierarchy_checkpoint(restored, path)
    assert provenance["classifier_transferred"] is True
    assert provenance["epoch"] == 17
    assert provenance["path"] == str(path.resolve())
    assert provenance["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for key, expected in trained_hierarchy.state_dict().items():
        assert torch.equal(restored.state_dict()[key], expected), key
    original_inputs, sequence_inputs = _tiny_inputs()
    with torch.no_grad():
        assert torch.equal(trained_hierarchy(original_inputs),
                           PreservedHierarchyReadout(restored).eval()(sequence_inputs))


@pytest.mark.parametrize("omitted", ["classifier.weight", "classifier.bias", "point.0.weight"])
def test_full_checkpoint_rejects_missing_trained_parameters(tmp_path, trained_hierarchy, omitted):
    state = _checkpoint(trained_hierarchy)
    del state["model"][omitted]
    path = tmp_path / "incomplete.pt"
    torch.save(state, path)
    target = HierarchyV1(num_classes=3, height=32, width=32)
    previous = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(ValueError):
        load_full_hierarchy_checkpoint(target, path)
    assert all(torch.equal(target.state_dict()[key], value) for key, value in previous.items())


@pytest.mark.parametrize("field,value", [
    ("point_contract_sha256", "incompatible"),
    ("variant", "hierarchy_ts_residual"),
])
def test_full_checkpoint_rejects_incompatible_provenance(tmp_path, trained_hierarchy, field, value):
    state = _checkpoint(trained_hierarchy)
    state["identity"][field] = value
    path = tmp_path / "incompatible.pt"
    torch.save(state, path)
    target = HierarchyV1(num_classes=3, height=32, width=32)
    with pytest.raises(ValueError):
        load_full_hierarchy_checkpoint(target, path)


def test_full_checkpoint_rejects_headless_backbone_export(tmp_path, trained_hierarchy):
    state = {"variant": "hierarchy", "backbone": {
        key: value for key, value in trained_hierarchy.state_dict().items()
        if not key.startswith("classifier.")}}
    path = tmp_path / "backbone_only.pt"
    torch.save(state, path)
    with pytest.raises(ValueError):
        load_full_hierarchy_checkpoint(HierarchyV1(num_classes=3, height=32, width=32), path)
