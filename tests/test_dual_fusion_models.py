from dataclasses import replace

import pytest
import torch

from ebackbone_v3.dual_fusion_data import (
    DualFusionDataset, MODES, collate, input_keys, prepare_inputs,
)
from ebackbone_v3.dual_fusion_models import (
    DualFusionBackbone, export_backbone, load_backbone_export, profile_macs,
)
from ebackbone_v3.errors import DatasetError
from ebackbone_v3.hierarchy_data import prepare_points
from ebackbone_v3.hierarchy_models import HierarchyV1
from ebackbone_v3.v1_data import render_v1
from tests.test_hierarchy import sample as point_sample
from tests.test_manifest_dataset import fixture_release
from tests.test_v1_models_data import fields_source


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sample(index=0, mode="dual"):
    result = point_sample(index)
    generator = torch.Generator().manual_seed(70 + index)
    result["inputs"].update(event_frame=torch.rand(2, 32, 32, generator=generator),
                            voxel_grid=torch.rand(2, 8, 32, 32, generator=generator),
                            time_surface=torch.rand(2, 32, 32, generator=generator))
    result["inputs"] = {key: value for key, value in result["inputs"].items() if key in input_keys(mode)}
    return result


def batch(mode="dual"):
    return collate([sample(0, mode), sample(1, mode)], height=32, width=32)


def test_shared_initialization_isolation_and_unchanged_hierarchy():
    state = torch.get_rng_state().clone()
    models = {mode: DualFusionBackbone(mode, height=32, width=32, seed=44) for mode in MODES}
    assert torch.equal(state, torch.get_rng_state())
    dual = models["dual"]
    assert dual.a.ndim == 0 and dual.a.item() == 0 and dual.a.sigmoid().item() == .5
    assert not hasattr(dual.hierarchy, "classifier")
    for mode, model in models.items():
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, dual.state_dict()[name], atol=0, rtol=0)
        if mode != "dual":
            assert not hasattr(model, "a")
    assert not hasattr(models["hierarchy_only"], "latent")
    assert not hasattr(models["latent_only"], "hierarchy")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(44)
        original = HierarchyV1(height=32, width=32)
    for name, value in dual.hierarchy.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize("mode", MODES)
def test_tensor_contract_gradients_updates_checkpoint_and_export(mode, tmp_path):
    model = DualFusionBackbone(mode, height=32, width=32)
    inputs, labels = batch(mode)["inputs"], batch(mode)["labels"]
    embeddings = model.branch_embeddings(inputs)
    assert all(value.shape == (2, 256) for value in embeddings.values())
    for value in embeddings.values():
        torch.testing.assert_close(value.mean(-1), torch.zeros(2), atol=1e-6, rtol=0)
    assert model(inputs).shape == (2, 100)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    loss = torch.nn.functional.cross_entropy(model(inputs), labels)
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for name, module in model.named_modules():
        parameters = list(module.parameters(recurse=False))
        if parameters:
            assert sum(p.grad.abs().sum().item() for p in parameters) > 0, name
    if mode == "dual":
        assert model.a.grad.abs().item() > 0
    optimizer.step()
    for name, module in model.named_children():
        assert any(not torch.equal(before[n], p) for n, p in model.named_parameters()
                   if n.startswith(name + ".")), name
    if mode == "dual":
        assert model.a.item() != 0
    model.eval()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    restored = DualFusionBackbone(**model.construction_config()).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)["model"], strict=True)
    torch.testing.assert_close(restored(inputs), model(inputs), atol=0, rtol=0)
    path = tmp_path / "backbone.pt"
    export_backbone(model, path)
    exported = load_backbone_export(path)
    assert not hasattr(exported, "classifier")
    assert not any("classifier" in name for name in exported.state_dict())
    torch.testing.assert_close(exported(inputs), model.forward_embedding(inputs), atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        export_backbone(model, path)


def test_dual_weighted_formula_endpoints_ablation_and_batch_isolation():
    model = DualFusionBackbone(height=32, width=32).eval()
    inputs = batch()["inputs"]
    branches = model.branch_embeddings(inputs)
    torch.testing.assert_close(model.forward_embedding(inputs), (branches["hierarchy"] + branches["latent"]) / 2,
                               atol=0, rtol=0)
    for scalar, branch, mode in ((-100., "hierarchy", "hierarchy_only"), (100., "latent", "latent_only")):
        with torch.no_grad():
            model.a.fill_(scalar)
        ablation = DualFusionBackbone(mode, height=32, width=32).eval()
        torch.testing.assert_close(model.forward_embedding(inputs), branches[branch], atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(model(inputs), ablation({k: inputs[k] for k in input_keys(mode)}), atol=1e-7, rtol=1e-7)
    with torch.no_grad():
        model.a.zero_()
    together = model(inputs)
    separate = torch.cat([model(collate([sample(i)], height=32, width=32)["inputs"]) for i in (0, 1)])
    torch.testing.assert_close(together, separate, atol=3e-6, rtol=1e-5)
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["event_frame"].zero_()
    assert not torch.allclose(model(changed), together)
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["points"].zero_()
    assert not torch.allclose(model(changed), together)


def test_full_event_alignment_zero_duration_and_no_window_splitting():
    for times in ((0, 25, 50, 100), (10, 10, 10, 10)):
        fields, source = fields_source(times)
        dual = prepare_inputs(fields, source)
        expected_points = prepare_points(fields, source)
        expected_views = render_v1(fields, source, multiview=True)
        assert dual["points"].shape == (source.event_count, 4)
        assert dual["event_counts"].tolist() == [4]
        for key, value in expected_points.items():
            torch.testing.assert_close(dual[key], value, atol=0, rtol=0)
        for key, value in expected_views.items():
            torch.testing.assert_close(dual[key], torch.from_numpy(value), atol=0, rtol=0)
        torch.testing.assert_close(dual["event_frame"].expm1().sum(), torch.tensor(4.))
        torch.testing.assert_close(dual["voxel_grid"].expm1().sum(), torch.tensor(4.))
        assert dual["voxel_grid"].shape == (2, 8, 480, 640)
        for mode in MODES:
            prepared = prepare_inputs(fields, source, mode=mode)
            for key, value in prepared.items():
                torch.testing.assert_close(value, dual[key], atol=0, rtol=0)


def test_dataset_identity_labels_and_final_test_gate(fixture_release, tmp_path):
    data = DualFusionDataset(fixture_release.manifest_dir, fixture_release.dataset_root, "train")
    original = data[0]
    row = data.rows[0]
    data.rows = (replace(row, class_label=(row.class_label + 1) % 100), *data.rows[1:])
    changed = data[0]
    assert original["source"] == changed["source"]
    assert original["label"] != changed["label"]
    for key in original["inputs"]:
        torch.testing.assert_close(original["inputs"][key], changed["inputs"][key], atol=0, rtol=0)
    with pytest.raises(DatasetError, match="test|final"):
        DualFusionDataset(tmp_path, tmp_path, "test")


def test_profile_event_dependence_and_additive_compute():
    profiles = {mode: profile_macs(DualFusionBackbone(mode), event_count=4) for mode in MODES}
    assert profiles["hierarchy_only"]["parameters"] == 2737028
    assert profiles["latent_only"]["parameters"] == 300940
    assert profiles["dual"]["parameters"] == 3012269
    assert profiles["dual"]["macs"] == profiles["hierarchy_only"]["macs"] + profiles["latent_only"]["macs"] - 25600
    for mode in MODES:
        later = profile_macs(DualFusionBackbone(mode), event_count=17)
        assert later["macs"] - profiles[mode]["macs"] == (13 * 1152 if mode != "latent_only" else 0)


def test_invalid_inputs_and_tampered_exports(tmp_path):
    model = DualFusionBackbone(height=32, width=32)
    inputs = batch()["inputs"]
    with pytest.raises(ValueError, match="exactly"):
        model({**inputs, "clip": torch.zeros(1)})
    with pytest.raises(ValueError, match="event_frame"):
        model({**inputs, "event_frame": inputs["event_frame"][:, :, :16]})
    with pytest.raises(ValueError, match="int64"):
        model({**inputs, "event_counts": inputs["event_counts"].float()})
    path = tmp_path / "backbone.pt"
    export_backbone(model, path)
    payload = torch.load(path, weights_only=True)
    payload["input_contract_sha256"] = "changed"
    torch.save(payload, tmp_path / "bad_contract.pt")
    with pytest.raises(ValueError, match="contract"):
        load_backbone_export(tmp_path / "bad_contract.pt")
    payload = torch.load(path, weights_only=True)
    payload["state_dict"]["classifier.weight"] = model.classifier.weight.detach()
    torch.save(payload, tmp_path / "bad_state.pt")
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_backbone_export(tmp_path / "bad_state.pt")
