from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from ebackbone_v3 import hierarchy_training as training
from ebackbone_v3.errors import DatasetError
from ebackbone_v3.hierarchy_data import HierarchyDataset, collate, prepare_points
from ebackbone_v3.hierarchy_models import HierarchyV1, PointToVoxel, profile_macs
from tests.test_manifest_dataset import fixture_release
from tests.test_v1_models_data import fields_source
from tests.test_v1_training import assert_nested_equal


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_point_contract_identity_boundary_and_zero_duration():
    fields, source = fields_source()
    inputs = prepare_points(fields, source)
    assert inputs["points"].shape == (4, 4)
    assert inputs["points"].dtype == torch.float32
    torch.testing.assert_close(inputs["points"][-1], torch.tensor([1., 1., 1., 1.]))
    torch.testing.assert_close(inputs["points"][0], torch.tensor([0., 0., 0., -1.]))
    assert inputs["voxel_upper"][-1] == 8 * 120 * 160 - 1
    assert inputs["alpha"][1] == .75  # t=25/100 -> bin 1 + 0.75
    fields, source = fields_source((10, 10, 10, 10))
    zero = prepare_points(fields, source)
    assert torch.all(zero["points"][:, 2] == 1)
    assert torch.all(zero["voxel_lower"] // (120 * 160) == 7)
    fields["x"][0] = 4
    with pytest.raises(ValueError, match="identity|subset_id"):
        prepare_points(fields, source)


def test_weighted_mean_count_conservation_and_point_gradients():
    fields, source = fields_source()
    inputs = prepare_points(fields, source)
    values = torch.arange(4, dtype=torch.float32)[:, None].expand(-1, 32).clone().requires_grad_()
    grid = PointToVoxel()(values, inputs)
    assert grid.shape == (1, 33, 8, 120, 160)
    mass = grid[:, 32].expm1()
    torch.testing.assert_close(mass.sum(), torch.tensor(4.))
    assert mass[0, 1, 0, 0].item() == pytest.approx(.25)
    assert mass[0, 2, 0, 0].item() == pytest.approx(.75)
    assert grid[0, 0, 1, 0, 0] == 1  # weighted mean, not attenuated by bin mass
    assert grid[0, 0, 2, 0, 0] == 1
    assert grid[0, 0, 7, -1, -1] == 3
    assert torch.count_nonzero(grid[:, :, :, 10, 10]) == 0
    grid[:, :32].sum().backward()
    assert torch.all(values.grad > 0)


def sample(index=0, split="train"):
    n = 5 + index
    g = torch.Generator().manual_seed(10 + index)
    return {"inputs": {"points": torch.rand(n, 4, generator=g),
                       "voxel_lower": torch.randint(0, 7 * 8 * 8, (n,), generator=g),
                       "voxel_upper": torch.randint(0, 8 * 8 * 8, (n,), generator=g),
                       "alpha": torch.rand(n, generator=g), "event_counts": torch.tensor([n])},
            "label": index % 2, "sample_id": f"{split}/{index}", "split": split,
            "source_split": "train", "source": {"sample_id": f"{split}/{index}"},
            "render_seconds": .01, "decode_seconds": .02}


def test_packed_batch_isolation_and_all_components_receive_ce_gradients(tmp_path):
    torch.manual_seed(6)
    model = HierarchyV1(height=32, width=32)
    a, b = sample(0), sample(3)
    inputs = collate([a, b], height=32, width=32)["inputs"]
    assert inputs["points"].shape == (13, 4)
    assert inputs["event_counts"].tolist() == [5, 8]
    assert torch.all(inputs["voxel_lower"][5:] >= 512)
    model.eval()
    individual = torch.cat([model(s["inputs"]) for s in (a, b)])
    # Convolution kernels may change summation order with batch size.
    torch.testing.assert_close(model(inputs), individual, atol=3e-6, rtol=1e-5)
    packed_grid = model.point_to_voxel(model.point(inputs["points"]), inputs)
    separate_grid = torch.cat([model.point_to_voxel(model.point(s["inputs"]["points"]), s["inputs"])
                               for s in (a, b)])
    torch.testing.assert_close(packed_grid, separate_grid, atol=1e-6, rtol=1e-6)
    assert model.forward_features(inputs).shape == (2, 256, 1, 1)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    loss = torch.nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1]))
    assert torch.isfinite(loss)
    loss.backward()
    for name, module in model.named_children():
        params = list(module.parameters())
        if params:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params), name
            assert sum(p.grad.abs().sum().item() for p in params) > 0, name
    optimizer.step()
    for name, module in model.named_children():
        if list(module.parameters()):
            assert any(not torch.equal(before[n], p) for n, p in model.named_parameters()
                       if n.startswith(name + ".")), name
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = HierarchyV1(height=32, width=32).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(model(inputs), restored(inputs), atol=0, rtol=0)
    # Suppressing learned point features must affect the final classifier.
    hook = model.point.register_forward_hook(lambda module, args, output: output * 0)
    altered = model(inputs)
    hook.remove()
    assert not torch.allclose(altered, model(inputs))
    with pytest.raises(ValueError, match="exactly"):
        model({**inputs, "event_frame": torch.zeros(2, 2, 32, 32)})


def test_profile_event_dependence_and_actual_shapes():
    model = HierarchyV1()
    a, b = profile_macs(model, 4), profile_macs(model, 17)
    assert a["parameters"] == 2670724
    assert b["macs"] - a["macs"] == 13 * 1152
    assert a["fixed_macs"] == 7240115200
    assert a["shapes_batch1"]["point_to_voxel"] == [1, 33, 8, 120, 160]
    assert a["shapes_batch1"]["frame_stage"] == [1, 256, 15, 20]


def test_real_adapter_no_image_rendering_or_label_input(fixture_release, monkeypatch, tmp_path):
    from ebackbone_v3 import v1_data

    def forbidden(*args, **kwargs):
        raise AssertionError("hierarchy must not render raw frames/voxels/TS")

    monkeypatch.setattr(v1_data, "render_v1", forbidden)
    locations = dict(manifest_dir=fixture_release.manifest_dir, dataset_root=fixture_release.dataset_root)
    data = HierarchyDataset(**locations, split="train")
    item = data[0]
    assert len(item["inputs"]["points"]) == item["source"]["event_count"]
    row = data.rows[0]
    data.rows = (replace(row, class_label=(row.class_label + 1) % 100), *data.rows[1:])
    changed_label = data[0]
    for key in item["inputs"]:
        torch.testing.assert_close(item["inputs"][key], changed_label["inputs"][key], atol=0, rtol=0)
    with pytest.raises(DatasetError, match="test|final"):
        HierarchyDataset(tmp_path, tmp_path, "test")


class FakeDataset:
    def __init__(self, manifest_dir=None, dataset_root=None, split="train", *, raw_cache=None, limit=None, seed=None):
        self.split = split
        self.rows = [SimpleNamespace(sample_id=f"{split}/{i}", raw_content_sha256=f"{split}:{i}")
                     for i in range(limit or (5 if split == "train" else 3))]
        self.manifest_sha256 = split + "_immutable"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return sample(index, self.split)


@pytest.fixture
def runner_config(monkeypatch):
    monkeypatch.setattr(training, "HierarchyDataset", FakeDataset)
    # Restore accepts explicit dimensions, so retain a normal signature.
    monkeypatch.setattr(training, "HierarchyV1", lambda num_classes=100, height=32, width=32:
                        HierarchyV1(num_classes, height, width))
    monkeypatch.setattr(training, "collate", partial(collate, height=32, width=32))
    return training.TrainConfig(epochs=2, batch_size=2, accumulation_steps=2, device="cpu",
                                precision="float32", num_workers=0, cpu_threads=1)


def run(config, output, **kwargs):
    return training.run_model("hierarchy", config, manifest_dir="unused", dataset_root="unused",
                              output_dir=output, **kwargs)


def test_runner_exact_resume_and_guards(runner_config, tmp_path):
    # Single-thread CPU isolates resume state from tiny multi-thread reduction
    # differences observed between independent runs on this PyTorch build.
    whole = run(runner_config, tmp_path / "whole")
    assert run(runner_config, tmp_path / "resume", stop_after_epoch=1)["status"] == "partial"
    resumed = run(runner_config, tmp_path / "resume", resume=True)
    a = torch.load(tmp_path / "whole/checkpoint_last.pt", weights_only=False)
    b = torch.load(tmp_path / "resume/checkpoint_last.pt", weights_only=False)
    for key in ("model", "optimizer", "scheduler"):
        assert_nested_equal(a[key], b[key])
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert resumed["checkpoint_verification"]["logits_bit_exact"]
    for x, y in zip(whole["history"], resumed["history"]):
        for split in ("train", "validation"):
            for key in ("loss", "top1", "top5", "sample_order_sha256", "samples", "events"):
                assert x[split][key] == y[split][key]
    with pytest.raises(ValueError, match="overwrite"):
        run(runner_config, tmp_path / "whole")
    with pytest.raises(ValueError, match="identity mismatch"):
        run(replace(runner_config, learning_rate=.03), tmp_path / "resume", resume=True)


def test_packed_accumulation_matches_effective_batch(runner_config):
    torch.manual_seed(8)
    first, second = HierarchyV1(height=32, width=32), HierarchyV1(height=32, width=32)
    second.load_state_dict(first.state_dict())
    a, b = (torch.optim.SGD(m.parameters(), lr=.001) for m in (first, second))
    training.run_epoch(first, FakeDataset(), runner_config, 1, optimizer=a)
    training.run_epoch(second, FakeDataset(), replace(runner_config, batch_size=4, accumulation_steps=1), 1, optimizer=b)
    for p, q in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)


def test_config_rejected_before_data(runner_config, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("must validate before reading data")
    monkeypatch.setattr(training, "HierarchyDataset", forbidden)
    for change in ({"epochs": 0}, {"accumulation_steps": 0}, {"device": "cuda:0"}, {"learning_rate": float("nan")}):
        with pytest.raises(ValueError):
            run(replace(runner_config, **change), tmp_path / "unused")
