from dataclasses import replace
from functools import partial

import numpy as np
import pytest
import torch

from ebackbone_v3 import hierarchy_ts_training as training
from ebackbone_v3.errors import DatasetError
from ebackbone_v3.hierarchy_data import HierarchyDataset
from ebackbone_v3.hierarchy_models import HierarchyV1
from ebackbone_v3.hierarchy_ts_data import HierarchyTSDataset, collate, render_time_surface
from ebackbone_v3.hierarchy_ts_models import HierarchyTSV1, profile_macs
from ebackbone_v3.v1_data import render_v1
from tests.test_hierarchy import FakeDataset as PointFakeDataset, sample as point_sample
from tests.test_manifest_dataset import fixture_release
from tests.test_v1_models_data import fields_source
from tests.test_v1_training import assert_nested_equal


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("times", [(0, 25, 50, 100), (10, 10, 10, 10)])
def test_ts_latest_polarity_background_and_reference_equivalence(times):
    fields, source = fields_source(times)
    ts = render_time_surface(fields, source)
    np.testing.assert_array_equal(ts.numpy(), render_v1(fields, source, multiview=True)["time_surface"])
    assert ts.shape == (2, 480, 640) and ts.dtype == torch.float32
    assert ts[1, 479, 639] == 1 and ts[1, 0, 0] == 0
    assert ts[0, 0, 0] == pytest.approx(1 if times[0] == times[-1] else np.exp(-.75/.2))
    fields["x"][0] = 4
    with pytest.raises(ValueError, match="identity|subset_id"):
        render_time_surface(fields, source)


def sample(index=0, split="train"):
    item = point_sample(index, split)
    item["inputs"]["time_surface"] = torch.rand(2, 32, 32, generator=torch.Generator().manual_seed(index))
    return item


def test_identity_same_backbone_rng_two_step_gradients_and_reload(tmp_path):
    torch.manual_seed(42)
    original = HierarchyV1(height=32, width=32).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    model = HierarchyTSV1(height=32, width=32).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in original.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
    inputs = collate([sample(0), sample(1)], height=32, width=32)["inputs"]
    point_inputs = {k: v for k, v in inputs.items() if k != "time_surface"}
    torch.testing.assert_close(model(inputs), original(point_inputs), atol=0, rtol=0)
    model.capture_gate_stats = True
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, weight_decay=0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1]))
        loss.backward()
        for name, module in model.named_children():
            params = list(module.parameters())
            if not params:
                continue
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params), name
            grad = sum(float(p.grad.abs().sum()) for p in params)
            if step == 0 and name == "ts_encoder":
                assert grad == 0
            else:
                assert grad > 0, name
        optimizer.step()
    assert .5 <= model.last_gate_stats["multiplier_min"] <= model.last_gate_stats["multiplier_max"] <= 1.5
    assert model.last_gate_stats["relative_feature_change"] > 0
    other = {**inputs, "time_surface": inputs["time_surface"].flip(0)}
    assert not torch.equal(model(inputs), model(other))
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = HierarchyTSV1(height=32, width=32).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(model(inputs), restored(inputs), atol=0, rtol=0)
    individual = torch.cat([model(collate([sample(i)], height=32, width=32)["inputs"]) for i in range(2)])
    torch.testing.assert_close(model(inputs), individual, atol=3e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="exactly"):
        model(point_inputs)
    with pytest.raises(ValueError, match="aligned"):
        model({**inputs, "time_surface": inputs["time_surface"][:1]})


def test_meta_profile():
    a = profile_macs(HierarchyTSV1())
    assert a["parameters"] == 2739588
    assert a["macs"] == 7627790080
    assert a["macs_by_stage"]["ts_encoder"] + a["macs_by_stage"]["ts_gate"] == 253132800
    assert a["shapes_batch1"]["ts_gate"] == [1, 128, 30, 40]
    assert a["shapes_batch1"]["frame_stage"] == [1, 256, 15, 20]


def test_one_decode_same_source_label_isolation_and_no_other_rendering(fixture_release, monkeypatch, tmp_path):
    from ebackbone_v3 import v1_data
    locations = dict(manifest_dir=fixture_release.manifest_dir, dataset_root=fixture_release.dataset_root)
    original = HierarchyDataset(**locations, split="train")[0]
    calls = []
    decode = v1_data.raw._decode_event_fields

    def tracked(*args, **kwargs):
        calls.append(1)
        return decode(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("raw frame/voxel rendering must not run")

    monkeypatch.setattr(v1_data.raw, "_decode_event_fields", tracked)
    monkeypatch.setattr(v1_data, "render_v1", forbidden)
    data = HierarchyTSDataset(**locations, split="train")
    item = data[0]
    assert len(calls) == 1 and item["source"] == original["source"]
    for key, value in original["inputs"].items():
        assert torch.equal(value, item["inputs"][key])
    row = data.rows[0]
    data.rows = (replace(row, class_label=(row.class_label + 1) % 100), *data.rows[1:])
    altered = data[0]
    assert all(torch.equal(v, altered["inputs"][k]) for k, v in item["inputs"].items())
    with pytest.raises(DatasetError, match="test|final"):
        HierarchyTSDataset(tmp_path, tmp_path, "test")


class FakeDataset(PointFakeDataset):
    def __getitem__(self, index):
        return sample(index, self.split)


def test_runner_resume_and_gate_diagnostics(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "HierarchyDataset", FakeDataset)
    monkeypatch.setattr(training, "HierarchyV1", lambda num_classes=100, height=32, width=32:
                        HierarchyTSV1(num_classes, height, width))
    monkeypatch.setattr(training, "collate", partial(collate, height=32, width=32))
    config = training.TrainConfig(epochs=2, batch_size=2, accumulation_steps=1, device="cpu",
                                  precision="float32", num_workers=0, cpu_threads=1)

    def run(out, **kwargs):
        return training.run_model("hierarchy_ts", config, manifest_dir="unused", dataset_root="unused",
                                  output_dir=tmp_path / out, **kwargs)

    whole = run("whole")
    assert run("resume", stop_after_epoch=1)["status"] == "partial"
    resumed = run("resume", resume=True)
    a, b = (torch.load(tmp_path / name / "checkpoint_last.pt", weights_only=False) for name in ("whole", "resume"))
    for key in ("model", "optimizer", "scheduler"):
        assert_nested_equal(a[key], b[key])
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    assert a["rng"]["python"] == b["rng"]["python"]
    for x, y in zip(a["rng"]["numpy"], b["rng"]["numpy"]):
        np.testing.assert_array_equal(x, y)
    assert resumed["checkpoint_verification"]["logits_bit_exact"]
    for x, y in zip(whole["history"], resumed["history"]):
        for split in ("train", "validation"):
            for key in ("loss", "top1", "samples", "sample_order_sha256", "gate_first_batch"):
                assert x[split][key] == y[split][key]
    first = whole["history"][0]["train"]
    assert first["gate_first_batch"]["multiplier_mean"] == 1
    assert first["first_update"]["gradient_l1_by_stage"]["ts_encoder"] == 0
    assert first["first_update"]["second_backward_gradient_l1_by_stage"]["ts_encoder"] > 0
    with pytest.raises(ValueError, match="overwrite"):
        run("whole")
