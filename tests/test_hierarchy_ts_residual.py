from functools import partial

import pytest
import torch

from ebackbone_v3 import hierarchy_ts_residual_training as training
from ebackbone_v3.hierarchy_models import HierarchyV1
from ebackbone_v3.hierarchy_ts_models import HierarchyTSV1
from ebackbone_v3.hierarchy_ts_data import collate
from ebackbone_v3.hierarchy_ts_residual_models import HierarchyTSResidualV2, TSResidualFusion, profile_macs
from tests.test_hierarchy_ts import FakeDataset, sample
from tests.test_v1_training import assert_nested_equal


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_fusion_identity_both_inputs_control_weight_and_additive_capacity():
    torch.manual_seed(4)
    fusion = TSResidualFusion()
    f, t = torch.randn(2, 128, 4, 4), torch.randn(2, 64, 4, 4)
    assert torch.equal(fusion(f, t), f)
    assert torch.equal(fusion(f, t.flip(0)), f)
    w = fusion.confidence(torch.cat((f, t), 1))
    assert w.shape == (2, 1, 4, 4) and torch.all((w >= 0) & (w <= 1))
    assert not torch.equal(w, fusion.confidence(torch.cat((f.flip(0), t), 1)))
    assert not torch.equal(w, fusion.confidence(torch.cat((f, t.flip(0)), 1)))
    with torch.no_grad():
        fusion.residual.bias.fill_(1)
    # A zero hierarchy feature can receive TS-branch content: pure scaling cannot do this.
    assert torch.all(fusion(torch.zeros_like(f), t) > 0)


def test_identical_initial_backbone_encoder_rng_and_two_step_gradients(tmp_path):
    torch.manual_seed(42)
    base = HierarchyV1(height=32, width=32).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    old = HierarchyTSV1(height=32, width=32).eval()
    torch.manual_seed(42)
    model = HierarchyTSResidualV2(height=32, width=32).eval()
    assert torch.equal(rng, torch.get_rng_state())
    assert not hasattr(model, "ts_gate")
    for name, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    for name, value in old.ts_encoder.state_dict().items():
        assert torch.equal(value, model.ts_encoder.state_dict()[name])
    inputs = collate([sample(0), sample(1)], height=32, width=32)["inputs"]
    point_inputs = {k: v for k, v in inputs.items() if k != "time_surface"}
    assert torch.equal(base(point_inputs), model(inputs))
    model.capture_gate_stats = True
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, weight_decay=0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1]))
        loss.backward()
        for name, module in {**dict(model.named_children()),
                             "confidence": model.ts_fusion.confidence,
                             "residual": model.ts_fusion.residual}.items():
            params = list(module.parameters())
            if not params:
                continue
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params), name
            grad = sum(float(p.grad.abs().sum()) for p in params)
            if step == 0 and name in ("ts_encoder", "confidence"):
                assert grad == 0, name
            else:
                assert grad > 0, name
        optimizer.step()
    assert model.last_gate_stats["relative_feature_change"] > 0
    assert not torch.equal(model(inputs), model({**inputs, "time_surface": inputs["time_surface"].flip(0)}))
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    restored = HierarchyTSResidualV2(height=32, width=32).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(model(inputs), restored(inputs))
    individual = torch.cat([model(collate([sample(i)], height=32, width=32)["inputs"]) for i in range(2)])
    torch.testing.assert_close(model(inputs), individual, atol=3e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="exactly"):
        model(point_inputs)
    with pytest.raises(ValueError, match="aligned"):
        model({**inputs, "time_surface": inputs["time_surface"][:1]})


def test_profile_native_shapes_and_cost():
    result = profile_macs(HierarchyTSResidualV2())
    assert result["model_version"] == "hierarchy-ts-content-residual-v2-1"
    assert result["parameters"] == 2745829
    assert result["macs"] == 7635201280
    assert result["shapes_batch1"]["ts_fusion"] == [1, 128, 30, 40]
    assert result["shapes_batch1"]["frame_stage"] == [1, 256, 15, 20]


def test_runner_resume_and_configuration_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "HierarchyDataset", FakeDataset)
    monkeypatch.setattr(training, "HierarchyV1", lambda num_classes=100, height=32, width=32:
                        HierarchyTSResidualV2(num_classes, height, width))
    monkeypatch.setattr(training, "collate", partial(collate, height=32, width=32))
    config = training.TrainConfig(epochs=2, batch_size=2, accumulation_steps=1, device="cpu",
                                  precision="float32", num_workers=0, cpu_threads=1)

    def run(name, **kwargs):
        return training.run_model("hierarchy_ts_residual", config, manifest_dir="unused", dataset_root="unused",
                                  output_dir=tmp_path / name, **kwargs)

    whole = run("whole")
    assert run("resume", stop_after_epoch=1)["status"] == "partial"
    resumed = run("resume", resume=True)
    a, b = (torch.load(tmp_path / n / "checkpoint_last.pt", weights_only=False) for n in ("whole", "resume"))
    for key in ("model", "optimizer", "scheduler"):
        assert_nested_equal(a[key], b[key])
    assert resumed["checkpoint_verification"]["logits_bit_exact"]
    for left, right in zip(whole["history"], resumed["history"]):
        for split in ("train", "validation"):
            for key in ("top1", "loss", "gate_first_batch", "sample_order_sha256"):
                assert left[split][key] == right[split][key]
    assert whole["history"][0]["train"]["gate_first_batch"]["relative_feature_change"] == 0
    with pytest.raises(ValueError, match="overwrite"):
        run("whole")
