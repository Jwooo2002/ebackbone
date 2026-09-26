from functools import partial

import pytest
import torch

from ebackbone_v3 import hierarchy_ts_confidence_training as training
from ebackbone_v3.hierarchy_models import HierarchyV1
from ebackbone_v3.hierarchy_ts_confidence_models import ConfidenceControl, HierarchyTSConfidenceV3, profile_macs
from ebackbone_v3.hierarchy_ts_data import collate
from tests.test_hierarchy_ts import FakeDataset, sample
from tests.test_v1_training import assert_nested_equal


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_confidence_monotone_detached_and_threshold_trainable():
    control = ConfidenceControl()
    logits = torch.tensor([[0., 0.], [1., 0.], [8., 0.]], requires_grad=True)
    strength = control(logits).flatten()
    assert 1 > strength[0] > strength[1] > strength[2] > 0
    strength.sum().backward()
    assert logits.grad is None
    assert control.threshold_logit.grad > 0
    assert control(torch.zeros(2, 100)).shape == (2, 1, 1, 1)


def test_identity_shared_head_gradients_and_checkpoint(tmp_path):
    torch.manual_seed(42)
    base = HierarchyV1(height=32, width=32).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    model = HierarchyTSConfidenceV3(height=32, width=32).eval()
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.requires_grad for p in model.parameters())
    for name, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[name])
    inputs = collate([sample(0), sample(1)], height=32, width=32)["inputs"]
    plain = {k: v for k, v in inputs.items() if k != "time_surface"}
    assert torch.equal(base(plain), model(inputs))
    calls = []
    handle = model.classifier.register_forward_hook(lambda *_: calls.append(torch.is_grad_enabled()))
    model(inputs)
    handle.remove()
    assert calls == [False, True]
    model.capture_gate_stats = True
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, weight_decay=0)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(inputs), torch.tensor([0, 1])).backward()
        for name, module in {**dict(model.named_children()), "spatial_weight": model.ts_fusion.confidence}.items():
            params = list(module.parameters())
            if not params:
                continue
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params), name
            grad = sum(float(p.grad.abs().sum()) for p in params)
            if step == 0 and name in ("ts_encoder", "spatial_weight", "confidence_control"):
                assert grad == 0, name
            else:
                assert grad > 0, name
        optimizer.step()
    assert model.last_gate_stats["relative_feature_change"] > 0
    assert not torch.equal(model(inputs), model({**inputs, "time_surface": inputs["time_surface"].flip(0)}))
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    restored = HierarchyTSConfidenceV3(height=32, width=32).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert torch.equal(model(inputs), restored(inputs))
    with pytest.raises(ValueError, match="exactly"):
        model(plain)


def test_profile_counts_shared_tail_twice():
    from ebackbone_v3.hierarchy_ts_residual_models import HierarchyTSResidualV2, profile_macs as v2_profile
    old = v2_profile(HierarchyTSResidualV2())
    new = profile_macs(HierarchyTSConfidenceV3())
    assert new["parameters"] == old["parameters"] + 1
    assert new["macs"] == old["macs"] + old["macs_by_stage"]["frame_stage"] + old["macs_by_stage"]["classifier"]
    assert new["shapes_batch1"]["confidence_control"] == [1, 1, 1, 1]
    assert new["shapes_batch1"]["frame_stage"] == [1, 256, 15, 20]


def test_bfloat16_shared_tail_keeps_trainable_gradients():
    model = HierarchyTSConfidenceV3(height=32, width=32).train()
    inputs = collate([sample(0), sample(1)], height=32, width=32)["inputs"]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor([0, 1]))
    assert logits.dtype == torch.bfloat16
    loss.backward()
    for name, module in (("frame_stage", model.frame_stage), ("classifier", model.classifier)):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()), name
        assert sum(float(p.grad.abs().sum()) for p in module.parameters()) > 0


def test_runner_exact_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(training, "HierarchyDataset", FakeDataset)
    monkeypatch.setattr(training, "HierarchyV1", lambda num_classes=100, height=32, width=32:
                        HierarchyTSConfidenceV3(num_classes, height, width))
    monkeypatch.setattr(training, "collate", partial(collate, height=32, width=32))
    config = training.TrainConfig(epochs=2, batch_size=2, accumulation_steps=1, device="cpu",
                                  precision="float32", num_workers=0, cpu_threads=1)
    def run(name, **kwargs):
        return training.run_model("hierarchy_ts_confidence", config, manifest_dir="unused", dataset_root="unused",
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
    with pytest.raises(ValueError, match="overwrite"):
        run("whole")
