import json
from contextlib import contextmanager, nullcontext

import pytest
import torch

from ebackbone_v3.hierarchy_clip_preflight import assert_training_gate, state_fingerprint
from ebackbone_v3.hierarchy_clip_train import config_fingerprint, VARIANTS
from ebackbone_v3.hierarchy_clip_checkpoint import file_sha256


def test_state_fingerprint_checks_values_without_tensor_storage_identity():
    state = {"model": torch.tensor([1., 2.]), "sampler": {"seed": 7, "epoch": 6}}
    same = {"sampler": {"epoch": 6, "seed": 7}, "model": state["model"].clone()}
    assert state_fingerprint(state) == state_fingerprint(same)
    same["model"][0] += 1
    assert state_fingerprint(state) != state_fingerprint(same)


def test_training_gate_requires_all_variants_matching_config_and_reports(tmp_path):
    config = {"training_gate_path": str(tmp_path / "gate.json"), "training": {"epochs": 50}}
    reports = {}
    for variant in VARIANTS:
        path = tmp_path / f"{variant}.json"
        path.write_text(json.dumps({"status": "PASS", "config_sha256": config_fingerprint(config)}))
        reports[variant] = {"path": str(path), "sha256": file_sha256(path)}
    gate = {"status": "PASS", "all_variants_passed": True, "variants": list(VARIANTS),
            "config_sha256": config_fingerprint(config), "source_sha256": {}, "reports": reports}
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(gate))
    assert assert_training_gate(config)["status"] == "PASS"
    with pytest.raises(ValueError, match="matching"):
        assert_training_gate({**config, "training": {"epochs": 40}})
    gate["variants"].pop()
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError, match="all three"):
        assert_training_gate(config)
    gate["variants"] = list(VARIANTS)
    path.write_text(json.dumps(gate))
    (tmp_path / f"{VARIANTS[0]}.json").write_text('{"status":"FAIL"}')
    with pytest.raises(ValueError, match="evidence changed"):
        assert_training_gate(config)


def test_gradient_reference_keeps_backward_outside_autocast(monkeypatch):
    from ebackbone_v3 import hierarchy_clip_preflight as preflight
    from tests.test_hierarchy_clip_temporal import _tiny_sample
    active = [False]

    @contextmanager
    def tracked_amp(device):
        active[0] = True
        try:
            yield
        finally:
            active[0] = False

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(4, 3)

        def forward(self, inputs):
            logits = self.classifier(torch.ones(len(inputs["window_mask"]), 4))
            def check_backward(gradient):
                assert not active[0], "reference backward must use the same AMP scope as training"
                return gradient
            logits.register_hook(check_backward)
            return logits

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = Model()

        def forward(self, inputs):
            return self.module(inputs)

        def no_sync(self):
            return nullcontext()

    monkeypatch.setattr(preflight.train, "amp_context", tracked_amp)
    monkeypatch.setattr(preflight.dist, "all_reduce", lambda gradient: gradient.mul_(2))
    result = preflight.verify_gradient_average(Wrapped(), [_tiny_sample(0), _tiny_sample(1)], "cpu")
    assert result["explicit_average_matches"]
