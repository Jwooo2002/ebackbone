from __future__ import annotations

import pytest
import torch

from ebackbone_v3.smoke import run_synthetic_smoke


@pytest.mark.parametrize(
    ("baseline", "input_names"),
    [
        ("b0", ["event_frame"]),
        ("b1", ["event_frame", "voxel_grid", "time_surface"]),
    ],
)
def test_synthetic_smoke_forward_loss_backward_and_step(
    baseline: str,
    input_names: list[str],
) -> None:
    report = run_synthetic_smoke(baseline)
    assert report["status"] == "PASS"
    assert report["contract_scope"] == "synthetic_smoke_only"
    assert report["real_data_contract_confirmed"] is False
    assert report["temporal_alignment"] == "not_applicable_to_synthetic_fixture"
    assert report["device"] == "cpu"
    assert report["input_names"] == input_names
    assert report["objective"] == "cross_entropy"
    assert report["initialization"] == "random"
    assert report["classification_head"] == "linear"
    assert report["forward"]["logits_shape"] == [2, 3]
    assert report["forward"]["logits_finite"] is True
    assert report["loss"] > 0
    assert report["backward"]["completed"] is True
    assert report["backward"]["parameter_gradients_finite"] is True
    assert set(report["backward"]["input_gradients"]) == set(input_names)
    assert all(
        gradient["present"] and gradient["finite"] and gradient["nonzero"]
        for gradient in report["backward"]["input_gradients"].values()
    )
    assert report["optimizer_step"]["parameter_changed"] is True
    assert report["optimizer_step"]["changed_parameter_count"] > 0
    assert all(value == "TBD" for value in report["production_decisions"].values())


@pytest.mark.parametrize("baseline", ["b0", "b1"])
def test_synthetic_smoke_is_deterministic(baseline: str) -> None:
    assert run_synthetic_smoke(baseline) == run_synthetic_smoke(baseline)


def test_synthetic_smoke_restores_cpu_rng_and_deterministic_state() -> None:
    original_enabled = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_rng_state = torch.random.get_rng_state().clone()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        state_before = torch.random.get_rng_state().clone()
        run_synthetic_smoke("b0")
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.is_deterministic_algorithms_warn_only_enabled() is True
        assert torch.equal(torch.random.get_rng_state(), state_before)
    finally:
        torch.random.set_rng_state(original_rng_state)
        torch.use_deterministic_algorithms(
            original_enabled,
            warn_only=original_warn_only,
        )
