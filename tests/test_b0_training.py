from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from ebackbone_v3.b0_training import (
    CompactDebugB0FrameClassifier,
    save_checkpoint,
    select_tiny_subset,
    train_one_optimizer_step,
    verify_checkpoint_round_trip,
)


def _frames(batch_size: int = 2) -> torch.Tensor:
    return torch.linspace(
        0.0,
        1.0,
        steps=batch_size * 2 * 480 * 640,
        dtype=torch.float32,
    ).reshape(batch_size, 2, 480, 640)


def test_b0_model_output_shape() -> None:
    model = CompactDebugB0FrameClassifier(class_count=100)
    logits = model(_frames())
    assert logits.shape == (2, 100)


def test_b0_one_optimizer_step_has_finite_loss_gradients_and_parameter_update() -> None:
    torch.manual_seed(3)
    model = CompactDebugB0FrameClassifier(class_count=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    labels = torch.tensor([1, 3], dtype=torch.long)
    result = train_one_optimizer_step(model, optimizer, _frames(), labels)
    assert result["loss"] > 0.0
    assert result["gradients_finite"] is True
    assert result["parameter_updated"] is True
    assert result["parameter_update_l2"] > 0.0
    assert torch.isfinite(F.cross_entropy(model(_frames()), labels))


def test_b0_checkpoint_round_trip_is_strict_and_logit_exact(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = CompactDebugB0FrameClassifier(class_count=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    inputs = _frames()
    labels = torch.tensor([0, 2], dtype=torch.long)
    train_one_optimizer_step(model, optimizer, inputs, labels)
    selection = select_tiny_subset([f"train/n00000000/sample_{index}.npz" for index in range(16)], subset_size=16, seed=9)
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        selection=selection,
        epochs=[],
    )
    verification = verify_checkpoint_round_trip(
        checkpoint,
        reference_model=model,
        reference_inputs=inputs,
        device=torch.device("cpu"),
    )
    assert verification["verified"] is True
    assert verification["strict_load_missing_keys"] == []
    assert verification["strict_load_unexpected_keys"] == []
    assert verification["logits_max_abs_difference"] == 0.0


def test_tiny_subset_selection_is_deterministic_and_label_independent() -> None:
    sample_ids = [f"train/n00000000/sample_{index:03d}.npz" for index in range(40)]
    first = select_tiny_subset(sample_ids, subset_size=16, seed=20260715)
    second = select_tiny_subset(sample_ids, subset_size=16, seed=20260715)
    changed_seed = select_tiny_subset(sample_ids, subset_size=16, seed=20260716)
    assert first == second
    assert first.sample_ids != changed_seed.sample_ids
    assert len(first.sample_ids) == 16
    assert "labels are not read" in first.selection_rule


def test_fixed_native_resolution_tiny_subset_substantially_reduces_loss() -> None:
    torch.manual_seed(12)
    frames = torch.zeros((4, 2, 480, 640), dtype=torch.float32)
    frames[0, 0].fill_(1.0)
    frames[1, 1].fill_(1.0)
    frames[2, 0].fill_(0.75)
    frames[3, 1].fill_(0.75)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    model = CompactDebugB0FrameClassifier(class_count=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    with torch.no_grad():
        initial_loss = float(F.cross_entropy(model(frames), labels).item())
    for _ in range(24):
        train_one_optimizer_step(model, optimizer, frames, labels)
    with torch.no_grad():
        final_loss = float(F.cross_entropy(model(frames), labels).item())

    assert final_loss < initial_loss * 0.25
