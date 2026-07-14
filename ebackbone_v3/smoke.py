"""Synthetic-only execution harness for classifier plumbing.

Nothing in this module is a production B0 or B1 architecture. In particular,
the private flatten/project/concatenate operations exist only to prove that a
CPU forward pass, cross-entropy loss, backward pass, and optimizer step work.
"""

from __future__ import annotations

import platform
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from ebackbone_v3.errors import SmokeError


SMOKE_SCHEMA_VERSION = 1
SMOKE_SEED = 20260714
SMOKE_BATCH_SIZE = 2
SMOKE_CLASS_COUNT = 3
SMOKE_PROJECTION_SIZE = 5
SMOKE_INPUT_SHAPES: dict[str, tuple[int, ...]] = {
    "event_frame": (SMOKE_BATCH_SIZE, 2, 3),
    "voxel_grid": (SMOKE_BATCH_SIZE, 3, 2),
    "time_surface": (SMOKE_BATCH_SIZE, 1, 2, 3),
}


class _SyntheticAutogradHarness(nn.Module):
    """Private test double; deliberately not exported as a baseline model."""

    def __init__(self, input_shapes: Mapping[str, tuple[int, ...]]) -> None:
        super().__init__()
        self.input_names = tuple(input_shapes)
        self.toy_projections = nn.ModuleDict(
            {
                name: nn.Linear(_feature_count(shape), SMOKE_PROJECTION_SIZE)
                for name, shape in input_shapes.items()
            }
        )
        self.linear_classification_head = nn.Linear(
            SMOKE_PROJECTION_SIZE * len(input_shapes),
            SMOKE_CLASS_COUNT,
        )

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        if tuple(inputs) != self.input_names:
            raise SmokeError(
                f"synthetic input names must be {self.input_names}; got {tuple(inputs)}"
            )
        toy_embeddings = [
            torch.tanh(self.toy_projections[name](inputs[name].flatten(start_dim=1)))
            for name in self.input_names
        ]
        fixture_only_combination = torch.cat(toy_embeddings, dim=1)
        return self.linear_classification_head(fixture_only_combination)


def run_synthetic_smoke(baseline: str) -> dict[str, Any]:
    """Exercise a synthetic B0/B1 input contract on CPU and report proof."""

    normalized_baseline = baseline.lower()
    if normalized_baseline not in {"b0", "b1"}:
        raise SmokeError(f"unsupported baseline {baseline!r}; expected 'b0' or 'b1'")
    input_names = (
        ("event_frame",)
        if normalized_baseline == "b0"
        else ("event_frame", "voxel_grid", "time_surface")
    )
    input_shapes = {name: SMOKE_INPUT_SHAPES[name] for name in input_names}

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(SMOKE_SEED)
            inputs = _make_inputs(input_shapes)
            labels = torch.tensor([0, 2], dtype=torch.long, device="cpu")
            harness = _SyntheticAutogradHarness(input_shapes).to(device="cpu")
            optimizer = torch.optim.SGD(harness.parameters(), lr=0.01)
            optimizer.zero_grad(set_to_none=True)
            logits = harness(inputs)
            loss = nn.functional.cross_entropy(logits, labels)
            if logits.shape != (SMOKE_BATCH_SIZE, SMOKE_CLASS_COUNT):
                raise SmokeError(f"unexpected classifier output shape: {tuple(logits.shape)}")
            if not bool(torch.isfinite(loss).item()):
                raise SmokeError("cross-entropy loss is not finite")
            loss.backward()

            input_gradients = {name: _gradient_report(tensor.grad) for name, tensor in inputs.items()}
            parameter_gradients = {
                name: _gradient_report(parameter.grad)
                for name, parameter in harness.named_parameters()
            }
            if not all(report["finite"] and report["nonzero"] for report in input_gradients.values()):
                raise SmokeError("a synthetic input branch did not receive a finite nonzero gradient")
            if not all(report["finite"] for report in parameter_gradients.values()):
                raise SmokeError("a trainable synthetic harness parameter has a missing/non-finite gradient")

            parameters_before_step = {
                name: parameter.detach().clone()
                for name, parameter in harness.named_parameters()
            }
            optimizer.step()
            changed_parameters = [
                name
                for name, parameter in harness.named_parameters()
                if not torch.equal(parameters_before_step[name], parameter.detach())
            ]
            if not changed_parameters:
                raise SmokeError("optimizer step did not change any trainable parameter")

            report = {
                "schema_version": SMOKE_SCHEMA_VERSION,
                "status": "PASS",
                "command": "smoke",
                "baseline": normalized_baseline,
                "contract_scope": "synthetic_smoke_only",
                "real_data_contract_confirmed": False,
                "temporal_alignment": "not_applicable_to_synthetic_fixture",
                "device": "cpu",
                "seed": SMOKE_SEED,
                "deterministic_algorithms": True,
                "initialization": "random",
                "objective": "cross_entropy",
                "classification_head": "linear",
                "input_names": list(input_names),
                "synthetic_assumptions": {
                    "batch_size": SMOKE_BATCH_SIZE,
                    "class_count": SMOKE_CLASS_COUNT,
                    "labels": labels.tolist(),
                    "input_shapes": {name: list(shape) for name, shape in input_shapes.items()},
                    "input_semantics": (
                        "opaque tiny tensors for execution testing; shapes, channels, and values are not "
                        "real event-representation facts"
                    ),
                    "toy_projection": (
                        "flatten plus independent linear projection, used only inside this smoke harness"
                    ),
                    "toy_combination": (
                        "single toy branch for b0"
                        if normalized_baseline == "b0"
                        else "concatenated toy projections for b1 autograd coverage only"
                    ),
                },
                "production_decisions": {
                    "event_frame_accumulation_or_sequence": "TBD",
                    "fusion": "TBD",
                    "encoder_sharing": "TBD",
                    "pooling": "TBD",
                    "normalization": "TBD",
                    "voxel_bins": "TBD",
                    "time_surface_definition": "TBD",
                },
                "forward": {"logits_shape": list(logits.shape), "logits_finite": True},
                "loss": float(loss.detach().item()),
                "backward": {
                    "completed": True,
                    "input_gradients": input_gradients,
                    "parameter_gradients_finite": all(
                        item["finite"] for item in parameter_gradients.values()
                    ),
                    "all_parameter_gradients_nonzero": all(
                        item["nonzero"] for item in parameter_gradients.values()
                    ),
                },
                "optimizer_step": {
                    "optimizer": "SGD",
                    "learning_rate": 0.01,
                    "parameter_changed": True,
                    "changed_parameter_count": len(changed_parameters),
                },
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in harness.parameters() if parameter.requires_grad
                ),
                "runtime": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                },
            }
    finally:
        torch.use_deterministic_algorithms(
            previous_deterministic,
            warn_only=previous_warn_only,
        )
    return report


def _make_inputs(input_shapes: Mapping[str, tuple[int, ...]]) -> dict[str, Tensor]:
    inputs: dict[str, Tensor] = {}
    for index, (name, shape) in enumerate(input_shapes.items()):
        element_count = _total_count(shape)
        values = torch.linspace(-1.0, 1.0, steps=element_count, dtype=torch.float32)
        inputs[name] = (values.reshape(shape) + index * 0.125).requires_grad_()
    return inputs


def _feature_count(shape: tuple[int, ...]) -> int:
    if len(shape) < 2 or shape[0] != SMOKE_BATCH_SIZE:
        raise SmokeError(f"invalid synthetic shape: {shape}")
    return _total_count(shape[1:])


def _total_count(shape: tuple[int, ...]) -> int:
    count = 1
    for size in shape:
        if size <= 0:
            raise SmokeError(f"synthetic dimensions must be positive: {shape}")
        count *= size
    return count


def _gradient_report(gradient: Tensor | None) -> dict[str, bool]:
    if gradient is None:
        return {"present": False, "finite": False, "nonzero": False}
    return {
        "present": True,
        "finite": bool(torch.isfinite(gradient).all().item()),
        "nonzero": bool(torch.count_nonzero(gradient).item() > 0),
    }


__all__ = ["run_synthetic_smoke"]
