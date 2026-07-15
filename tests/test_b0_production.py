from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ebackbone_v3.b0_models import (
    B0_BOUNDED_CPU_BATCH_SIZE,
    B0_BOUNDED_CPU_RSS_LIMIT_BYTES,
    B0FrameClassifier,
    PRODUCTION_MODEL_NAME,
    build_b0_model,
    native_input_batch_bytes,
    trainable_parameter_count,
)
from ebackbone_v3.b0_production import (
    _atomic_torch_save,
    _checkpoint_payload,
    _run_validation_epoch,
    collate_production_b0,
    topk_correct,
    verify_production_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


def _frames(batch_size: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    return torch.rand((batch_size, 2, 480, 640), generator=generator)


def test_b0_model_contract_parameter_count_and_random_initialization() -> None:
    torch.manual_seed(7)
    model = B0FrameClassifier(class_count=100).eval()
    inputs = _frames()
    logits = model(inputs)
    embedding = model.encode(inputs)

    assert logits.shape == (2, 100)
    assert embedding.shape == (2, 64)
    assert trainable_parameter_count(model) == 68_148
    assert [module for module in model.modules() if isinstance(module, nn.Linear)] == [
        model.classifier
    ]
    assert model.classifier.in_features == 64
    assert model.classifier.out_features == 100
    normalization_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.GroupNorm,
        nn.InstanceNorm2d,
        nn.LayerNorm,
    )
    assert not any(isinstance(module, normalization_types) for module in model.modules())

    torch.manual_seed(7)
    same_seed = build_b0_model()
    torch.manual_seed(8)
    changed_seed = build_b0_model()
    assert all(
        torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), same_seed.state_dict().values())
    )
    assert any(
        not torch.equal(left, right)
        for left, right in zip(model.state_dict().values(), changed_seed.state_dict().values())
    )


def test_b0_gradients_reach_encoder_and_classifier_and_step_changes_both() -> None:
    torch.manual_seed(8)
    model = B0FrameClassifier(class_count=100)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    labels = torch.tensor([3, 9], dtype=torch.long)
    encoder_before = model.encoder[0].weight.detach().clone()
    classifier_before = model.classifier.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(_frames()), labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder[0].weight.grad is not None
    assert bool(torch.isfinite(model.encoder[0].weight.grad).all())
    assert bool(torch.count_nonzero(model.encoder[0].weight.grad))
    assert model.classifier.weight.grad is not None
    assert bool(torch.isfinite(model.classifier.weight.grad).all())
    assert bool(torch.count_nonzero(model.classifier.weight.grad))
    optimizer.step()

    assert not torch.equal(encoder_before, model.encoder[0].weight)
    assert not torch.equal(classifier_before, model.classifier.weight)


def test_b0_collate_batches_only_native_event_frames() -> None:
    samples = [
        SimpleNamespace(
            tensors={"event_frame": np.full((2, 480, 640), index, dtype=np.float32)},
            metadata=SimpleNamespace(
                label=index,
                sample_id=f"train/n00000000/sample_{index}.npz",
                split="train",
                source_split="train",
            ),
        )
        for index in range(2)
    ]
    batch = collate_production_b0(samples)  # type: ignore[arg-type]
    assert batch["event_frames"].shape == (2, 2, 480, 640)
    assert batch["event_frames"].dtype == torch.float32
    assert batch["labels"].tolist() == [0, 1]
    assert set(batch) == {
        "event_frames",
        "labels",
        "sample_ids",
        "project_splits",
        "source_splits",
    }


def test_validation_inference_does_not_update_model_state() -> None:
    torch.manual_seed(9)
    model = B0FrameClassifier(class_count=100)
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    loader = [
        {
            "event_frames": _frames(),
            "labels": torch.tensor([1, 2], dtype=torch.long),
            "sample_ids": ["train/n0/a.npz", "train/n0/b.npz"],
            "project_splits": ["validation", "validation"],
            "source_splits": ["train", "train"],
        }
    ]
    metrics = _run_validation_epoch(
        model=model,
        loader=loader,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        use_amp=False,
        epoch=1,
        expected_sample_count=2,
    )
    assert metrics["split_audit"]["project_test_samples_opened"] == 0
    assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_top1_and_top5_metrics() -> None:
    logits = torch.tensor(
        [[9.0, 8.0, 7.0, 6.0, 5.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
    )
    labels = torch.tensor([0, 1])
    assert topk_correct(logits, labels) == {1: 1, 5: 2}


def test_production_checkpoint_strict_round_trip_reproduces_logits(tmp_path: Path) -> None:
    torch.manual_seed(10)
    model = B0FrameClassifier(class_count=100).eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    run_config = {
        "seed": 10,
        "optimizer": {"name": "SGD"},
        "scheduler": {"name": "CosineAnnealingLR"},
        "manifests": {"train": {"sha256": "a"}, "validation": {"sha256": "b"}},
        "renderer_provenance": {"renderer_fingerprint": "c"},
    }
    payload = _checkpoint_payload(
        epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        run_config=run_config,
        history=[],
        best=None,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    _atomic_torch_save(checkpoint_path, payload)
    verification = verify_production_checkpoint(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reloaded = build_b0_model(checkpoint["model"]["name"]).eval()
    incompatible = reloaded.load_state_dict(checkpoint["model_state_dict"], strict=True)
    inputs = _frames(1)
    with torch.no_grad():
        reference_logits = model(inputs)
        reloaded_logits = reloaded(inputs)

    assert payload["model"]["name"] == PRODUCTION_MODEL_NAME
    assert verification["verified"] is True
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert torch.equal(reference_logits, reloaded_logits)


def test_native_resolution_cpu_batch_stays_within_documented_memory_bound() -> None:
    script = """
import json
import resource
import torch
from torch.nn import functional as F
from ebackbone_v3.b0_models import B0FrameClassifier, native_input_batch_bytes
torch.set_num_threads(1)
torch.manual_seed(11)
batch_size = 4
model = B0FrameClassifier()
inputs = torch.zeros((batch_size, 2, 480, 640), dtype=torch.float32)
labels = torch.arange(batch_size, dtype=torch.long)
loss = F.cross_entropy(model(inputs), labels)
loss.backward()
print(json.dumps({
    "batch_size": batch_size,
    "input_bytes": native_input_batch_bytes(batch_size),
    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    measurement = json.loads(result.stdout)
    assert measurement["batch_size"] == B0_BOUNDED_CPU_BATCH_SIZE
    assert measurement["input_bytes"] == native_input_batch_bytes(
        B0_BOUNDED_CPU_BATCH_SIZE
    )
    assert measurement["peak_rss_bytes"] < B0_BOUNDED_CPU_RSS_LIMIT_BYTES
