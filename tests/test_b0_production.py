from __future__ import annotations

import inspect
import json
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import ebackbone_v3.b0_models as b0_models
from ebackbone_v3.b0_models import (
    B0_BOUNDED_CPU_BATCH_SIZE,
    B0_BOUNDED_CPU_RSS_LIMIT_BYTES,
    B0_EMBEDDING_DIM,
    CompactDebugB0FrameClassifier,
    DEBUG_MODEL_NAME,
    PRODUCTION_MODEL_NAME,
    ProductionB0ResNet18,
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
from ebackbone_v3.errors import TrainingError


ROOT = Path(__file__).resolve().parents[1]


def _frames(batch_size: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    return torch.rand((batch_size, 2, 480, 640), generator=generator)


def test_production_b0_is_standard_two_channel_resnet18_contract() -> None:
    torch.manual_seed(7)
    model = build_b0_model().eval()

    assert isinstance(model, ProductionB0ResNet18)
    assert PRODUCTION_MODEL_NAME == "resnet18"
    assert model.conv1.in_channels == 2
    assert model.conv1.out_channels == 64
    assert model.conv1.kernel_size == (7, 7)
    assert model.conv1.stride == (2, 2)
    assert model.conv1.padding == (3, 3)
    assert model.conv1.bias is None
    assert isinstance(model.bn1, nn.BatchNorm2d)
    assert isinstance(model.maxpool, nn.MaxPool2d)
    assert model.maxpool.kernel_size == 3
    assert model.maxpool.stride == 2
    assert model.maxpool.padding == 1
    assert [len(stage) for stage in (model.layer1, model.layer2, model.layer3, model.layer4)] == [
        2,
        2,
        2,
        2,
    ]
    assert sum(isinstance(module, nn.BatchNorm2d) for module in model.modules()) == 20
    assert not any(isinstance(module, nn.GroupNorm) for module in model.modules())
    assert isinstance(model.avgpool, nn.AdaptiveAvgPool2d)
    assert model.avgpool.output_size == (1, 1)
    assert trainable_parameter_count(model) == 11_224_676


def test_production_b0_output_embedding_and_single_linear_classifier() -> None:
    model = ProductionB0ResNet18().eval()
    inputs = _frames(batch_size=2)
    with torch.no_grad():
        embedding = model.encode(inputs)
        logits = model(inputs)

    assert B0_EMBEDDING_DIM == 512
    assert embedding.shape == (2, 512)
    assert logits.shape == (2, 100)
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    assert linear_layers == [model.classifier]
    assert model.classifier.in_features == 512
    assert model.classifier.out_features == 100
    with pytest.raises(ValueError, match="class_count must be exactly 100"):
        ProductionB0ResNet18(class_count=99)


@pytest.mark.parametrize("channels", [1, 3])
def test_production_b0_rejects_non_two_channel_inputs(channels: int) -> None:
    model = ProductionB0ResNet18().eval()
    with pytest.raises(ValueError, match=r"\[B, 2, 480, 640\]"):
        model(torch.zeros((1, channels, 480, 640), dtype=torch.float32))


def test_model_construction_is_random_only_and_cannot_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_network(*args: object, **kwargs: object) -> object:
        raise AssertionError("model construction attempted network access")

    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden_network)
    signature = inspect.signature(ProductionB0ResNet18)
    assert set(signature.parameters) == {"class_count"}
    source = inspect.getsource(b0_models)
    assert "load_state_dict_from_url(" not in source
    assert "torch.hub.load(" not in source
    assert "torchvision.models.resnet18(" not in source

    torch.manual_seed(21)
    first = build_b0_model()
    torch.manual_seed(21)
    same_seed = build_b0_model()
    torch.manual_seed(22)
    changed_seed = build_b0_model()
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.state_dict().values(), same_seed.state_dict().values())
    )
    assert any(
        not torch.equal(left, right)
        for left, right in zip(first.state_dict().values(), changed_seed.state_dict().values())
    )


def test_compact_debug_is_explicit_and_not_the_production_default() -> None:
    production = build_b0_model()
    debug = build_b0_model(DEBUG_MODEL_NAME)

    assert isinstance(production, ProductionB0ResNet18)
    assert isinstance(debug, CompactDebugB0FrameClassifier)
    assert DEBUG_MODEL_NAME == "compact_debug"
    assert trainable_parameter_count(debug) == 68_148
    assert debug.classifier.in_features == 64


def test_b0_gradients_reach_resnet_body_and_classifier() -> None:
    torch.manual_seed(8)
    model = ProductionB0ResNet18(class_count=100)
    labels = torch.tensor([3], dtype=torch.long)
    loss = F.cross_entropy(model(_frames()), labels)
    loss.backward()

    assert torch.isfinite(loss)
    for parameter in (model.conv1.weight, model.layer4[1].conv2.weight, model.classifier.weight):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert bool(torch.count_nonzero(parameter.grad))


def test_b0_collate_requests_only_native_event_frames() -> None:
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
        "archive_decode_seconds",
        "frame_stage_seconds",
    }

    invalid = SimpleNamespace(
        tensors={
            "event_frame": np.zeros((2, 480, 640), dtype=np.float32),
            "voxel_grid": np.zeros((2, 5, 480, 640), dtype=np.float32),
        },
        metadata=samples[0].metadata,
    )
    with pytest.raises(TrainingError, match="non-frame representation"):
        collate_production_b0([invalid])  # type: ignore[list-item]


def test_validation_inference_does_not_update_model_state() -> None:
    torch.manual_seed(9)
    model = ProductionB0ResNet18(class_count=100)
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    loader = [
        {
            "event_frames": _frames(),
            "labels": torch.tensor([1], dtype=torch.long),
            "sample_ids": ["train/n0/a.npz"],
            "project_splits": ["validation"],
            "source_splits": ["train"],
        }
    ]
    metrics = _run_validation_epoch(
        model=model,
        loader=loader,  # type: ignore[arg-type]
        device=torch.device("cpu"),
        use_amp=False,
        epoch=1,
        expected_sample_count=1,
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
    model = ProductionB0ResNet18(class_count=100).eval()
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
    inputs = _frames()
    with torch.no_grad():
        reference_logits = model(inputs)
        reloaded_logits = reloaded(inputs)

    assert payload["model"]["name"] == PRODUCTION_MODEL_NAME
    assert verification["verified"] is True
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert torch.equal(reference_logits, reloaded_logits)


def test_native_resolution_cpu_batch_one_forward_backward_is_bounded() -> None:
    script = """
import json
import resource
import torch
from torch.nn import functional as F
from ebackbone_v3.b0_models import ProductionB0ResNet18, native_input_batch_bytes
torch.set_num_threads(1)
torch.manual_seed(11)
batch_size = 1
model = ProductionB0ResNet18()
inputs = torch.rand((batch_size, 2, 480, 640), dtype=torch.float32)
labels = torch.tensor([7], dtype=torch.long)
loss = F.cross_entropy(model(inputs), labels)
loss.backward()
print(json.dumps({
    "batch_size": batch_size,
    "input_bytes": native_input_batch_bytes(batch_size),
    "loss": float(loss.detach()),
    "stem_gradient_finite": bool(torch.isfinite(model.conv1.weight.grad).all()),
    "classifier_gradient_finite": bool(torch.isfinite(model.classifier.weight.grad).all()),
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
    assert measurement["batch_size"] == B0_BOUNDED_CPU_BATCH_SIZE == 1
    assert measurement["input_bytes"] == native_input_batch_bytes(1)
    assert measurement["loss"] > 0.0
    assert measurement["stem_gradient_finite"] is True
    assert measurement["classifier_gradient_finite"] is True
    assert measurement["peak_rss_bytes"] < B0_BOUNDED_CPU_RSS_LIMIT_BYTES
