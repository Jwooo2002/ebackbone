from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ebackbone_v3.b0_models import (
    PRODUCTION_MODEL_NAME,
    ProductionB0ResNet18,
    trainable_parameter_count,
)
from ebackbone_v3.b0_production import (
    _atomic_torch_save,
    _checkpoint_payload,
    topk_correct,
    verify_production_checkpoint,
)


def test_production_resnet18_contract_and_parameter_count() -> None:
    torch.manual_seed(7)
    model = ProductionB0ResNet18(class_count=100).eval()
    inputs = torch.zeros((1, 2, 480, 640), dtype=torch.float32)
    logits = model(inputs)
    assert logits.shape == (1, 100)
    assert trainable_parameter_count(model) == 11_224_676
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    assert linear_layers == [model.classifier]
    assert model.classifier.in_features == 512
    assert model.classifier.out_features == 100


def test_production_resnet18_has_finite_cross_entropy_and_optimizer_step() -> None:
    torch.manual_seed(8)
    model = ProductionB0ResNet18(class_count=100).eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    inputs = torch.zeros((1, 2, 480, 640), dtype=torch.float32)
    labels = torch.tensor([3], dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(inputs), labels)
    loss.backward()
    before = model.classifier.bias.detach().clone()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.classifier.bias)


def test_top1_and_top5_metrics() -> None:
    logits = torch.tensor(
        [[9.0, 8.0, 7.0, 6.0, 5.0, 4.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
    )
    labels = torch.tensor([0, 1])
    assert topk_correct(logits, labels) == {1: 1, 5: 2}


def test_production_checkpoint_strict_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(9)
    model = ProductionB0ResNet18(class_count=100)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    run_config = {
        "seed": 9,
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
    assert payload["model"]["name"] == PRODUCTION_MODEL_NAME
    assert payload["seed"] == 9
    assert payload["manifests"] == run_config["manifests"]
    assert payload["renderer_provenance"] == run_config["renderer_provenance"]
    assert verification["verified"] is True
    assert verification["epoch"] == 1
    assert verification["strict_load_missing_keys"] == []
    assert verification["strict_load_unexpected_keys"] == []
