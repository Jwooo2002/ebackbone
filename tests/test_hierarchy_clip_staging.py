import copy

import pytest
import torch
from torch import nn

from ebackbone_v3.hierarchy_models import HierarchyV1
from ebackbone_v3.hierarchy_clip_staging import (
    configure_stage, gradient_report, load_comparison_checkpoint,
    load_hierarchy_backbone, save_comparison_checkpoint, set_staged_train,
)


class StageFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.hierarchy = HierarchyV1(3, 32, 32)
        self.visual = nn.Module()
        self.visual.transformer = nn.Module()
        self.visual.transformer.resblocks = nn.Sequential(*(nn.Linear(4, 4) for _ in range(3)))
        self.visual.ln_post = nn.LayerNorm(4)
        self.temporal = nn.Linear(4, 4)
        self.adapter = nn.Linear(4, 4)
        self.projection = nn.Linear(4, 3)
        self.text_encoder = nn.Linear(4, 4)

    def forward(self, x):
        return self.projection(self.temporal(self.visual.transformer.resblocks(self.adapter(x))))


def test_selective_policy_optimizer_groups_and_frozen_gradients():
    m = StageFixture()
    groups = configure_stage(m, "connectors")
    assert len(groups) == 1
    loss = m(torch.randn(2, 4)).square().sum()
    loss.backward()
    assert gradient_report(m)["adapter"]["gradient_l1"] > 0
    assert all(p.grad is None for p in m.visual.parameters())
    groups = configure_stage(m, "selective", learning_rate=0.001)
    assert [g["lr"] for g in groups] == [0.001, 0.0001]
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert any(n.startswith("encoder.hierarchy.frame_stage") for n in names)
    assert any(n.startswith("encoder.hierarchy.temporal_collapse") for n in names)
    assert not any(n.startswith("encoder.hierarchy.point") for n in names)
    assert not any(n.startswith("visual.transformer.resblocks.0") for n in names)
    assert not any(n.startswith("text_encoder") for n in names)
    grouped = [id(p) for g in groups for p in g["params"]]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(p) for p in m.parameters() if p.requires_grad}
    assert not m.text_encoder.training
    assert not m.encoder.hierarchy.point.training


def test_stage_roundtrip_includes_optimizer_and_rejects_prompt_mismatch(tmp_path):
    m = StageFixture()
    optimizer = torch.optim.AdamW(configure_stage(m, "connectors"))
    x = torch.randn(2, 4)
    m(x).square().sum().backward()
    optimizer.step()
    identity = {"class_names": ["a", "b", "c"], "split_hash": "fixed"}
    path = tmp_path / "bounded.pt"
    save_comparison_checkpoint(path, m, optimizer, identity)
    restored = StageFixture()
    opt2 = torch.optim.AdamW(configure_stage(restored, "connectors"))
    load_comparison_checkpoint(path, restored, opt2, identity)
    torch.testing.assert_close(m(x), restored(x), rtol=0, atol=0)
    assert len(optimizer.state) == len(opt2.state)
    with pytest.raises(ValueError, match="identity"):
        load_comparison_checkpoint(path, restored, opt2, {**identity, "class_names": ["c", "b", "a"]})
    with pytest.raises(FileExistsError):
        save_comparison_checkpoint(path, m, optimizer, identity)


def test_backbone_loading_rejects_partial_and_wrong_variant(tmp_path):
    source = HierarchyV1(100, 32, 32)
    target = HierarchyV1(3, 32, 32)
    classifier = copy.deepcopy(target.classifier.state_dict())
    state = {k: v for k, v in source.state_dict().items() if not k.startswith("classifier.")}
    path = tmp_path / "export.pt"
    torch.save({"variant": "hierarchy", "backbone": state, "epoch": 50}, path)
    evidence = load_hierarchy_backbone(target, path)
    assert evidence["strict_backbone"]
    for k, v in classifier.items():
        torch.testing.assert_close(target.classifier.state_dict()[k], v, rtol=0, atol=0)
    for k, v in state.items():
        torch.testing.assert_close(target.state_dict()[k], v, rtol=0, atol=0)
    torch.save({"variant": "hierarchy_ts_residual", "backbone": state}, path)
    with pytest.raises(ValueError, match="hierarchy-only"):
        load_hierarchy_backbone(target, path)
    state.pop(next(iter(state)))
    torch.save({"variant": "hierarchy", "backbone": state}, path)
    with pytest.raises(ValueError, match="keys mismatch"):
        load_hierarchy_backbone(target, path)
