"""Exact next-update resume/transfer checks on small CPU models only."""
import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from ebackbone_v3 import hierarchy_clip_checkpoint as checkpoint
from ebackbone_v3.hierarchy_clip_staging import configure_stage


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class FixtureComparison(nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.hierarchy = nn.Module()
        self.encoder.hierarchy.point = nn.Linear(4, 4)
        self.encoder.hierarchy.temporal_collapse = nn.Linear(4, 4)
        self.encoder.hierarchy.frame_stage = nn.Linear(4, 4)
        self.encoder.hierarchy.classifier = nn.Linear(4, 100)  # unused pretrained CE head
        self.temporal = nn.Sequential(nn.Linear(4, 4), nn.Dropout(.25))
        self.classifier = nn.Linear(4, classes)

    def forward_features(self, x):
        h = self.encoder.hierarchy
        return self.temporal(h.frame_stage(h.temporal_collapse(h.point(x))))

    def forward(self, x):
        return self.classifier(self.forward_features(x))


def identity():
    return {"world_size": 1, "variant": "fixture", "config": {"epochs": 50, "seed": 42},
            "train_manifest_sha256": "train-fixed", "validation_manifest_sha256": "val-fixed",
            "class_names": ["a", "b", "c"], "prompts": ["a photo of a a.", "a photo of a b.", "a photo of a c."],
            "source_sha256": {"source.py": "immutable"}, "hierarchy_checkpoint_sha256": "hierarchy",
            "clip_checkpoint_sha256": "clip"}


def create(stage="connectors"):
    model = FixtureComparison()
    optimizer = torch.optim.AdamW(configure_stage(model, stage, learning_rate=.001), weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    return model, optimizer, scheduler


def update(model, optimizer, scheduler=None):
    # Every global generator affects the next update, including dropout.
    x = torch.randn(3, 4) * np.random.uniform(.5, 1.5) + random.random()
    optimizer.zero_grad(set_to_none=True)
    loss = model(x).square().mean()
    loss.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss.detach(), gradients


def equal_tree(first, second):
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            equal_tree(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            equal_tree(a, b)
    else:
        assert first == second


def save(tmp_path, model, optimizer, scheduler, *, epoch=5, is_best=True):
    return checkpoint.save_training_checkpoint(tmp_path, model, optimizer, scheduler,
        identity=identity(), epoch=epoch, global_step=5, stage=model._fine_tuning_stage,
        sampler_state={"seed": 42, "epoch": epoch + 1, "offset": 0, "world_size": 1},
        rng_by_rank=[checkpoint.capture_rng_state()], history=[{"epoch": epoch}],
        best={"epoch": epoch, "top1": .3}, is_best=is_best)


def test_exact_next_update_rng_adam_and_scheduler_roundtrip(tmp_path):
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    model, optimizer, scheduler = create()
    for _ in range(5):
        update(model, optimizer, scheduler)
    paths = save(tmp_path, model, optimizer, scheduler)
    expected = update(model, optimizer, scheduler)
    restored, opt2, sched2 = create()
    state = checkpoint.load_training_checkpoint(paths["last"]["path"], restored, opt2, sched2, identity=identity())
    actual = update(restored, opt2, sched2)
    equal_tree(expected, actual)
    equal_tree(model.state_dict(), restored.state_dict())
    equal_tree(optimizer.state_dict(), opt2.state_dict())
    equal_tree(scheduler.state_dict(), sched2.state_dict())
    assert state["epoch"] == 5 and state["sampler"]["epoch"] == 6
    assert all(not key.startswith("module.") for key in state["model"])
    assert any(key.startswith("encoder.hierarchy.classifier.") for key in state["model"])


def transition(model, optimizer):
    # Same production contract: configure selective, rebuild optimizer, preserve
    # moments for existing connector parameters, add upper backbone parameters.
    replacement = torch.optim.AdamW(configure_stage(model, "selective", learning_rate=.00097), weight_decay=.01)
    for parameter in replacement.param_groups[0]["params"]:
        if parameter in optimizer.state:
            replacement.state[parameter] = copy.deepcopy(optimizer.state[parameter])
    return replacement


def test_epoch5_restore_then_epoch6_transition_preserves_exact_update(tmp_path):
    model, optimizer, _ = create()
    update(model, optimizer)
    schedule = {"schedule": "absolute_cosine50", "completed_epoch": 5, "total_epochs": 50}
    save(tmp_path, model, optimizer, schedule)
    optimizer = transition(model, optimizer)
    expected = update(model, optimizer)
    restored, opt2, _ = create()
    peek = checkpoint.inspect_training_checkpoint(tmp_path / "checkpoint_last.pt", identity())
    assert peek["stage"] == "connectors"
    checkpoint.load_checkpoint(peek, restored, opt2, identity=identity(), device="cpu")
    opt2 = transition(restored, opt2)
    actual = update(restored, opt2)
    equal_tree(expected, actual)
    equal_tree(model.state_dict(), restored.state_dict())
    equal_tree(optimizer.state_dict(), opt2.state_dict())


@pytest.mark.parametrize("field", ["class_names", "prompts", "source_sha256", "world_size", "config", "train_manifest_sha256", "clip_checkpoint_sha256"])
def test_identity_mismatch_fails_before_model_mutation(tmp_path, field):
    model, optimizer, scheduler = create()
    save(tmp_path, model, optimizer, scheduler)
    restored, opt2, sched2 = create()
    before = copy.deepcopy(restored.state_dict())
    different = identity()
    if field in ("class_names", "prompts"):
        different[field] = list(reversed(different[field]))
    elif field == "world_size":
        different[field] = 2
    else:
        different[field] = "changed"
    with pytest.raises(ValueError, match="identity mismatch"):
        checkpoint.load_training_checkpoint(tmp_path / "checkpoint_last.pt", restored, opt2, sched2, identity=different)
    equal_tree(before, restored.state_dict())


def test_wrong_stage_optimizer_order_and_pending_accumulation_rejected(tmp_path):
    model, optimizer, scheduler = create()
    save(tmp_path, model, optimizer, scheduler)
    restored, opt2, _ = create("selective")
    with pytest.raises(ValueError, match="configure checkpoint stage"):
        checkpoint.load_checkpoint(tmp_path / "checkpoint_last.pt", restored, opt2, identity=identity())
    restored, opt2, _ = create()
    opt2.param_groups[0]["params"].reverse()
    with pytest.raises(ValueError, match="parameter order"):
        checkpoint.load_checkpoint(tmp_path / "checkpoint_last.pt", restored, opt2, identity=identity())
    state = checkpoint.inspect_training_checkpoint(tmp_path / "checkpoint_last.pt")
    state["pending_accumulation_steps"] = 1
    with pytest.raises(ValueError, match="gradient accumulation"):
        checkpoint.inspect_training_checkpoint(state)


def test_no_legacy_checkpoint_overwrite_and_best_preserved(tmp_path):
    model, optimizer, scheduler = create()
    (tmp_path / "checkpoint_last.pt").write_bytes(b"protected legacy")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save(tmp_path, model, optimizer, scheduler)
    assert (tmp_path / "checkpoint_last.pt").read_bytes() == b"protected legacy"
    output = tmp_path / "new"
    save(output, model, optimizer, scheduler)
    best_hash = checkpoint.file_sha256(output / "checkpoint_best.pt")
    save(output, model, optimizer, scheduler, epoch=6, is_best=False)
    assert checkpoint.file_sha256(output / "checkpoint_best.pt") == best_hash
    assert checkpoint.inspect_training_checkpoint(output / "checkpoint_last.pt")["epoch"] == 6
    assert not list(output.glob("*.tmp"))


def test_rank_rng_sampler_worldsize_and_absolute_schedule_validation(tmp_path):
    model, optimizer, scheduler = create()
    with pytest.raises(ValueError, match="scheduler completed_epoch"):
        save(tmp_path, model, optimizer, {"completed_epoch": 4})
    save(tmp_path, model, optimizer, scheduler)
    state = checkpoint.inspect_training_checkpoint(tmp_path / "checkpoint_last.pt")
    state["sampler"]["world_size"] = 2
    with pytest.raises(ValueError, match="world_size"):
        checkpoint.inspect_training_checkpoint(state)
    state["sampler"]["world_size"] = 1
    state["rng_by_rank"] = []
    with pytest.raises(ValueError, match="RNG world size"):
        checkpoint.inspect_training_checkpoint(state)


def test_epoch_wrapper_exports_reconstructs_logits_and_retains_new_class_head(tmp_path):
    model, optimizer, _ = create()
    update(model, optimizer)
    result = checkpoint.save_epoch(tmp_path, model, optimizer, epoch=5, stage="connectors", identity=identity(),
        history=[], best={"epoch": 5}, improved=True, scheduler_state={"completed_epoch": 5},
        sampler_state={"seed": 42, "epoch": 6, "offset": 0, "world_size": 1})
    source = torch.load(result["exports"]["last"]["path"], map_location="cpu", weights_only=False)
    assert all(not key.startswith("classifier.") for key in source["backbone"])
    assert set(source["components"]) == {"temporal"}
    restored = FixtureComparison()
    checkpoint.load_transfer_bundle(tmp_path / "transfer_last.pt", restored, expected_identity=identity())
    x = torch.randn(2, 4)
    model.eval(); restored.eval()
    assert torch.equal(model(x), restored(x))
    target = FixtureComparison(classes=7)
    target_head = copy.deepcopy(target.classifier.state_dict())
    with pytest.raises(ValueError, match="allow_new_classes"):
        checkpoint.load_transfer_bundle(tmp_path / "transfer_last.pt", target, include_task_head=False)
    evidence = checkpoint.load_transfer_bundle(tmp_path / "transfer_last.pt", target,
                                             include_task_head=False, allow_new_classes=True)
    target.eval()
    assert torch.equal(model.forward_features(x), target.forward_features(x))
    equal_tree(target_head, target.classifier.state_dict())
    assert evidence["new_classes"] and not evidence["source_task_loaded"]
    assert source["checkpoint_sha256"] == result["checkpoints"]["last"]["sha256"]


def test_clip_transfer_separates_fixed_text_bank_and_rejects_reordered_prompts(tmp_path):
    from tests.test_hierarchy_clip_models import fixture_pretrained, TinySequenceEncoder, fixture_inputs
    from ebackbone_v3.hierarchy_clip_models import HierarchyCLIPViTAlignment
    pretrained = fixture_pretrained()
    model = HierarchyCLIPViTAlignment(TinySequenceEncoder(), pretrained, ["a", "b", "c"])
    checkpoint.export_transfer_bundle(tmp_path / "transfer.pt", model, identity=identity(), epoch=50, stage="selective")
    restored = HierarchyCLIPViTAlignment(TinySequenceEncoder(), pretrained, ["a", "b", "c"])
    checkpoint.load_transfer_bundle(tmp_path / "transfer.pt", restored)
    model.eval(); restored.eval()
    inputs = fixture_inputs()
    assert torch.equal(model.forward_embeddings(inputs), restored.forward_embeddings(inputs))
    assert torch.equal(model(inputs), restored(inputs))
    target = HierarchyCLIPViTAlignment(TinySequenceEncoder(), pretrained, ["c", "b", "a"])
    target_text = target.text_bank.embeddings.clone()
    with pytest.raises(ValueError, match="class order/prompts"):
        checkpoint.load_transfer_bundle(tmp_path / "transfer.pt", target)
    checkpoint.load_transfer_bundle(tmp_path / "transfer.pt", target, include_task_head=False, allow_new_classes=True)
    assert torch.equal(target.text_bank.embeddings, target_text)
    target.eval()
    assert torch.equal(model.forward_embeddings(inputs), target.forward_embeddings(inputs))
