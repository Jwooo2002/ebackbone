from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from ebackbone_v3 import hierarchy_clip_train as train
from ebackbone_v3.hierarchy_ddp import ExactBatchShard
from tests.test_hierarchy_clip_temporal import _tiny_sample


SETTINGS = {"batch_size": 2, "microbatch_size": 1, "seed": 321, "num_workers": 0,
            "cpu_threads": 1, "learning_rate": 1e-4, "backbone_lr_ratio": .1,
            "weight_decay": .01, "connector_epochs": 5, "epochs": 50}


class TinyHierarchy(nn.Module):
    def __init__(self):
        super().__init__()
        self.point = nn.Linear(4, 8)
        self.temporal_collapse = nn.Linear(8, 8)
        self.frame_stage = nn.Linear(8, 8)
        self.classifier = nn.Linear(8, 3)  # unused pretrained CE head stays frozen

    def forward(self, inputs):
        points = self.point(inputs["packed"]["points"])
        counts = inputs["packed"]["event_counts"]
        groups, start = [], 0
        for count in counts.tolist():
            groups.append(points[start:start + count].mean(0) if count else points.new_zeros(8))
            start += count
        windows = torch.stack(groups).reshape(*inputs["window_mask"].shape, 8)
        return self.frame_stage(torch.tanh(self.temporal_collapse(windows)))


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hierarchy = TinyHierarchy()

    def forward(self, inputs):
        return self.hierarchy(inputs)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.temporal = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        self.classifier = nn.Linear(8, 3)

    def forward(self, inputs):
        maps = self.encoder(inputs)
        mask = inputs["window_mask"][..., None]
        features = self.temporal(maps).masked_fill(~mask, 0).sum(1) / mask.sum(1).clamp_min(1)
        return self.classifier(features)


class TinyData:
    def __init__(self, split="train", size=7):
        self.split = split
        self.rows = [SimpleNamespace(sample_id=f"{split}/{index}") for index in range(size)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        sample = _tiny_sample(index)
        sample.update(sample_id=self.rows[index].sample_id, split=self.split)
        return sample


def _nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _nested_equal(a, b)
    else:
        assert left == right


def test_exact_global64_shards_and_continuous_cosine():
    shards = [ExactBatchShard(124395, rank, 2, 32, 321, 5, True) for rank in range(2)]
    assert shards[0].global_sizes[-1] == 43
    indices = [[index for batch in shard for index in batch] for shard in shards]
    assert set(indices[0]).isdisjoint(indices[1])
    assert sorted(indices[0] + indices[1]) == list(range(124395))
    assert max(map(len, shards[0].batches)) == max(map(len, shards[1].batches)) == 32
    model = TinyModel()
    _, optimizer, _ = train.setup_stage(model, "connectors", SETTINGS, "cpu", wrap_ddp=False)
    at1 = train.set_epoch_learning_rate(optimizer, 1)["connectors"]
    at5 = train.set_epoch_learning_rate(optimizer, 5)["connectors"]
    _, optimizer, _ = train.setup_stage(model, "selective", SETTINGS, "cpu", optimizer, wrap_ddp=False)
    at6 = train.set_epoch_learning_rate(optimizer, 6)
    at50 = train.set_epoch_learning_rate(optimizer, 50)
    assert at1 == 1e-4 and at50["connectors"] < at6["connectors"] < at5
    assert at6["pretrained_upper"] == pytest.approx(at6["connectors"] * .1)
    assert [train.stage_for_epoch(epoch) for epoch in (1, 5, 6, 50)] == ["connectors", "connectors", "selective", "selective"]


def test_full_protocol_guards_and_canonical_fingerprint():
    config = {"full_training_authorized": True, "num_windows": 4, "training": {
        **SETTINGS, "batch_size": 32, "microbatch_size": 8, "world_size": 2, "precision": "bfloat16"}}
    train.validate_training_config(config)
    for key, value in (("epochs", 5), ("batch_size", 16), ("connector_epochs", 6), ("world_size", 1), ("microbatch_size", 33)):
        altered = deepcopy(config)
        altered["training"][key] = value
        with pytest.raises(ValueError):
            train.validate_training_config(altered)
    assert train.config_fingerprint({"a": 1, "b": 2}) == train.config_fingerprint({"b": 2, "a": 1})


def _worker(rank, init_file, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=2)
    device = torch.device("cpu")
    data = TinyData(size=7)
    train.seed_everything(SETTINGS, device)
    model = TinyModel()
    ddp, optimizer, _ = train.setup_stage(model, "connectors", SETTINGS, device)
    before = deepcopy(model.state_dict())
    # Uneven2/1 local slices, microbatch1: DDP's averaged scaled gradients must
    # equal an independent ordinary model's mean CE over all three examples.
    reference = TinyModel()
    reference.load_state_dict(before)
    _, ref_optimizer, _ = train.setup_stage(reference, "connectors", SETTINGS, device, wrap_ddp=False)
    global_samples = [data[index] for index in range(3)]
    local_samples = global_samples[rank::2]
    result = train.optimizer_update(ddp, local_samples, optimizer, device, 3, microbatch_size=1,
                                    height=32, width=32, probe=True)
    inputs = train.sequence_collate(global_samples, height=32, width=32)
    torch.nn.functional.cross_entropy(reference(inputs["inputs"]), inputs["labels"]).backward()
    for parameter, expected in zip(model.parameters(), reference.parameters()):
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, expected.grad, atol=2e-8, rtol=2e-5)
        else:
            assert parameter.grad is expected.grad is None
    assert result["gradient_report"]["frozen_parameters_have_no_gradient"]
    report = train.run_epoch(ddp, data, SETTINGS, device, 5, optimizer, probe=True, height=32, width=32)
    evaluation = train.run_epoch(ddp, TinyData("validation", 5), SETTINGS, device, 5, height=32, width=32)
    assert report["samples"] == 7 and report["optimizer_steps"] == 2
    assert report["gradient_checks"][-1]["global_size"] == 3
    assert evaluation["rank_sample_counts"] == [3, 2]
    assert evaluation["duplicates"] == evaluation["missing"] == 0
    # Snapshot exactly at epoch5 boundary, including connector Adam moments.
    checkpoint = {"model": deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict()),
                  "rng": torch.get_rng_state().clone()}
    connector_state = {name: deepcopy(optimizer.state[parameter]) for name, parameter in model.named_parameters()
                       if parameter.requires_grad}
    del ddp
    ddp, optimizer, transition = train.setup_stage(model, "selective", SETTINGS, device, optimizer)
    assert set(transition["preserved_adam_parameters"]) == set(connector_state)
    for name, parameter in model.named_parameters():
        if name in connector_state:
            _nested_equal(optimizer.state[parameter], connector_state[name])
        elif parameter.requires_grad:
            assert parameter not in optimizer.state
    train.set_epoch_learning_rate(optimizer, 6)
    continued = train.run_epoch(ddp, data, SETTINGS, device, 6, optimizer, probe=True, height=32, width=32)
    expected = {"model": deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict()),
                "rng": torch.get_rng_state().clone()}
    assert all(check["stage_l1"]["encoder"] > 0 for check in continued["gradient_checks"])
    del ddp, optimizer, model
    # Reconstruct at the saved connectors stage, restore, then transition.
    model = TinyModel()
    model.load_state_dict(checkpoint["model"], strict=True)
    ddp, optimizer, _ = train.setup_stage(model, "connectors", SETTINGS, device)
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng"])
    del ddp
    ddp, optimizer, _ = train.setup_stage(model, "selective", SETTINGS, device, optimizer)
    train.set_epoch_learning_rate(optimizer, 6)
    resumed = train.run_epoch(ddp, data, SETTINGS, device, 6, optimizer, probe=True, height=32, width=32)
    _nested_equal(expected["model"], model.state_dict())
    _nested_equal(expected["optimizer"], optimizer.state_dict())
    _nested_equal(expected["rng"], torch.get_rng_state())
    for key in ("loss", "top1", "top5", "rank_order_sha256", "gradient_checks"):
        assert resumed[key] == continued[key]
    # Save at an optimizer boundary in selective stage; prove exact next update.
    current = {"model": deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict())}
    train.optimizer_update(ddp, local_samples, optimizer, device, 3, microbatch_size=1, height=32, width=32)
    next_update = {"model": deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict())}
    del ddp, optimizer, model
    model = TinyModel()
    model.load_state_dict(current["model"])
    ddp, optimizer, _ = train.setup_stage(model, "selective", SETTINGS, device)
    optimizer.load_state_dict(current["optimizer"])
    train.optimizer_update(ddp, local_samples, optimizer, device, 3, microbatch_size=1, height=32, width=32)
    _nested_equal(next_update["model"], model.state_dict())
    _nested_equal(next_update["optimizer"], optimizer.state_dict())
    if rank == 0:
        Path(output_dir, "cpu_checks.json").write_text(json.dumps({"weighted_gradient": "PASS",
            "coverage": report, "validation": evaluation, "stage_boundary_resume": "bit_exact",
            "next_update_resume": "bit_exact"}, indent=2))
    dist.destroy_process_group()


def test_two_rank_weighting_staging_coverage_and_exact_resume(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "init"), str(tmp_path)), nprocs=2, join=True)
    report = json.loads((tmp_path / "cpu_checks.json").read_text())
    assert report["stage_boundary_resume"] == report["next_update_resume"] == "bit_exact"
