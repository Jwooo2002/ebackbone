from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from ebackbone_v3 import v1_training as training
from ebackbone_v3.v1_models import V1Backbone, example_inputs


class FakeDataset:
    """Small tensors exercise the real models/runner; not scientific data."""
    def __init__(self, manifest_dir=None, dataset_root=None, split="train", *, multiview=False,
                 raw_cache=None, limit=None, seed=None):
        self.split, self.multiview = split, multiview
        self.rows = [SimpleNamespace(sample_id=f"{split}/{i}", raw_content_sha256=f"{split}:{i}")
                     for i in range(limit or (5 if split == "train" else 3))]
        self.manifest_sha256 = split + "_immutable"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index + 42)
        inputs = example_inputs("same_arch" if self.multiview else "frame", 16, 24)
        return {"inputs": {k: torch.rand(v.shape[1:], generator=generator) for k, v in inputs.items()},
                "label": index % 2, "sample_id": self.rows[index].sample_id, "split": self.split,
                "source_split": "train", "render_seconds": 0.01, "decode_seconds": 0.02,
                "source": {"sample_id": self.rows[index].sample_id}}


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr(training, "V1Dataset", FakeDataset)
    return training.TrainConfig(epochs=2, batch_size=2, accumulation_steps=2, device="cpu",
                                precision="float32", num_workers=0, cpu_threads=2)


def run(config, path, **kwargs):
    return training.run_model("frame", config, manifest_dir="unused", dataset_root="unused",
                              output_dir=path, **kwargs)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_exact_epoch_resume(config, tmp_path):
    whole = run(config, tmp_path / "whole")
    stopped = run(config, tmp_path / "resumed", stop_after_epoch=1)
    assert stopped["status"] == "partial"
    resumed = run(config, tmp_path / "resumed", resume=True)
    assert whole["status"] == resumed["status"] == "complete"
    a = torch.load(tmp_path / "whole/checkpoint_last.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed/checkpoint_last.pt", weights_only=False)
    for key in ("model", "optimizer", "scheduler"):
        assert_nested_equal(a[key], b[key])
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    for left, right in zip(whole["history"], resumed["history"]):
        for split in ("train", "validation"):
            for field in ("loss", "top1", "top5", "sample_order_sha256", "samples"):
                assert left[split][field] == right[split][field]
    assert resumed["checkpoint_verification"]["logits_bit_exact"]


def test_accumulation_including_partial_group(config):
    torch.set_num_threads(2)
    torch.manual_seed(8)
    first, second = V1Backbone("frame"), V1Backbone("frame")
    second.load_state_dict(first.state_dict())
    a = torch.optim.SGD(first.parameters(), lr=0.001)
    b = torch.optim.SGD(second.parameters(), lr=0.001)
    data = FakeDataset()
    training.run_epoch(first, data, config, 1, optimizer=a)
    training.run_epoch(second, data, replace(config, batch_size=4, accumulation_steps=1), 1, optimizer=b)
    for p, q in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-5)


def test_four_way_comparison_and_resume_guards(config, tmp_path):
    report = training.compare(replace(config, epochs=1), manifest_dir="unused", dataset_root="unused",
                              output_dir=tmp_path / "compare", diagnostic_samples=2)
    assert report["status"] == "complete"
    assert report["evidence_kind"] == "train-only engineering diagnostic"
    assert report["identical_training_settings_and_order"]
    assert not report["final_test_accessed"]
    assert len(report["models"]) == 4
    path = tmp_path / "guards"
    run(config, path, stop_after_epoch=1)
    with pytest.raises(ValueError, match="overwrite"):
        run(config, path)
    with pytest.raises(ValueError, match="identity mismatch"):
        run(replace(config, learning_rate=0.03), path, resume=True)
    with pytest.raises(ValueError, match="identity mismatch"):
        run(config, path, resume=True, diagnostic_samples=2)


def test_loader_order_independent_of_model_rng(config):
    data = FakeDataset()
    torch.manual_seed(1)
    a = [sample for batch in training.loader(data, config, 1, True) for sample in batch["sample_ids"]]
    torch.manual_seed(999)
    b = [sample for batch in training.loader(data, config, 1, True) for sample in batch["sample_ids"]]
    assert a == b


def test_bad_config_fails_before_data_access(config, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("dataset must not be constructed")
    monkeypatch.setattr(training, "V1Dataset", forbidden)
    for change in ({"epochs": 0}, {"accumulation_steps": 0}, {"device": "cuda:0"},
                   {"learning_rate": float("nan")}, {"precision": "bfloat16"}):
        with pytest.raises(ValueError):
            run(replace(config, **change), tmp_path / "unused")
