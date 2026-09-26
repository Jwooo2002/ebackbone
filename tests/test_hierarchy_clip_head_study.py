"""Synthetic CPU verification of the preregistered CLIP head study."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from ebackbone_v3 import hierarchy_clip_head_study as study

ROOT_PROTOCOL = Path("/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_clip_head_study_artifacts/protocol.draft.json")


def _protocol(tmp_path):
    protocol = json.loads(ROOT_PROTOCOL.read_text())
    protocol["output_dir"] = str(tmp_path)
    protocol["_sha256"] = "synthetic-protocol"
    return protocol


def _features(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    g = np.random.default_rng(901)
    path = tmp_path / "train.npz"
    np.savez(path, features=g.standard_normal((12, 256), dtype=np.float32), labels=np.arange(12, dtype=np.int64) % 3)
    return path


def _text(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    g = np.random.default_rng(902)
    path = tmp_path / "text.npz"
    bank = g.standard_normal((100, 512), dtype=np.float32)
    bank /= np.linalg.norm(bank, axis=1, keepdims=True)
    np.savez(path, embeddings=bank)
    return path, bank


def _prepared_with_target_banks(protocol, tmp_path, feature, text):
    targets = study._prepare_target_banks(protocol, text)
    digest = lambda path: __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest()
    prepared = tmp_path / "features"; prepared.mkdir(exist_ok=True)
    (prepared / "prepared.json").write_text(json.dumps({
        "status": "COMPLETE", "identity": {"protocol_sha256": protocol["_sha256"]},
        "features": {"train": {"path": str(feature), "sha256": digest(feature)},
                      "validation": {"path": str(feature), "sha256": digest(feature)}},
        "clip_text_bank": {"path": str(text), "sha256": digest(text)},
        "target_banks": targets,
    }))
    return targets


def test_real_protocol_contains_exact_matrix_and_seeds():
    protocol = json.loads(ROOT_PROTOCOL.read_text())
    conditions = protocol["conditions"]
    assert len(conditions) == 45
    assert {c["optimization_seed"] for c in conditions} == {6101, 6102, 6103}
    assert [sum(c["kind"] == kind for c in conditions) for kind in ("learned", "clip", "random", "permuted_clip")] == [3, 3, 30, 9]
    assert protocol["training"]["epochs"] == 50 and protocol["training"]["batch_size"] == 256
    assert protocol["training"]["logit_scale_weight_decay"] == 0.0


def test_condition_gradient_policy_and_shared_projector_initialization():
    bank = np.eye(512, dtype=np.float32)[:100]
    learned_spec = {"id": "learned", "kind": "learned", "optimization_seed": 6101, "bank_seed": 101}
    clip_spec = {"id": "clip", "kind": "clip", "optimization_seed": 6101, "bank_seed": 0}
    learned = study._condition(learned_spec, bank)
    fixed = study._condition(clip_spec, bank)
    assert learned.prototypes.requires_grad and not fixed.prototypes.requires_grad
    torch.testing.assert_close(learned.projector.weight, fixed.projector.weight, rtol=0, atol=0)
    torch.testing.assert_close(learned.projector.bias, fixed.projector.bias, rtol=0, atol=0)
    assert learned.logit_scale.item() == fixed.logit_scale.item() == pytest.approx(np.log(10.0))
    assert learned.logit_scale.requires_grad and fixed.logit_scale.requires_grad


def test_production_fit_pairing_digests_and_initial_learned_random_logits(tmp_path):
    protocol = _protocol(tmp_path)
    train = _features(tmp_path); text, bank = _text(tmp_path)
    specs = [next(c for c in protocol["conditions"] if c["id"] == name) for name in
             ("learned_opt6101_bank101", "random_opt6101_bank101", "clip_opt6101_bank0")]
    states = []
    for spec in specs:
        checkpoint = study._fit(protocol, spec, {"train": train}, text)
        states.append(torch.load(checkpoint, map_location="cpu", weights_only=False))
    assert len({s["pairing"]["projector_initial_digest"] for s in states}) == 1
    assert len({s["pairing"]["batch_order_digest"] for s in states}) == 1
    learned = study._condition(specs[0], bank)
    random = study._condition(specs[1], bank)
    features = torch.from_numpy(np.load(train)["features"])
    with torch.no_grad():
        learned_logits = learned(features); random_logits = random(features)
    torch.testing.assert_close(learned_logits, random_logits, rtol=0, atol=0)


def test_exported_target_banks_feed_fits_and_corruption_is_rejected(tmp_path):
    protocol = _protocol(tmp_path)
    protocol["conditions"] = [next(c for c in protocol["conditions"] if c["kind"] == kind)
                               for kind in ("learned", "random", "clip")]
    text, bank = _text(tmp_path)
    targets = study._prepare_target_banks(protocol, text)
    train = _features(tmp_path)
    states = []
    for condition in protocol["conditions"]:
        path = study._fit(protocol, condition, {"train": train}, text,
                          Path(targets[condition["id"]]["path"]))
        states.append(torch.load(path, map_location="cpu", weights_only=False))
    assert len({state["pairing"]["projector_initial_digest"] for state in states}) == 1
    assert len({state["pairing"]["batch_order_digest"] for state in states}) == 1
    clip_target = Path(targets[protocol["conditions"][-1]["id"]]["path"])
    clip_target.write_bytes(clip_target.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="target-bank identity mismatch"):
        study._prepare_target_banks(protocol, text)


def test_permuted_bank_is_direct_seeded_permutation():
    bank = np.eye(512, dtype=np.float32)[:100]
    spec = {"id": "perm", "kind": "permuted_clip", "optimization_seed": 6101, "bank_seed": 301}
    head = study._condition(spec, bank)
    order = torch.randperm(100, generator=torch.Generator(device="cpu").manual_seed(301)).numpy()
    np.testing.assert_array_equal(head.prototypes.detach().numpy(), bank[order])


def test_repeated_production_fit_is_bit_exact_and_reloadable(tmp_path):
    condition = {"id": "clip_opt6101_bank0", "kind": "clip", "optimization_seed": 6101, "bank_seed": 0}
    protocol = _protocol(tmp_path / "a"); train = _features(tmp_path / "a"); text, _ = _text(tmp_path / "a")
    first = study._fit(protocol, condition, {"train": train}, text)
    protocol2 = _protocol(tmp_path / "b"); train2 = _features(tmp_path / "b"); text2, _ = _text(tmp_path / "b")
    second = study._fit(protocol2, condition, {"train": train2}, text2)
    one = torch.load(first, map_location="cpu", weights_only=False); two = torch.load(second, map_location="cpu", weights_only=False)
    for key in one["model"]:
        torch.testing.assert_close(one["model"][key], two["model"][key], rtol=0, atol=0)
    assert one["train_history"] == two["train_history"]
    assert one["diagnostics"]["final_logit_scale"] <= 100.0


def test_evaluation_refuses_before_metrics_when_final_is_missing(tmp_path, monkeypatch):
    protocol = _protocol(tmp_path)
    (tmp_path / "fits").mkdir(parents=True)
    ids = [condition["id"] for condition in protocol["conditions"]]
    feature = tmp_path / "feature.npz"
    np.savez(feature, sample_ids=np.array(["a"]), labels=np.array([0]), features=np.zeros((1, 256), np.float32), original_logits=np.zeros((1, 100), np.float32))
    text, _ = _text(tmp_path)
    targets = _prepared_with_target_banks(protocol, tmp_path, feature, text)
    (tmp_path / "fits" / "complete.json").write_text(json.dumps({"all_finals_complete": True, "protocol_sha256": protocol["_sha256"], "fit_checkpoints": {identifier: {"path": str(tmp_path / "fits" / identifier / "final.pt"), "sha256": "missing"} for identifier in ids}}))
    calls = []
    monkeypatch.setattr(study, "_metrics", lambda *args: calls.append(args) or {})
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        study.evaluate(protocol)
    assert calls == []


@pytest.mark.parametrize("wrong_state", [False, True])
def test_evaluation_rejects_corrupt_or_wrong_identity_final_before_metrics(tmp_path, monkeypatch, wrong_state):
    protocol = _protocol(tmp_path)
    ids = [condition["id"] for condition in protocol["conditions"]]
    feature = tmp_path / "feature.npz"; np.savez(feature, sample_ids=np.array(["a"]), labels=np.array([0]), features=np.zeros((1, 256), np.float32), original_logits=np.zeros((1, 100), np.float32))
    text, _ = _text(tmp_path)
    targets = _prepared_with_target_banks(protocol, tmp_path, feature, text)
    first = tmp_path / "fits" / ids[0] / "final.pt"; first.parent.mkdir(parents=True)
    if wrong_state:
        torch.save({"identity": {"wrong": True}, "complete": True, "final_epoch": 50, "train_history": [{}] * 50}, first)
    else:
        first.write_bytes(b"corrupt checkpoint")
    digest = lambda path: __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest()
    digest_first = digest(first)
    records = {identifier: {"path": str(first if index == 0 else tmp_path / "fits" / identifier / "final.pt"), "sha256": digest_first if index == 0 else "missing"} for index, identifier in enumerate(ids)}
    (tmp_path / "fits").mkdir(exist_ok=True)
    (tmp_path / "fits" / "complete.json").write_text(json.dumps({"all_finals_complete": True, "protocol_sha256": protocol["_sha256"], "fit_checkpoints": records}))
    calls = []
    monkeypatch.setattr(study, "_metrics", lambda *args: calls.append(args) or {})
    expected = (pickle.UnpicklingError, RuntimeError, EOFError) if not wrong_state else (ValueError,)
    with pytest.raises(expected):
        study.evaluate(protocol)
    assert calls == []


def test_unsealed_protocol_is_rejected_before_any_run(tmp_path):
    protocol = json.loads(ROOT_PROTOCOL.read_text())
    path = tmp_path / "protocol.json"; path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="SEALED_BEFORE_OUTCOMES|integrity_files"):
        study._load_protocol(path)


def test_sealed_protocol_rejects_source_hash_mismatch(tmp_path):
    protocol = json.loads(ROOT_PROTOCOL.read_text())
    protocol["status"] = "SEALED_BEFORE_OUTCOMES"
    protocol["source_sha256"] = {"hierarchy_clip_head_study.py": "0" * 64}
    protocol["integrity_files"] = {str(ROOT_PROTOCOL): "unused"}
    path = tmp_path / "protocol.json"; path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="sealed source hash mismatch"):
        study._load_protocol(path)
