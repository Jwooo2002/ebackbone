"""Synthetic, isolated verification of the post-evaluation weight exporter."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


EXPORT = Path("/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_clip_head_study_artifacts/export_weights.py")


def _module():
    spec = importlib.util.spec_from_file_location("synthetic_export_weights", EXPORT)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fixture(tmp_path):
    conditions = [{"id": f"condition_{index:02d}", "kind": "clip", "optimization_seed": index, "bank_seed": 0} for index in range(45)]
    hierarchy = tmp_path / "hierarchy.pt"
    backbone = {"frame.weight": torch.ones(2, 2), "classifier.weight": torch.ones(3, 2), "classifier.bias": torch.zeros(3)}
    torch.save({"identity": {"variant": "hierarchy"}, "model": backbone}, hierarchy)
    classes = tmp_path / "classes.json"; classes.write_text(json.dumps({"class_names": [f"class_{i}" for i in range(100)]}))
    output = tmp_path / "study"; output.mkdir()
    protocol = {"status": "SEALED_BEFORE_OUTCOMES", "output_dir": str(output), "conditions": conditions,
                "hierarchy_checkpoint": str(hierarchy), "class_names_path": str(classes),
                "integrity_files": {str(hierarchy): _sha(hierarchy), str(classes): _sha(classes)},
                "feature_contract": {"dim": 256}, "prompt_template": "a photo of a {class_name}."}
    protocol_path = tmp_path / "protocol.json"; protocol_path.write_text(json.dumps(protocol, sort_keys=True))
    protocol_hash = _sha(protocol_path)
    fits = output / "fits"; fits.mkdir()
    records = {}
    for condition in conditions:
        state = {"projector.weight": torch.zeros(512, 256), "projector.bias": torch.zeros(512),
                 "prototypes": torch.nn.functional.normalize(torch.ones(100, 512), dim=1), "logit_scale": torch.tensor(1.)}
        checkpoint = fits / f"{condition['id']}.pt"
        torch.save({"identity": {"protocol_sha256": protocol_hash, "condition": condition}, "complete": True,
                    "final_epoch": 50, "train_history": [{}] * 50, "model": state}, checkpoint)
        records[condition["id"]] = {"path": str(checkpoint), "sha256": _sha(checkpoint)}
    (fits / "complete.json").write_text(json.dumps({"status": "COMPLETE", "protocol_sha256": protocol_hash,
        "all_finals_complete": True, "fit_checkpoints": records}))
    evaluation = output / "evaluation"; evaluation.mkdir()
    (evaluation / "report.json").write_text(json.dumps({"status": "COMPLETE", "protocol_sha256": protocol_hash,
        "all_matrix_finals_validated": True}))
    return protocol_path, protocol, hierarchy


def test_synthetic_export_is_exact_45_heads_reusable_and_classifier_free(tmp_path):
    module = _module(); protocol_path, protocol, hierarchy = _fixture(tmp_path)
    result = module.export(protocol_path, tmp_path / "exports")
    assert result["status"] == "COMPLETE" and len(result["heads"]) == 45
    for condition_id, record in result["heads"].items():
        payload = torch.load(record["path"], map_location="cpu", weights_only=False)
        assert set(payload["state"]) == {"projector.weight", "projector.bias", "prototypes", "logit_scale"}
        assert _sha(record["path"]) == record["sha256"]
    frozen = torch.load(result["original_frozen_backbone"]["path"], map_location="cpu", weights_only=False)
    assert frozen["classifier_excluded"] is True
    assert not any(key.startswith("classifier.") for key in frozen["state"])
    assert "classifier.weight" in frozen["excluded_keys"]
    assert module.export(protocol_path, tmp_path / "exports") == result


def test_export_refuses_before_complete_evaluation_and_corrupt_final(tmp_path):
    module = _module(); protocol_path, protocol, _ = _fixture(tmp_path)
    report = Path(protocol["output_dir"]) / "evaluation" / "report.json"
    report.unlink()
    with pytest.raises(ValueError, match="fits and evaluation"):
        module.export(protocol_path, tmp_path / "premature")
