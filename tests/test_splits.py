from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ebackbone_v3.errors import SplitError
from ebackbone_v3.n_imagenet_mini_index import build_archive_index
from ebackbone_v3.splits import (
    CHECKSUM_FILENAME,
    MANIFEST_FILENAMES,
    assign_project_splits,
    load_split_build_config,
    publish_split_manifests,
    verify_splits,
)
from tests.test_n_imagenet_mini_index import _write_release


def _index(tmp_path: Path, *, reverse_members: bool = False):
    dataset_root = tmp_path / "n_imagenet"
    _write_release(
        dataset_root,
        reverse_members=reverse_members,
        train_samples_per_class=3,
    )
    return build_archive_index(dataset_root)


def _artifact_bytes(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def _rows(directory: Path, split: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (directory / MANIFEST_FILENAMES[split]).read_text(
            encoding="utf-8"
        ).splitlines()
    ]


def _rewrite_checksums(directory: Path) -> None:
    names = sorted(path.name for path in directory.iterdir() if path.name != CHECKSUM_FILENAME)
    payload = "".join(
        f"{hashlib.sha256((directory / name).read_bytes()).hexdigest()}  {name}\n"
        for name in names
    )
    (directory / CHECKSUM_FILENAME).write_text(payload, encoding="ascii")


def _write_canonical_rows(directory: Path, split: str, rows: list[dict[str, object]]) -> None:
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in sorted(rows, key=lambda row: str(row["sample_id"]))
    ).encode("utf-8")
    (directory / MANIFEST_FILENAMES[split]).write_bytes(payload)


def _refresh_manifest_metadata(directory: Path, *splits: str) -> None:
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for split in splits:
        payload = (directory / MANIFEST_FILENAMES[split]).read_bytes()
        provenance["manifest_files"][split]["size_bytes"] = len(payload)
        provenance["manifest_files"][split]["sha256"] = hashlib.sha256(payload).hexdigest()
        provenance["manifest_files"][split]["sample_count"] = len(payload.splitlines())
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rewrite_checksums(directory)


def test_generation_is_byte_identical_and_same_target_is_not_mutated(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    first_dir = tmp_path / "manifests-a"
    second_dir = tmp_path / "manifests-b"

    first = publish_split_manifests(
        index,
        manifest_dir=first_dir,
        seed=20260715,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )
    second = publish_split_manifests(
        index,
        manifest_dir=second_dir,
        seed=20260715,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )

    assert first["publication"] == second["publication"] == "created"
    assert _artifact_bytes(first_dir) == _artifact_bytes(second_dir)
    before = {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in first_dir.iterdir()
    }
    repeated = publish_split_manifests(
        index,
        manifest_dir=first_dir,
        seed=20260715,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )
    after = {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.read_bytes())
        for path in first_dir.iterdir()
    }
    assert repeated["publication"] == "verified_existing_unchanged"
    assert before == after


def test_assignment_and_bytes_do_not_depend_on_index_iteration_order(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    reversed_index = replace(index, samples=tuple(reversed(index.samples)))

    forward_assignment = {
        record.source.sample_id: record.split
        for record in assign_project_splits(
            index, seed=20260715, internal_validation_per_class=1
        )
    }
    reverse_assignment = {
        record.source.sample_id: record.split
        for record in assign_project_splits(
            reversed_index, seed=20260715, internal_validation_per_class=1
        )
    }
    assert forward_assignment == reverse_assignment

    first_dir = tmp_path / "ordered"
    second_dir = tmp_path / "reversed"
    publish_split_manifests(
        index,
        manifest_dir=first_dir,
        seed=20260715,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )
    publish_split_manifests(
        reversed_index,
        manifest_dir=second_dir,
        seed=20260715,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )
    assert _artifact_bytes(first_dir) == _artifact_bytes(second_dir)


def test_stratification_disjointness_and_official_validation_to_test_mapping(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    directory = tmp_path / "manifests"
    report = publish_split_manifests(
        index,
        manifest_dir=directory,
        seed=20260715,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )

    rows = {split: _rows(directory, split) for split in MANIFEST_FILENAMES}
    assert report["sample_counts"] == {"train": 200, "validation": 100, "test": 100}
    assert all(len({row["class_id"] for row in rows[split]}) == 100 for split in rows)
    assert all(row["source_split"] == "train" for row in rows["train"])
    assert all(row["source_split"] == "train" for row in rows["validation"])
    assert all(row["source_split"] == "validation" for row in rows["test"])
    assert all(row["split"] == "test" for row in rows["test"])
    assert all(str(row["sample_id"]).startswith("validation/") for row in rows["test"])
    ids = {split: {row["sample_id"] for row in split_rows} for split, split_rows in rows.items()}
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])


def test_seed_and_quota_change_provenance_and_cannot_overwrite(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    target = tmp_path / "fixed"
    original = publish_split_manifests(
        index,
        manifest_dir=target,
        seed=20260715,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )
    before = _artifact_bytes(target)

    with pytest.raises(SplitError, match="refusing to regenerate or overwrite"):
        publish_split_manifests(
            index,
            manifest_dir=target,
            seed=20260716,
            internal_validation_per_class=1,
            enforce_official_release=False,
        )
    assert _artifact_bytes(target) == before

    with pytest.raises(SplitError, match="refusing to regenerate or overwrite"):
        publish_split_manifests(
            index,
            manifest_dir=target,
            seed=20260715,
            internal_validation_per_class=2,
            enforce_official_release=False,
        )
    assert _artifact_bytes(target) == before

    seed_dir = tmp_path / "different-seed"
    changed_seed = publish_split_manifests(
        index,
        manifest_dir=seed_dir,
        seed=20260716,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )
    quota_dir = tmp_path / "different-quota"
    changed_quota = publish_split_manifests(
        index,
        manifest_dir=quota_dir,
        seed=20260715,
        internal_validation_per_class=2,
        enforce_official_release=False,
    )
    assert original["generation_fingerprint_sha256"] != changed_seed[
        "generation_fingerprint_sha256"
    ]
    assert original["generation_fingerprint_sha256"] != changed_quota[
        "generation_fingerprint_sha256"
    ]
    assert changed_quota["sample_counts"] == {
        "train": 100,
        "validation": 200,
        "test": 100,
    }


def test_provenance_alone_reproduces_internal_validation_membership(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    directory = tmp_path / "manifests"
    publish_split_manifests(
        index,
        manifest_dir=directory,
        seed=20260715,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )

    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    generation = provenance["generation_parameters"]
    assert generation["selection_rank"] == (
        "SHA256(UTF8(selection_domain_utf8) || NUL || ASCII(decimal_seed) || "
        "NUL || UTF8(source_stable_sample_id)); sort by (digest, sample_id) within class"
    )
    domain = generation["selection_domain_utf8"].encode("utf-8")
    seed = str(generation["seed"]).encode("ascii")
    quota = generation["internal_validation_per_class"]
    train_rows = _rows(directory, "train")
    validation_rows = _rows(directory, "validation")
    source_train_rows = train_rows + validation_rows
    expected_validation_ids: set[str] = set()
    for class_id in sorted(provenance["class_to_index"]):
        class_ids = [
            str(row["sample_id"])
            for row in source_train_rows
            if row["class_id"] == class_id
        ]
        ranked = sorted(
            class_ids,
            key=lambda sample_id: (
                hashlib.sha256(
                    domain
                    + b"\0"
                    + seed
                    + b"\0"
                    + sample_id.encode("utf-8")
                ).digest(),
                sample_id,
            ),
        )
        expected_validation_ids.update(ranked[:quota])
    assert expected_validation_ids == {
        str(row["sample_id"]) for row in validation_rows
    }


def test_verify_rejects_data_tampering_and_self_consistent_provenance_tampering(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    directory = tmp_path / "manifests"
    publish_split_manifests(
        index,
        manifest_dir=directory,
        seed=20260715,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )

    train_path = directory / MANIFEST_FILENAMES["train"]
    train_path.write_bytes(train_path.read_bytes() + b"\n")
    with pytest.raises(SplitError, match="SHA256SUMS"):
        verify_splits(directory)

    clean = tmp_path / "clean"
    publish_split_manifests(
        index,
        manifest_dir=clean,
        seed=20260715,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )
    provenance_path = clean / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["project_counts"]["train"] += 1
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_checksums(clean)
    with pytest.raises(SplitError, match="project_counts"):
        verify_splits(clean)


def test_verify_recomputes_exact_seeded_assignment_after_self_consistent_swap(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    directory = tmp_path / "manifests"
    publish_split_manifests(
        index,
        manifest_dir=directory,
        seed=20260715,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )
    train_rows = _rows(directory, "train")
    validation_rows = _rows(directory, "validation")
    class_id = str(validation_rows[0]["class_id"])
    selected = next(row for row in validation_rows if row["class_id"] == class_id)
    replacement = next(row for row in train_rows if row["class_id"] == class_id)
    validation_rows.remove(selected)
    train_rows.remove(replacement)
    selected["split"] = "train"
    replacement["split"] = "validation"
    train_rows.append(selected)
    validation_rows.append(replacement)
    _write_canonical_rows(directory, "train", train_rows)
    _write_canonical_rows(directory, "validation", validation_rows)
    _refresh_manifest_metadata(directory, "train", "validation")

    with pytest.raises(SplitError, match="deterministic SHA-256 rank"):
        verify_splits(directory)


def test_verify_rejects_wrong_source_roles_and_nonstandard_json_constants(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    wrong_role = tmp_path / "wrong-role"
    publish_split_manifests(
        index,
        manifest_dir=wrong_role,
        seed=20260715,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )
    provenance_path = wrong_role / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["source_files"][0]["role"] = "not_an_archive_role"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rewrite_checksums(wrong_role)
    with pytest.raises(SplitError, match="paths, roles, or authority"):
        verify_splits(wrong_role)

    nonstandard = tmp_path / "nonstandard"
    publish_split_manifests(
        index,
        manifest_dir=nonstandard,
        seed=20260715,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )
    provenance_path = nonstandard / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["project_counts"]["train"] = float("nan")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rewrite_checksums(nonstandard)
    with pytest.raises(SplitError, match="non-standard numeric constant"):
        verify_splits(nonstandard)


def test_verify_rejects_source_member_semantic_tampering(tmp_path: Path) -> None:
    index = _index(tmp_path)
    directory = tmp_path / "member-tamper"
    publish_split_manifests(
        index,
        manifest_dir=directory,
        seed=20260715,
        internal_validation_per_class=1,
        enforce_official_release=False,
    )
    train_rows = _rows(directory, "train")
    row = train_rows[0]
    row["source_members"][1] = str(row["source_members"][1]).replace(
        ".npz", "_wrong.npz"
    )
    row["source_locator"] = "!".join(
        [str(row["source_archive"]), *map(str, row["source_members"])]
    )
    _write_canonical_rows(directory, "train", train_rows)
    _refresh_manifest_metadata(directory, "train")

    with pytest.raises(SplitError, match="locator semantics mismatch"):
        verify_splits(directory)


def test_partial_existing_directory_and_exact_content_duplicate_are_refused(
    tmp_path: Path,
) -> None:
    index = _index(tmp_path)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "train.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SplitError, match="exactly the immutable artifact set"):
        publish_split_manifests(
            index,
            manifest_dir=partial,
            seed=20260715,
            internal_validation_per_class=1,
            enforce_official_release=False,
        )

    samples = list(index.samples)
    samples[1] = replace(
        samples[1], raw_member_sha256=samples[0].raw_member_sha256
    )
    duplicate_index = replace(index, samples=tuple(samples))
    with pytest.raises(SplitError, match="exact duplicate raw NPZ payload"):
        publish_split_manifests(
            duplicate_index,
            manifest_dir=tmp_path / "duplicates",
            seed=20260715,
            internal_validation_per_class=1,
            enforce_official_release=False,
        )


def test_config_is_strict_and_resolves_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "splits.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_root": "../data",
                "manifest_dir": "../manifests",
                "seed": 20260715,
                "internal_validation_per_class": 50,
            }
        ),
        encoding="utf-8",
    )
    config = load_split_build_config(config_path)
    assert config.dataset_root == (tmp_path / "data").resolve()
    assert config.manifest_dir == (tmp_path / "manifests").resolve()
    assert config.seed == 20260715
    assert config.internal_validation_per_class == 50

    malformed = tmp_path / "unknown.json"
    malformed.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_root": "data",
                "manifest_dir": "manifests",
                "seed": 1,
                "internal_validation_per_class": 1,
                "random_split_at_runtime": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SplitError, match="unsupported split configuration keys"):
        load_split_build_config(malformed)

    nonstandard = tmp_path / "nonstandard.json"
    nonstandard.write_text(
        '{"schema_version":1,"dataset_root":"data","manifest_dir":"manifests",'
        '"seed":NaN,"internal_validation_per_class":1}',
        encoding="utf-8",
    )
    with pytest.raises(SplitError, match="non-standard numeric constant"):
        load_split_build_config(nonstandard)
