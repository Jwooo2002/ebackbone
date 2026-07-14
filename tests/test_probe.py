from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ebackbone_v3.errors import ProbeError
from ebackbone_v3.probe import run_probe


def _write_config(tmp_path: Path, callable_name: str, **kwargs: object) -> Path:
    path = tmp_path / "probe.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider": {
                    "callable": f"tests.probe_fixture_provider:{callable_name}",
                    "kwargs": kwargs,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_aligned_provider_reports_full_contract(tmp_path: Path) -> None:
    config = _write_config(tmp_path, "load_aligned_sample")
    report = run_probe(config)
    assert report["status"] == "PASS"
    assert report["command"] == "probe"
    assert report["config"]["path"] == str(config.resolve())
    assert report["config"]["sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert report["dataset"] == {"name": "contract-test-fixture", "split": "train"}
    assert report["sample_id"] == "train/test-sample-0001"
    raw = report["raw_events"]
    assert set(raw["fields"]) == {"coord_x", "coord_y", "timestamp_us", "polarity_bit"}
    assert raw["fields"]["timestamp_us"]["dtype"] == "int64"
    assert raw["event_count"] == 4
    assert raw["event_count_before_filter"] == 5
    assert raw["event_count_after_filter"] == 4
    assert raw["timestamp"]["observed_start"] == 100
    assert raw["timestamp"]["observed_end"] == 103
    assert raw["timestamp"]["unit"] == "microsecond"
    assert report["representations"]["event_frame"]["shape"] == [2, 2, 2]
    assert report["representations"]["voxel_grid"]["shape"] == [3, 2, 2]
    assert report["representations"]["time_surface"]["shape"] == [2, 2, 2]
    assert report["representations"]["voxel_grid"]["value_range"] == {
        "min": -1.0,
        "max": 1.0,
    }
    assert raw["decoded_shape"] == [4, 4]
    assert raw["coordinates"]["bounds_verified"] is True
    assert raw["polarity"]["encoding_verified"] is True
    assert raw["timestamp"]["observed_ordering"] == "strictly_increasing"
    assert report["temporal_alignment"]["status"] == "VERIFIED_FROM_PROVIDER_EVIDENCE"
    subset_id = report["temporal_alignment"]["evidence"]["event_subset_id"]
    assert subset_id.startswith("sha256:")
    assert subset_id == raw["event_subset_id"]
    assert all(
        representation["provenance"]["event_subset_id"] == subset_id
        for representation in report["representations"].values()
    )


@pytest.mark.parametrize(
    ("provider", "error_text"),
    [
        ("load_misaligned_sample", "temporal alignment failed for time_surface"),
        ("load_mismatched_subset_sample", "event subset provenance failed for voxel_grid"),
        ("load_arbitrary_shared_subset_sample", "event subset provenance failed for event_frame"),
        ("load_empty_raw_sample", "raw_events.fields.coord_x must not be empty"),
        ("load_nonfinite_representation_sample", "representations.event_frame.tensor contains non-finite"),
        ("load_bad_channel_count_sample", "channel_count=999 does not match C axis size"),
        ("load_bad_spatial_resolution_sample", "spatial_resolution=.*does not match H/W"),
        ("load_bad_layout_sample", "tensor_layout must be comma-separated"),
        ("load_scalar_representation_sample", "tensor_layout has 3 axes but tensor rank is 0"),
        ("load_empty_parameters_sample", "parameters must not be empty"),
        ("load_bad_voxel_bins_sample", "parameters.bins=99 does not match axis C size"),
        (
            "load_accumulated_frame_with_extra_axis_sample",
            "frame_axis='none' is incompatible with extra tensor axes",
        ),
        ("load_batch_axis_sample", "cannot contain batch axes"),
        ("load_frame_axis_reuses_channel_sample", "frame_axis cannot reuse channel or spatial"),
        ("load_voxel_axis_reuses_spatial_sample", "bin_axis cannot reuse spatial axis"),
        ("load_out_of_bounds_coordinates_sample", "raw coordinates violate zero_based_xy bounds"),
        ("load_duplicate_strict_timestamps_sample", "timestamp ordering does not match declaration"),
        ("load_wrong_polarity_encoding_sample", "do not match declared encoding"),
    ],
)
def test_invalid_provider_result_fails_actionably(
    tmp_path: Path,
    provider: str,
    error_text: str,
) -> None:
    config = _write_config(tmp_path, provider)
    with pytest.raises(ProbeError, match=error_text):
        run_probe(config)


def test_missing_config_reports_resolved_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ProbeError) as caught:
        run_probe(missing)
    assert str(missing.resolve()) in str(caught.value)
    assert "does not exist" in str(caught.value)


def test_malformed_config_reports_json_location(tmp_path: Path) -> None:
    config = tmp_path / "bad.json"
    config.write_text('{"schema_version": 1,', encoding="utf-8")
    with pytest.raises(ProbeError, match=r"invalid JSON at line 1 column"):
        run_probe(config)


def test_tbd_config_lists_dotted_keys(tmp_path: Path) -> None:
    config = tmp_path / "tbd.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider": {
                    "callable": "TBD",
                    "kwargs": {"dataset_root": "TBD"},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProbeError) as caught:
        run_probe(config)
    message = str(caught.value)
    assert "provider.callable" in message
    assert "provider.kwargs.dataset_root" in message


def test_provider_missing_file_reports_exact_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing-sample.aedat"
    config = _write_config(tmp_path, "load_missing_file", sample_path=str(missing))
    with pytest.raises(ProbeError) as caught:
        run_probe(config)
    message = str(caught.value)
    assert str(missing) in message
    assert "could not find required file" in message


def test_unsigned_raw_timestamp_dtype_is_reported_and_summarized(tmp_path: Path) -> None:
    config = _write_config(tmp_path, "load_unsigned_timestamp_sample")
    report = run_probe(config)
    timestamp = report["raw_events"]["fields"]["timestamp_us"]
    assert timestamp["dtype"] == "uint32"
    assert timestamp["value_range"] == {"min": 100, "max": 103}


def test_high_uint64_timestamps_remain_exact_and_ordered(tmp_path: Path) -> None:
    config = _write_config(tmp_path, "load_high_uint64_timestamp_sample")
    report = run_probe(config)
    start = 2**63 + 1
    timestamp = report["raw_events"]["timestamp"]
    assert report["raw_events"]["fields"]["timestamp_us"]["dtype"] == "uint64"
    assert timestamp["observed_start"] == start
    assert timestamp["observed_end"] == start + 3
    assert timestamp["observed_ordering"] == "strictly_increasing"


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_json_constants_are_rejected(tmp_path: Path, constant: str) -> None:
    config = tmp_path / "nonstandard.json"
    config.write_text(
        '{"schema_version": 1, "provider": {"callable": '
        '"tests.probe_fixture_provider:load_aligned_sample", "kwargs": {}}, '
        f'"invalid": {constant}}}',
        encoding="utf-8",
    )
    with pytest.raises(ProbeError, match="non-standard numeric constant"):
        run_probe(config)


def test_unknown_top_level_config_key_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "unknown.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider": {
                    "callable": "tests.probe_fixture_provider:load_aligned_sample",
                    "kwargs": {},
                },
                "typo": "value",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProbeError, match="unsupported top-level keys: typo"):
        run_probe(config)
