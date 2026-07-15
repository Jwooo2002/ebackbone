"""Synthetic provider fixtures used only to test the real-probe contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from ebackbone_v3.contracts import (
    ClassificationRecord,
    ProbeSample,
    RawEventRecord,
    RepresentationRecord,
)
from ebackbone_v3.probe import compute_event_subset_id


def load_aligned_sample() -> ProbeSample:
    fields = {
        "coord_x": torch.tensor([0, 1, 0, 1], dtype=torch.int16),
        "coord_y": torch.tensor([0, 0, 1, 1], dtype=torch.int16),
        "timestamp_us": torch.tensor([100, 101, 102, 103], dtype=torch.int64),
        "polarity_bit": torch.tensor([0, 1, 0, 1], dtype=torch.int8),
    }
    raw = RawEventRecord(
        sample_id="train/test-sample-0001",
        fields=fields,
        field_roles={
            "x": "coord_x",
            "y": "coord_y",
            "timestamp": "timestamp_us",
            "polarity": "polarity_bit",
        },
        storage_layout="test fixture: separate one-dimensional fields",
        timestamp_unit="microsecond",
        timestamp_ordering="nondecreasing",
        coordinate_convention="zero_based_xy",
        polarity_encoding="0/1",
        spatial_resolution=(2, 2),
        temporal_start=100,
        temporal_end=104,
        interval_closure="[)",
        event_count_before_filter=5,
        event_count_after_filter=4,
        filtering="test fixture drops one declared out-of-bounds event",
    )
    subset_id = compute_event_subset_id(raw.fields, raw.field_roles)
    common = {
        "source_sample_id": raw.sample_id,
        "temporal_start": raw.temporal_start,
        "temporal_end": raw.temporal_end,
        "interval_closure": raw.interval_closure,
        "event_subset_id": subset_id,
        "event_count_before_filter": raw.event_count_before_filter,
        "event_count_after_filter": raw.event_count_after_filter,
        "spatial_resolution": raw.spatial_resolution,
        "tensor_layout": "C,H,W",
        "clipping": "none",
        "padding_or_truncation": "none",
        "deterministic": True,
    }
    representations = {
        "event_frame": RepresentationRecord(
            tensor=torch.arange(8, dtype=torch.float32).reshape(2, 2, 2),
            channel_count=2,
            polarity_handling="separate test-fixture channels",
            normalization="none",
            parameters={
                "definition": "test fixture count",
                "rendered_frames": 1,
                "frame_axis": "none",
            },
            **common,
        ),
        "voxel_grid": RepresentationRecord(
            tensor=torch.linspace(-1, 1, steps=12, dtype=torch.float32).reshape(3, 2, 2),
            channel_count=3,
            polarity_handling="signed test-fixture values",
            normalization="none",
            parameters={"definition": "test fixture voxel", "bins": 3, "bin_axis": "C"},
            **common,
        ),
        "time_surface": RepresentationRecord(
            tensor=torch.linspace(0, 1, steps=8, dtype=torch.float32).reshape(2, 2, 2),
            channel_count=2,
            polarity_handling="separate test-fixture channels",
            normalization="test fixture [0,1]",
            parameters={
                "definition": "test fixture last time",
                "reference": "test fixture interval start",
                "decay": "none",
            },
            **common,
        ),
    }
    return ProbeSample(
        dataset_name="contract-test-fixture",
        split="train",
        raw_events=raw,
        representations=representations,
        classification=ClassificationRecord(
            class_id="fixture-class-b",
            class_index=1,
            class_count=2,
            class_to_index={"fixture-class-a": 0, "fixture-class-b": 1},
            mapping_policy="test-fixture lexicographic mapping",
            label_source="test-fixture bookkeeping only",
        ),
    )


def load_misaligned_sample() -> ProbeSample:
    sample = load_aligned_sample()
    representations = dict(sample.representations)
    representations["time_surface"] = replace(
        representations["time_surface"],
        temporal_end=105,
    )
    return replace(sample, representations=representations)


def load_mismatched_subset_sample() -> ProbeSample:
    sample = load_aligned_sample()
    representations = dict(sample.representations)
    representations["voxel_grid"] = replace(
        representations["voxel_grid"],
        event_subset_id="sha256:different-test-subset",
    )
    return replace(sample, representations=representations)


def load_arbitrary_shared_subset_sample() -> ProbeSample:
    sample = load_aligned_sample()
    representations = {
        name: replace(record, event_subset_id="arbitrary-shared-claim")
        for name, record in sample.representations.items()
    }
    return replace(sample, representations=representations)


def load_bad_channel_count_sample() -> ProbeSample:
    return _replace_representation("event_frame", channel_count=999)


def load_bad_spatial_resolution_sample() -> ProbeSample:
    return _replace_representation("time_surface", spatial_resolution=(99, 88))


def load_bad_layout_sample() -> ProbeSample:
    return _replace_representation("voxel_grid", tensor_layout="not-a-layout")


def load_scalar_representation_sample() -> ProbeSample:
    return _replace_representation("event_frame", tensor=torch.tensor(1.0))


def load_empty_parameters_sample() -> ProbeSample:
    return _replace_representation("event_frame", parameters={})


def load_bad_voxel_bins_sample() -> ProbeSample:
    sample = load_aligned_sample()
    parameters = dict(sample.representations["voxel_grid"].parameters)
    parameters["bins"] = 99
    return _replace_representation("voxel_grid", parameters=parameters)


def load_accumulated_frame_with_extra_axis_sample() -> ProbeSample:
    tensor = torch.arange(56, dtype=torch.float32).reshape(2, 7, 2, 2)
    return _replace_representation(
        "event_frame",
        tensor=tensor,
        tensor_layout="C,T,H,W",
    )


def load_batch_axis_sample() -> ProbeSample:
    tensor = torch.arange(16, dtype=torch.float32).reshape(2, 2, 2, 2)
    return _replace_representation(
        "event_frame",
        tensor=tensor,
        tensor_layout="B,C,H,W",
    )


def load_frame_axis_reuses_channel_sample() -> ProbeSample:
    sample = load_aligned_sample()
    parameters = dict(sample.representations["event_frame"].parameters)
    parameters.update({"rendered_frames": 2, "frame_axis": "C"})
    return _replace_representation("event_frame", parameters=parameters)


def load_voxel_axis_reuses_spatial_sample() -> ProbeSample:
    sample = load_aligned_sample()
    parameters = dict(sample.representations["voxel_grid"].parameters)
    parameters.update({"bins": 2, "bin_axis": "H"})
    return _replace_representation("voxel_grid", parameters=parameters)


def load_out_of_bounds_coordinates_sample() -> ProbeSample:
    sample = load_aligned_sample()
    fields = dict(sample.raw_events.fields)
    fields["coord_x"] = fields["coord_x"].clone()
    fields["coord_x"][0] = 500
    return replace(sample, raw_events=replace(sample.raw_events, fields=fields))


def load_duplicate_strict_timestamps_sample() -> ProbeSample:
    sample = load_aligned_sample()
    fields = dict(sample.raw_events.fields)
    fields["timestamp_us"] = torch.tensor([100, 101, 101, 103], dtype=torch.int64)
    raw = replace(sample.raw_events, fields=fields, timestamp_ordering="strictly_increasing")
    return replace(sample, raw_events=raw)


def load_wrong_polarity_encoding_sample() -> ProbeSample:
    sample = load_aligned_sample()
    return replace(
        sample,
        raw_events=replace(sample.raw_events, polarity_encoding="-1/+1"),
    )


def load_unsigned_timestamp_sample() -> ProbeSample:
    sample = load_aligned_sample()
    fields = dict(sample.raw_events.fields)
    fields["timestamp_us"] = fields["timestamp_us"].to(torch.uint32)
    raw = replace(sample.raw_events, fields=fields)
    subset_id = compute_event_subset_id(raw.fields, raw.field_roles)
    representations = {
        name: replace(record, event_subset_id=subset_id)
        for name, record in sample.representations.items()
    }
    return replace(sample, raw_events=raw, representations=representations)


def load_high_uint64_timestamp_sample() -> ProbeSample:
    sample = load_aligned_sample()
    start = 2**63 + 1
    end = start + 4
    fields = dict(sample.raw_events.fields)
    fields["timestamp_us"] = torch.tensor(
        [start, start + 1, start + 2, start + 3],
        dtype=torch.uint64,
    )
    raw = replace(
        sample.raw_events,
        fields=fields,
        timestamp_ordering="strictly_increasing",
        temporal_start=start,
        temporal_end=end,
    )
    subset_id = compute_event_subset_id(raw.fields, raw.field_roles)
    representations = {
        name: replace(
            record,
            event_subset_id=subset_id,
            temporal_start=start,
            temporal_end=end,
        )
        for name, record in sample.representations.items()
    }
    return replace(sample, raw_events=raw, representations=representations)


def load_empty_raw_sample() -> ProbeSample:
    sample = load_aligned_sample()
    fields = {name: values[:0] for name, values in sample.raw_events.fields.items()}
    raw = replace(sample.raw_events, fields=fields, event_count_after_filter=1)
    return replace(sample, raw_events=raw)


def load_nonfinite_representation_sample() -> ProbeSample:
    sample = load_aligned_sample()
    representations = dict(sample.representations)
    tensor = representations["event_frame"].tensor.clone()
    tensor[0, 0, 0] = float("nan")
    representations["event_frame"] = replace(representations["event_frame"], tensor=tensor)
    return replace(sample, representations=representations)


def load_bad_class_mapping_sample() -> ProbeSample:
    sample = load_aligned_sample()
    assert sample.classification is not None
    return replace(
        sample,
        classification=replace(sample.classification, class_index=0),
    )


def load_missing_file(sample_path: str) -> ProbeSample:
    Path(sample_path).read_bytes()
    raise AssertionError("unreachable")


def _replace_representation(name: str, **changes: object) -> ProbeSample:
    sample = load_aligned_sample()
    representations = dict(sample.representations)
    representations[name] = replace(representations[name], **changes)
    return replace(sample, representations=representations)
