"""Explicit provider-side contracts for a real-data probe.

These records describe observations supplied by a dataset-specific adapter. They
do not select a dataset, storage format, renderer, temporal window, or model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RawEventRecord:
    """One filtered raw-event sample plus its declared provenance."""

    sample_id: str
    fields: Mapping[str, Any]
    field_roles: Mapping[str, str]
    storage_layout: str
    timestamp_unit: str
    timestamp_ordering: str
    coordinate_convention: str
    polarity_encoding: str
    spatial_resolution: tuple[int, int]
    temporal_start: int | float
    temporal_end: int | float
    interval_closure: str
    event_count_before_filter: int
    event_count_after_filter: int
    filtering: str


@dataclass(frozen=True)
class RepresentationRecord:
    """One rendered tensor with enough evidence to verify sample alignment."""

    tensor: Any
    source_sample_id: str
    temporal_start: int | float
    temporal_end: int | float
    interval_closure: str
    event_subset_id: str
    event_count_before_filter: int
    event_count_after_filter: int
    spatial_resolution: tuple[int, int]
    channel_count: int
    tensor_layout: str
    polarity_handling: str
    normalization: str
    clipping: str
    padding_or_truncation: str
    deterministic: bool
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ProbeSample:
    """Complete real-sample result returned by a configured provider."""

    dataset_name: str
    split: str
    raw_events: RawEventRecord
    representations: Mapping[str, RepresentationRecord]


__all__ = ["ProbeSample", "RawEventRecord", "RepresentationRecord"]
