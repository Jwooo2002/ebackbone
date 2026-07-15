"""Provider-based real-data inspection without implicit dataset assumptions."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from ebackbone_v3.contracts import (
    ClassificationRecord,
    ProbeSample,
    RawEventRecord,
    RepresentationRecord,
)
from ebackbone_v3.errors import ProbeError


PROBE_SCHEMA_VERSION = 1
REPRESENTATION_NAMES = ("event_frame", "voxel_grid", "time_surface")
FIELD_ROLES = ("x", "y", "timestamp", "polarity")
INTERVAL_CLOSURES = ("[)", "[]", "()", "(]")
TIMESTAMP_ORDERINGS = ("nondecreasing", "strictly_increasing", "not_sorted")
COORDINATE_CONVENTIONS = ("zero_based_xy", "one_based_xy")
POLARITY_ENCODINGS = ("0/1", "-1/+1", "false/true")
LAYOUT_TOKEN_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def run_probe(config_path: str | Path) -> dict[str, Any]:
    """Load one configured real sample and return a verified JSON-ready report."""

    path, config, config_sha256 = _load_config(config_path)
    provider_spec, provider_kwargs = _provider_config(config)
    provider = _load_provider(provider_spec)
    try:
        supplied = provider(**provider_kwargs)
    except ProbeError:
        raise
    except FileNotFoundError as exc:
        missing = exc.filename if exc.filename is not None else str(exc)
        raise ProbeError(f"probe provider could not find required file: {missing}") from exc
    except TypeError as exc:
        raise ProbeError(f"probe provider invocation failed for {provider_spec}: {exc}") from exc
    sample = _coerce_probe_sample(supplied)
    report = _build_report(sample)
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "PASS",
        "command": "probe",
        "config": {"path": str(path), "sha256": config_sha256},
        "provider": {"callable": provider_spec},
        **report,
    }


def compute_event_subset_id(
    fields: Mapping[str, Any],
    field_roles: Mapping[str, str],
) -> str:
    """Return a canonical SHA-256 identity for post-filter x/y/timestamp/polarity fields."""

    fields_mapping = _require_mapping(fields, "raw_events.fields")
    roles_mapping = _require_mapping(field_roles, "raw_events.field_roles")
    digest = hashlib.sha256()
    digest.update(b"ebackbone_v3:event_subset:v1\n")
    for role in FIELD_ROLES:
        if role not in roles_mapping:
            raise ProbeError(f"raw_events.field_roles is missing role: {role}")
        field_name = _require_text(roles_mapping[role], f"raw_events.field_roles.{role}")
        if field_name not in fields_mapping:
            raise ProbeError(
                f"raw_events.field_roles.{role} references missing raw field {field_name!r}"
            )
        tensor = _numeric_tensor(fields_mapping[field_name], f"raw_events.fields.{field_name}")
        if tensor.ndim != 1:
            raise ProbeError(
                f"raw_events.fields.{field_name} must be one-dimensional; got shape {tuple(tensor.shape)}"
            )
        digest.update(role.encode("utf-8"))
        digest.update(b"\0")
        digest.update(field_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        byte_view = tensor.detach().cpu().clone(memory_format=torch.contiguous_format).view(torch.uint8)
        digest.update(bytes(byte_view.untyped_storage()))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard numeric constant {value!r} is not permitted")


def _load_config(config_path: str | Path) -> tuple[Path, Mapping[str, Any], str]:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ProbeError(f"probe config does not exist: {path}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProbeError(f"could not read probe config {path}: {exc}") from exc
    try:
        config = json.loads(payload, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ProbeError(
            f"probe config is invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}"
        ) from exc
    except ValueError as exc:
        raise ProbeError(f"probe config is invalid JSON: {exc}") from exc
    if not isinstance(config, Mapping):
        raise ProbeError("probe config root must be a JSON object")
    unexpected_config_keys = sorted(set(config) - {"schema_version", "provider"})
    if unexpected_config_keys:
        raise ProbeError(
            f"probe config has unsupported top-level keys: {', '.join(unexpected_config_keys)}"
        )
    unresolved = _find_tbd(config)
    if unresolved:
        raise ProbeError(f"probe configuration contains unresolved values: {', '.join(unresolved)}")
    schema_version = config.get("schema_version")
    if schema_version != PROBE_SCHEMA_VERSION or isinstance(schema_version, bool):
        raise ProbeError(
            f"probe config key schema_version must equal {PROBE_SCHEMA_VERSION}; got {schema_version!r}"
        )
    return path, config, hashlib.sha256(payload).hexdigest()


def _provider_config(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    provider = config.get("provider")
    if not isinstance(provider, Mapping):
        raise ProbeError("probe config key provider must be an object")
    unexpected_provider_keys = sorted(set(provider) - {"callable", "kwargs"})
    if unexpected_provider_keys:
        raise ProbeError(
            "probe config key provider has unsupported keys: "
            f"{', '.join(unexpected_provider_keys)}"
        )
    provider_spec = provider.get("callable")
    if not isinstance(provider_spec, str) or not provider_spec.strip():
        raise ProbeError("probe config key provider.callable must be a non-empty 'module:function' string")
    kwargs = provider.get("kwargs", {})
    if not isinstance(kwargs, Mapping):
        raise ProbeError("probe config key provider.kwargs must be an object")
    return provider_spec.strip(), dict(kwargs)


def _load_provider(provider_spec: str) -> Callable[..., object]:
    module_name, separator, attribute_path = provider_spec.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ProbeError(
            f"invalid provider.callable {provider_spec!r}; expected a 'module:function' import path"
        )
    try:
        target: object = importlib.import_module(module_name)
    except Exception as exc:
        raise ProbeError(f"could not import probe provider module {module_name!r}: {exc}") from exc
    for attribute in attribute_path.split("."):
        if not attribute:
            raise ProbeError(f"invalid provider.callable attribute path: {provider_spec!r}")
        try:
            target = getattr(target, attribute)
        except AttributeError as exc:
            raise ProbeError(f"probe provider callable does not exist: {provider_spec}") from exc
    if not callable(target):
        raise ProbeError(f"probe provider target is not callable: {provider_spec}")
    return target


def _coerce_probe_sample(value: object) -> ProbeSample:
    if isinstance(value, ProbeSample):
        return value
    root = _require_mapping(value, "provider result")
    raw_value = _required(root, "raw_events", "provider result.raw_events")
    raw = raw_value if isinstance(raw_value, RawEventRecord) else _coerce_raw_event_record(raw_value)
    representations_value = _require_mapping(
        _required(root, "representations", "provider result.representations"),
        "provider result.representations",
    )
    representations: dict[str, RepresentationRecord] = {}
    for name, record in representations_value.items():
        if not isinstance(name, str):
            raise ProbeError("provider result.representations keys must be strings")
        representations[name] = (
            record if isinstance(record, RepresentationRecord) else _coerce_representation_record(record, name)
        )
    classification_value = root.get("classification")
    classification = (
        None
        if classification_value is None
        else classification_value
        if isinstance(classification_value, ClassificationRecord)
        else _coerce_classification_record(classification_value)
    )
    return ProbeSample(
        dataset_name=_required(root, "dataset_name", "provider result.dataset_name"),
        split=_required(root, "split", "provider result.split"),
        raw_events=raw,
        representations=representations,
        classification=classification,
    )


def _coerce_raw_event_record(value: object) -> RawEventRecord:
    record = _require_mapping(value, "provider result.raw_events")
    kwargs = {field: _required(record, field, f"provider result.raw_events.{field}") for field in RawEventRecord.__dataclass_fields__}
    try:
        return RawEventRecord(**kwargs)
    except TypeError as exc:
        raise ProbeError(f"invalid provider result.raw_events: {exc}") from exc


def _coerce_representation_record(value: object, name: str) -> RepresentationRecord:
    key = f"provider result.representations.{name}"
    record = _require_mapping(value, key)
    kwargs = {field: _required(record, field, f"{key}.{field}") for field in RepresentationRecord.__dataclass_fields__}
    try:
        return RepresentationRecord(**kwargs)
    except TypeError as exc:
        raise ProbeError(f"invalid {key}: {exc}") from exc


def _coerce_classification_record(value: object) -> ClassificationRecord:
    key = "provider result.classification"
    record = _require_mapping(value, key)
    kwargs = {
        field: _required(record, field, f"{key}.{field}")
        for field in ClassificationRecord.__dataclass_fields__
    }
    try:
        return ClassificationRecord(**kwargs)
    except TypeError as exc:
        raise ProbeError(f"invalid {key}: {exc}") from exc


def _build_report(sample: ProbeSample) -> dict[str, Any]:
    dataset_name = _require_text(sample.dataset_name, "provider result.dataset_name")
    split = _require_text(sample.split, "provider result.split")
    raw_report, raw_facts = _summarize_raw(sample.raw_events)
    representation_keys = set(sample.representations)
    expected_keys = set(REPRESENTATION_NAMES)
    missing = sorted(expected_keys - representation_keys)
    unexpected = sorted(representation_keys - expected_keys)
    if missing:
        raise ProbeError(f"provider result is missing representations: {', '.join(missing)}")
    if unexpected:
        raise ProbeError(f"provider result has unsupported representations: {', '.join(unexpected)}")

    representation_reports: dict[str, Any] = {}
    for name in REPRESENTATION_NAMES:
        report = _summarize_representation(name, sample.representations[name], raw_facts)
        representation_reports[name] = report
    report = {
        "dataset": {"name": dataset_name, "split": split},
        "sample_id": raw_facts["sample_id"],
        "raw_events": raw_report,
        "representations": representation_reports,
        "temporal_alignment": {
            "status": "VERIFIED_FROM_PROVIDER_EVIDENCE",
            "verification_scope": (
                "the generic probe verified metadata equality and bound every representation's "
                "provider attestation to the canonical post-filter raw-event fingerprint; the "
                "dataset-specific provider remains responsible for renderer correctness"
            ),
            "evidence": {
                "source_sample_id": raw_facts["sample_id"],
                "temporal_start": raw_facts["temporal_start"],
                "temporal_end": raw_facts["temporal_end"],
                "interval_closure": raw_facts["interval_closure"],
                "event_subset_id": raw_facts["event_subset_id"],
                "event_count_before_filter": raw_facts["event_count_before_filter"],
                "event_count_after_filter": raw_facts["event_count_after_filter"],
                "criteria": [
                    "matching source sample identity",
                    "matching temporal interval and closure",
                    "matching event counts before and after filtering",
                    "representation event-subset attestations match the canonical raw-event fingerprint",
                ],
            },
        },
    }
    if sample.classification is not None:
        report["classification"] = _summarize_classification(sample.classification)
    return report


def _summarize_classification(classification: ClassificationRecord) -> dict[str, Any]:
    key = "classification"
    class_id = _require_text(classification.class_id, f"{key}.class_id")
    class_index = _nonnegative_int(classification.class_index, f"{key}.class_index")
    class_count = _positive_int(classification.class_count, f"{key}.class_count")
    if class_index >= class_count:
        raise ProbeError(
            f"{key}.class_index={class_index} must be smaller than class_count={class_count}"
        )
    supplied_mapping = _require_mapping(classification.class_to_index, f"{key}.class_to_index")
    class_to_index: dict[str, int] = {}
    for supplied_class_id, supplied_index in supplied_mapping.items():
        mapped_class_id = _require_text(supplied_class_id, f"{key}.class_to_index key")
        if mapped_class_id in class_to_index:
            raise ProbeError(f"{key}.class_to_index contains duplicate class id {mapped_class_id!r}")
        class_to_index[mapped_class_id] = _nonnegative_int(
            supplied_index,
            f"{key}.class_to_index.{mapped_class_id}",
        )
    if len(class_to_index) != class_count:
        raise ProbeError(
            f"{key}.class_to_index has {len(class_to_index)} entries but class_count={class_count}"
        )
    if set(class_to_index.values()) != set(range(class_count)):
        raise ProbeError(f"{key}.class_to_index values must be exactly 0..{class_count - 1}")
    if class_to_index.get(class_id) != class_index:
        raise ProbeError(
            f"{key}.class_id/index pair does not match class_to_index: "
            f"{class_id!r} -> {class_to_index.get(class_id)!r}, declared {class_index}"
        )
    mapping_policy = _require_text(classification.mapping_policy, f"{key}.mapping_policy")
    label_source = _require_text(classification.label_source, f"{key}.label_source")
    unresolved = _find_tbd(
        {
            "class_id": class_id,
            "mapping_policy": mapping_policy,
            "label_source": label_source,
        },
        prefix=key,
    )
    if unresolved:
        raise ProbeError(f"classification metadata contains unresolved values: {', '.join(unresolved)}")
    return {
        "class_id": class_id,
        "class_index": class_index,
        "class_count": class_count,
        "class_to_index": dict(sorted(class_to_index.items())),
        "mapping_policy": mapping_policy,
        "label_source": label_source,
    }


def _summarize_raw(raw: RawEventRecord) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = _require_text(raw.sample_id, "raw_events.sample_id")
    fields = _require_mapping(raw.fields, "raw_events.fields")
    if not fields:
        raise ProbeError("raw_events.fields must not be empty")
    roles = _require_mapping(raw.field_roles, "raw_events.field_roles")
    missing_roles = [role for role in FIELD_ROLES if role not in roles]
    if missing_roles:
        raise ProbeError(f"raw_events.field_roles is missing roles: {', '.join(missing_roles)}")
    role_fields: dict[str, str] = {}
    for role in FIELD_ROLES:
        field_name = _require_text(roles[role], f"raw_events.field_roles.{role}")
        if field_name not in fields:
            raise ProbeError(
                f"raw_events.field_roles.{role} references missing raw field {field_name!r}"
            )
        role_fields[role] = field_name
    if len(set(role_fields.values())) != len(FIELD_ROLES):
        raise ProbeError("raw_events.field_roles must map x, y, timestamp, and polarity to distinct fields")

    field_reports: dict[str, Any] = {}
    field_tensors: dict[str, Tensor] = {}
    event_counts: set[int] = set()
    for field_name, values in fields.items():
        name = _require_text(field_name, "raw_events.fields key")
        tensor = _numeric_tensor(values, f"raw_events.fields.{name}")
        if tensor.ndim != 1:
            raise ProbeError(
                f"raw_events.fields.{name} must be one-dimensional; got shape {tuple(tensor.shape)}"
            )
        event_counts.add(int(tensor.shape[0]))
        field_tensors[name] = tensor
        field_reports[name] = _tensor_report(tensor)
    if len(event_counts) != 1:
        detail = ", ".join(f"{name}={tensor.shape[0]}" for name, tensor in field_tensors.items())
        raise ProbeError(f"raw event fields have inconsistent event counts: {detail}")
    observed_event_count = next(iter(event_counts))
    if observed_event_count <= 0:
        raise ProbeError("raw event fields contain no events")

    before = _positive_int(raw.event_count_before_filter, "raw_events.event_count_before_filter")
    after = _positive_int(raw.event_count_after_filter, "raw_events.event_count_after_filter")
    if before < after:
        raise ProbeError("raw_events.event_count_before_filter must be >= event_count_after_filter")
    if after != observed_event_count:
        raise ProbeError(
            "raw_events.event_count_after_filter does not match observed field length: "
            f"{after} vs {observed_event_count}"
        )

    temporal_start = _number(raw.temporal_start, "raw_events.temporal_start")
    temporal_end = _number(raw.temporal_end, "raw_events.temporal_end")
    if temporal_end <= temporal_start:
        raise ProbeError("raw_events.temporal_end must be greater than temporal_start")
    closure = _interval_closure(raw.interval_closure, "raw_events.interval_closure")
    timestamp_tensor = field_tensors[role_fields["timestamp"]]
    timestamp_start, timestamp_end = _tensor_extrema(timestamp_tensor)
    if not _interval_contains(timestamp_start, temporal_start, temporal_end, closure):
        raise ProbeError(
            f"raw timestamp minimum {timestamp_start!r} lies outside declared interval "
            f"{closure} {temporal_start!r}, {temporal_end!r}"
        )
    if not _interval_contains(timestamp_end, temporal_start, temporal_end, closure):
        raise ProbeError(
            f"raw timestamp maximum {timestamp_end!r} lies outside declared interval "
            f"{closure} {temporal_start!r}, {temporal_end!r}"
        )
    resolution = _resolution(raw.spatial_resolution, "raw_events.spatial_resolution")
    ordering_report = _validate_timestamp_ordering(timestamp_tensor, raw.timestamp_ordering)
    coordinate_report = _validate_coordinates(
        field_tensors[role_fields["x"]],
        field_tensors[role_fields["y"]],
        resolution,
        raw.coordinate_convention,
    )
    polarity_report = _validate_polarity(
        field_tensors[role_fields["polarity"]],
        raw.polarity_encoding,
    )
    event_subset_id = compute_event_subset_id(fields, role_fields)
    report = {
        "fields": field_reports,
        "field_roles": role_fields,
        "decoded_shape": [observed_event_count, len(field_tensors)],
        "storage_layout": _require_text(raw.storage_layout, "raw_events.storage_layout"),
        "event_count": observed_event_count,
        "event_count_before_filter": before,
        "event_count_after_filter": after,
        "filtering": _require_text(raw.filtering, "raw_events.filtering"),
        "spatial_resolution": list(resolution),
        "coordinates": coordinate_report,
        "event_subset_id": event_subset_id,
        "timestamp": {
            "field": role_fields["timestamp"],
            "unit": _require_text(raw.timestamp_unit, "raw_events.timestamp_unit"),
            **ordering_report,
            "observed_start": timestamp_start,
            "observed_end": timestamp_end,
            "observed_duration": _python_number(timestamp_end - timestamp_start),
        },
        "polarity": {
            "field": role_fields["polarity"],
            **polarity_report,
        },
        "temporal_interval": {
            "start": temporal_start,
            "end": temporal_end,
            "duration": _python_number(temporal_end - temporal_start),
            "closure": closure,
        },
    }
    facts = {
        "sample_id": sample_id,
        "temporal_start": temporal_start,
        "temporal_end": temporal_end,
        "interval_closure": closure,
        "event_count_before_filter": before,
        "event_count_after_filter": after,
        "event_subset_id": event_subset_id,
    }
    return report, facts


def _summarize_representation(
    name: str,
    representation: RepresentationRecord,
    raw_facts: Mapping[str, Any],
) -> dict[str, Any]:
    key = f"representations.{name}"
    tensor = _numeric_tensor(representation.tensor, f"{key}.tensor")
    source_sample_id = _require_text(representation.source_sample_id, f"{key}.source_sample_id")
    if source_sample_id != raw_facts["sample_id"]:
        raise ProbeError(
            f"cross-representation sample identity mismatch for {name}: "
            f"{source_sample_id!r} vs raw sample {raw_facts['sample_id']!r}"
        )
    temporal_start = _number(representation.temporal_start, f"{key}.temporal_start")
    temporal_end = _number(representation.temporal_end, f"{key}.temporal_end")
    closure = _interval_closure(representation.interval_closure, f"{key}.interval_closure")
    for field, observed, expected in (
        ("temporal_start", temporal_start, raw_facts["temporal_start"]),
        ("temporal_end", temporal_end, raw_facts["temporal_end"]),
        ("interval_closure", closure, raw_facts["interval_closure"]),
    ):
        if observed != expected:
            raise ProbeError(
                f"temporal alignment failed for {name}: {field}={observed!r} "
                f"does not match raw_events.{field}={expected!r}"
            )
    before = _positive_int(representation.event_count_before_filter, f"{key}.event_count_before_filter")
    after = _positive_int(representation.event_count_after_filter, f"{key}.event_count_after_filter")
    if before != raw_facts["event_count_before_filter"] or after != raw_facts["event_count_after_filter"]:
        raise ProbeError(
            f"event-count provenance failed for {name}: before/after=({before}, {after}) "
            "does not match raw_events"
        )
    subset_id = _require_text(representation.event_subset_id, f"{key}.event_subset_id")
    if subset_id != raw_facts["event_subset_id"]:
        raise ProbeError(
            f"event subset provenance failed for {name}: event_subset_id does not match "
            "the canonical post-filter raw-event fingerprint"
        )
    parameters = _require_mapping(representation.parameters, f"{key}.parameters")
    unresolved = _find_tbd(parameters, prefix=f"{key}.parameters")
    if unresolved:
        raise ProbeError(f"representation metadata contains unresolved values: {', '.join(unresolved)}")
    resolution = _resolution(representation.spatial_resolution, f"{key}.spatial_resolution")
    channel_count = _positive_int(representation.channel_count, f"{key}.channel_count")
    layout, axis_sizes = _validated_layout(
        representation.tensor_layout,
        tensor,
        resolution,
        channel_count,
        key,
    )
    validated_parameters = _validate_renderer_parameters(name, parameters, axis_sizes, key)
    if not isinstance(representation.deterministic, bool):
        raise ProbeError(f"{key}.deterministic must be a boolean")
    report = {
        **_tensor_report(tensor),
        "tensor_layout": layout,
        "spatial_resolution": list(resolution),
        "channel_count": channel_count,
        "polarity_handling": _require_text(
            representation.polarity_handling, f"{key}.polarity_handling"
        ),
        "normalization": _require_text(representation.normalization, f"{key}.normalization"),
        "clipping": _require_text(representation.clipping, f"{key}.clipping"),
        "padding_or_truncation": _require_text(
            representation.padding_or_truncation, f"{key}.padding_or_truncation"
        ),
        "deterministic": representation.deterministic,
        "parameters": validated_parameters,
        "provenance": {
            "source_sample_id": source_sample_id,
            "temporal_start": temporal_start,
            "temporal_end": temporal_end,
            "interval_closure": closure,
            "event_subset_id": subset_id,
            "event_count_before_filter": before,
            "event_count_after_filter": after,
        },
    }
    return report


def _numeric_tensor(value: object, key: str) -> Tensor:
    try:
        tensor = value.detach().cpu() if isinstance(value, Tensor) else torch.as_tensor(value)
    except Exception as exc:
        raise ProbeError(f"{key} must be convertible to a numeric tensor: {exc}") from exc
    if tensor.numel() == 0:
        raise ProbeError(f"{key} must not be empty")
    if tensor.is_complex():
        raise ProbeError(f"{key} must not use a complex dtype")
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
        raise ProbeError(f"{key} contains non-finite values")
    return tensor


def _tensor_report(tensor: Tensor) -> dict[str, Any]:
    minimum, maximum = _tensor_extrema(tensor)
    return {
        "shape": [int(size) for size in tensor.shape],
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "value_range": {
            "min": minimum,
            "max": maximum,
        },
    }


def _analysis_tensor(tensor: Tensor) -> Tensor:
    if str(tensor.dtype) in {"torch.uint16", "torch.uint32"}:
        return tensor.to(torch.int64)
    return tensor


def _tensor_extrema(tensor: Tensor) -> tuple[int | float, int | float]:
    if str(tensor.dtype) == "torch.uint64":
        values = tensor.detach().cpu().numpy()
        return _python_number(values.min().item()), _python_number(values.max().item())
    analysis = _analysis_tensor(tensor)
    return _python_number(analysis.min().item()), _python_number(analysis.max().item())


def _validate_timestamp_ordering(timestamps: Tensor, declaration: object) -> dict[str, Any]:
    ordering = _require_text(declaration, "raw_events.timestamp_ordering").lower()
    if ordering not in TIMESTAMP_ORDERINGS:
        raise ProbeError(
            "raw_events.timestamp_ordering must be one of "
            f"{', '.join(TIMESTAMP_ORDERINGS)}"
        )
    values = _numpy_values(timestamps, "raw timestamp field")
    nondecreasing = bool(values.size < 2 or np.all(values[1:] >= values[:-1]))
    strictly_increasing = bool(values.size < 2 or np.all(values[1:] > values[:-1]))
    declaration_matches = {
        "nondecreasing": nondecreasing,
        "strictly_increasing": strictly_increasing,
        "not_sorted": not nondecreasing,
    }[ordering]
    if not declaration_matches:
        raise ProbeError(f"raw timestamp ordering does not match declaration {ordering!r}")
    observed = (
        "strictly_increasing"
        if strictly_increasing
        else "nondecreasing"
        if nondecreasing
        else "not_sorted"
    )
    return {
        "declared_ordering": ordering,
        "observed_ordering": observed,
        "observed_nondecreasing": nondecreasing,
        "observed_strictly_increasing": strictly_increasing,
    }


def _validate_coordinates(
    x_tensor: Tensor,
    y_tensor: Tensor,
    resolution: tuple[int, int],
    declaration: object,
) -> dict[str, Any]:
    convention = _require_text(declaration, "raw_events.coordinate_convention").lower()
    if convention not in COORDINATE_CONVENTIONS:
        raise ProbeError(
            "raw_events.coordinate_convention must be one of "
            f"{', '.join(COORDINATE_CONVENTIONS)}"
        )
    x = _numpy_values(x_tensor, "raw x coordinate field")
    y = _numpy_values(y_tensor, "raw y coordinate field")
    if not bool(np.all(x == np.rint(x))):
        raise ProbeError("raw x coordinates must be integer-valued")
    if not bool(np.all(y == np.rint(y))):
        raise ProbeError("raw y coordinates must be integer-valued")
    height, width = resolution
    if convention == "zero_based_xy":
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        expected_bounds = {"x": [0, width - 1], "y": [0, height - 1]}
    else:
        valid = (x >= 1) & (x <= width) & (y >= 1) & (y <= height)
        expected_bounds = {"x": [1, width], "y": [1, height]}
    if not bool(np.all(valid)):
        invalid_count = int(np.count_nonzero(~valid))
        raise ProbeError(
            f"raw coordinates violate {convention} bounds for spatial_resolution={resolution}; "
            f"invalid_event_count={invalid_count}"
        )
    return {
        "convention": convention,
        "bounds_verified": True,
        "expected_bounds": expected_bounds,
    }


def _validate_polarity(polarity_tensor: Tensor, declaration: object) -> dict[str, Any]:
    encoding = _require_text(declaration, "raw_events.polarity_encoding").lower()
    if encoding not in POLARITY_ENCODINGS:
        raise ProbeError(
            f"raw_events.polarity_encoding must be one of {', '.join(POLARITY_ENCODINGS)}"
        )
    values = _numpy_values(polarity_tensor, "raw polarity field")
    observed_values = [_python_number(value.item()) for value in np.unique(values)]
    if encoding == "false/true":
        matches = polarity_tensor.dtype == torch.bool
    else:
        allowed = {0, 1} if encoding == "0/1" else {-1, 1}
        matches = set(observed_values).issubset(allowed)
    if not matches:
        raise ProbeError(
            f"raw polarity values {observed_values!r} do not match declared encoding {encoding!r}"
        )
    return {
        "encoding": encoding,
        "observed_values": observed_values,
        "encoding_verified": True,
    }


def _numpy_values(tensor: Tensor, key: str) -> np.ndarray[Any, Any]:
    try:
        return tensor.detach().cpu().numpy()
    except (TypeError, RuntimeError) as exc:
        raise ProbeError(f"{key} dtype {tensor.dtype} cannot be analyzed exactly: {exc}") from exc


def _validated_layout(
    declaration: object,
    tensor: Tensor,
    resolution: tuple[int, int],
    channel_count: int,
    key: str,
) -> tuple[str, dict[str, int]]:
    layout_text = _require_text(declaration, f"{key}.tensor_layout")
    axes = tuple(axis.strip().upper() for axis in layout_text.split(","))
    if any(not axis or not LAYOUT_TOKEN_PATTERN.fullmatch(axis) for axis in axes):
        raise ProbeError(
            f"{key}.tensor_layout must be comma-separated axis names such as 'C,H,W' or 'C,T,H,W'"
        )
    if len(axes) != tensor.ndim:
        raise ProbeError(
            f"{key}.tensor_layout has {len(axes)} axes but tensor rank is {tensor.ndim}"
        )
    if len(set(axes)) != len(axes):
        raise ProbeError(f"{key}.tensor_layout axis names must be unique")
    batch_axes = [axis for axis in axes if axis in {"B", "N", "BATCH"}]
    if batch_axes:
        raise ProbeError(
            f"{key}.tensor_layout must describe one sample and cannot contain batch axes: "
            f"{', '.join(batch_axes)}"
        )
    missing = [axis for axis in ("C", "H", "W") if axis not in axes]
    if missing:
        raise ProbeError(f"{key}.tensor_layout is missing required axes: {', '.join(missing)}")
    axis_sizes = {axis: int(tensor.shape[index]) for index, axis in enumerate(axes)}
    if axis_sizes["C"] != channel_count:
        raise ProbeError(
            f"{key}.channel_count={channel_count} does not match C axis size {axis_sizes['C']}"
        )
    height, width = resolution
    if axis_sizes["H"] != height or axis_sizes["W"] != width:
        raise ProbeError(
            f"{key}.spatial_resolution={resolution} does not match H/W axis sizes "
            f"({axis_sizes['H']}, {axis_sizes['W']})"
        )
    return ",".join(axes), axis_sizes


def _validate_renderer_parameters(
    name: str,
    parameters: Mapping[str, Any],
    axis_sizes: Mapping[str, int],
    key: str,
) -> dict[str, Any]:
    if not parameters:
        raise ProbeError(f"{key}.parameters must not be empty")
    _require_text(
        _required(parameters, "definition", f"{key}.parameters.definition"),
        f"{key}.parameters.definition",
    )
    if name == "event_frame":
        rendered_frames = _positive_int(
            _required(parameters, "rendered_frames", f"{key}.parameters.rendered_frames"),
            f"{key}.parameters.rendered_frames",
        )
        frame_axis = _require_text(
            _required(parameters, "frame_axis", f"{key}.parameters.frame_axis"),
            f"{key}.parameters.frame_axis",
        ).upper()
        extra_axes = set(axis_sizes) - {"C", "H", "W"}
        if frame_axis == "NONE":
            if rendered_frames != 1:
                raise ProbeError(
                    f"{key}.parameters.frame_axis='none' requires rendered_frames=1"
                )
            if extra_axes:
                raise ProbeError(
                    f"{key}.parameters.frame_axis='none' is incompatible with extra tensor axes: "
                    f"{', '.join(sorted(extra_axes))}"
                )
        elif frame_axis in {"C", "H", "W"}:
            raise ProbeError(
                f"{key}.parameters.frame_axis cannot reuse channel or spatial axis {frame_axis!r}"
            )
        elif frame_axis not in axis_sizes:
            raise ProbeError(f"{key}.parameters.frame_axis={frame_axis!r} is not in tensor_layout")
        elif axis_sizes[frame_axis] != rendered_frames:
            raise ProbeError(
                f"{key}.parameters.rendered_frames={rendered_frames} does not match "
                f"axis {frame_axis} size {axis_sizes[frame_axis]}"
            )
        elif extra_axes != {frame_axis}:
            raise ProbeError(
                f"{key}.tensor_layout has unexplained extra axes for event_frame: "
                f"{', '.join(sorted(extra_axes - {frame_axis}))}"
            )
    elif name == "voxel_grid":
        bins = _positive_int(
            _required(parameters, "bins", f"{key}.parameters.bins"),
            f"{key}.parameters.bins",
        )
        bin_axis = _require_text(
            _required(parameters, "bin_axis", f"{key}.parameters.bin_axis"),
            f"{key}.parameters.bin_axis",
        ).upper()
        if bin_axis in {"H", "W"}:
            raise ProbeError(
                f"{key}.parameters.bin_axis cannot reuse spatial axis {bin_axis!r}"
            )
        if bin_axis not in axis_sizes:
            raise ProbeError(f"{key}.parameters.bin_axis={bin_axis!r} is not in tensor_layout")
        if axis_sizes[bin_axis] != bins:
            raise ProbeError(
                f"{key}.parameters.bins={bins} does not match axis {bin_axis} size "
                f"{axis_sizes[bin_axis]}"
            )
    elif name == "time_surface":
        _require_text(
            _required(parameters, "reference", f"{key}.parameters.reference"),
            f"{key}.parameters.reference",
        )
        decay = _required(parameters, "decay", f"{key}.parameters.decay")
        if decay is None or (isinstance(decay, str) and not decay.strip()):
            raise ProbeError(f"{key}.parameters.decay must explicitly describe the decay or 'none'")
    else:
        raise ProbeError(f"unsupported representation for renderer validation: {name}")
    return _json_safe(parameters, f"{key}.parameters")


def _find_tbd(value: object, prefix: str = "") -> list[str]:
    unresolved: list[str] = []
    if isinstance(value, str) and value.strip().upper() == "TBD":
        unresolved.append(prefix or "<root>")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            unresolved.extend(_find_tbd(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            unresolved.extend(_find_tbd(child, child_prefix))
    return sorted(unresolved)


def _json_safe(value: object, key: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProbeError(f"{key} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {str(name): _json_safe(child, f"{key}.{name}") for name, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child, f"{key}[{index}]") for index, child in enumerate(value)]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item(), key)
        except (TypeError, ValueError, RuntimeError):
            pass
    raise ProbeError(f"{key} contains a value that is not JSON-serializable: {type(value).__name__}")


def _required(mapping: Mapping[str, Any], key: str, full_key: str) -> Any:
    if key not in mapping:
        raise ProbeError(f"missing required provider field: {full_key}")
    return mapping[key]


def _require_mapping(value: object, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProbeError(f"{key} must be an object/mapping")
    return value


def _require_text(value: object, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProbeError(f"{key} must be a non-empty string")
    text = value.strip()
    if text.upper() == "TBD":
        raise ProbeError(f"{key} is unresolved: TBD")
    return text


def _positive_int(value: object, key: str) -> int:
    if hasattr(value, "item") and not isinstance(value, (int, bool)):
        try:
            value = value.item()
        except (TypeError, ValueError, RuntimeError):
            pass
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ProbeError(f"{key} must be a positive integer")
    return value


def _nonnegative_int(value: object, key: str) -> int:
    if hasattr(value, "item") and not isinstance(value, (int, bool)):
        try:
            value = value.item()
        except (TypeError, ValueError, RuntimeError):
            pass
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProbeError(f"{key} must be a nonnegative integer")
    return value


def _number(value: object, key: str) -> int | float:
    if hasattr(value, "item") and not isinstance(value, (int, float, bool)):
        try:
            value = value.item()
        except (TypeError, ValueError, RuntimeError):
            pass
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProbeError(f"{key} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ProbeError(f"{key} must be a finite number")
    return value


def _python_number(value: object) -> int | float:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    number = float(value)
    if not math.isfinite(number):
        raise ProbeError("encountered a non-finite numeric observation")
    return number


def _resolution(value: object, key: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ProbeError(f"{key} must contain [height, width]")
    height = _positive_int(value[0], f"{key}[0]")
    width = _positive_int(value[1], f"{key}[1]")
    return height, width


def _interval_closure(value: object, key: str) -> str:
    closure = _require_text(value, key)
    if closure not in INTERVAL_CLOSURES:
        raise ProbeError(f"{key} must be one of {', '.join(INTERVAL_CLOSURES)}")
    return closure


def _interval_contains(value: int | float, start: int | float, end: int | float, closure: str) -> bool:
    left_ok = value >= start if closure[0] == "[" else value > start
    right_ok = value <= end if closure[1] == "]" else value < end
    return left_ok and right_ok


__all__ = [
    "PROBE_SCHEMA_VERSION",
    "REPRESENTATION_NAMES",
    "compute_event_subset_id",
    "run_probe",
]
