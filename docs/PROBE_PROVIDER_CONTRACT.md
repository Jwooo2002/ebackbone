# Real-data probe provider contract

## Purpose

`python main.py probe --config <config>` inspects one real raw-event sample. The
command does not choose a dataset, file format, temporal window, or renderer.
Those choices remain `TBD` until they are recorded as project decisions.

A dataset-specific provider is responsible for loading one sample, selecting one
label-independent event interval, and generating all three complementary event
representations from that same event subset. The generic probe then validates and
reports the resulting contract.

## Configuration

The configuration is JSON with schema version 1:

```json
{
  "schema_version": 1,
  "provider": {
    "callable": "my_adapter.probe:load_sample",
    "kwargs": {
      "dataset_root": "/absolute/path/to/dataset",
      "sample_id": "train/sample-0001"
    }
  }
}
```

Only `schema_version` and `provider` are accepted at the top level. JSON
`NaN`/`Infinity` constants are rejected. Dataset-specific resolved choices
belong in `provider.kwargs`.

The module must be importable in the current Python environment. An external
adapter can be made importable with an installed package or an explicit
`PYTHONPATH`. Probe configurations are trusted input because the configured
Python callable is executed.

Missing files should be raised as `FileNotFoundError` by the provider. The CLI
will report the exact missing path without inventing data statistics.

The checked-in `configs/probe.example.json` intentionally contains `TBD` values.
It demonstrates the expected actionable failure until a dataset and adapter are
selected.

## Provider return value

The provider must return either `ebackbone_v3.contracts.ProbeSample` or a mapping
with the equivalent fields.

### Top-level fields

- `dataset_name`: explicit dataset identity
- `split`: explicit split identity
- `raw_events`: raw-event record described below
- `representations`: exactly `event_frame`, `voxel_grid`, and `time_surface`
- optional `classification`: deterministic supervised-bookkeeping record described below

### Classification record

When a dataset has a verified class mapping, the provider may return:

- `class_id`: dataset-native class identifier;
- `class_index`: zero-based numeric class index;
- `class_count`: total number of mapped classes;
- `class_to_index`: complete bijection from class identifiers to `0..class_count-1`;
- `mapping_policy`: deterministic rule used to construct the mapping;
- `label_source`: where the class identifier is stored.

The generic probe verifies the bijection and the current sample's ID/index pair.
Classification metadata is bookkeeping only. It must not enter event selection,
temporal-window selection, filtering, or representation generation.

### Raw-event record

Required fields:

- `sample_id`: stable raw-event sample identifier
- `fields`: mapping from actual field names to one-dimensional numeric arrays
- `field_roles`: mapping for `x`, `y`, `timestamp`, and `polarity` to actual field names
- `storage_layout`: observed or adapter-decoded storage layout
- `timestamp_unit`: explicit unit
- `timestamp_ordering`: `nondecreasing`, `strictly_increasing`, or `not_sorted`
- `coordinate_convention`: `zero_based_xy` or `one_based_xy`
- `polarity_encoding`: `0/1`, `-1/+1`, or `false/true`
- `spatial_resolution`: `[height, width]`
- `temporal_start`, `temporal_end`, `interval_closure`
- `event_count_before_filter`, `event_count_after_filter`
- `filtering`: complete filtering description, including `none` when applicable

Supported interval-closure strings are `[)`, `[]`, `()`, and `(]`.

The arrays in `fields` are the post-filter event subset and must all have length
`event_count_after_filter`. The probe computes field shapes, dtypes, value ranges,
timestamp range, duration, observed timestamp ordering, and a decoded logical
shape from these arrays. It also checks coordinate bounds and polarity values
against the declared convention.

### Representation record

Every representation requires:

- `tensor`: non-empty finite numeric tensor or array
- `source_sample_id`
- `temporal_start`, `temporal_end`, `interval_closure`
- `event_subset_id`: the canonical fingerprint returned by
  `ebackbone_v3.probe.compute_event_subset_id(raw.fields, raw.field_roles)` for
  the exact post-filter event subset used by the renderer
- `event_count_before_filter`, `event_count_after_filter`
- `spatial_resolution`, `channel_count`, and `tensor_layout`
- `polarity_handling`
- `normalization`
- `clipping`
- `padding_or_truncation`
- `deterministic`
- `parameters`: complete renderer parameters as JSON-compatible values

Use explicit strings such as `none`; do not use `TBD` in a provider result that
is expected to pass.

`tensor_layout` is a comma-separated axis declaration. It must match tensor rank,
contain unique `C`, `H`, and `W` axes, and agree with `channel_count` and
`spatial_resolution`. The provider returns one sample, so batch axes `B`, `N`,
and `BATCH` are not allowed. Other explicit axes such as `T` are allowed when
their role is declared by the representation parameters.

Minimum representation-specific parameters are:

- event frame: `definition`, positive `rendered_frames`, and `frame_axis`; use
  `frame_axis: none` only for one accumulated frame with no extra axes, otherwise
  name one non-channel, non-spatial axis whose size equals `rendered_frames`
- voxel grid: `definition`, positive `bins`, and `bin_axis`; the named axis size
  must equal `bins`, and the bin axis cannot reuse `H` or `W`
- time surface: `definition`, `reference`, and an explicit `decay` description;
  use `decay: none` when applicable

## Alignment verification

The probe reports temporal alignment as `VERIFIED_FROM_PROVIDER_EVIDENCE` only
when all three
representations have:

- the raw record's sample identifier;
- the raw record's temporal start, end, and interval closure;
- the raw record's event counts before and after filtering; and
- the canonical raw-event fingerprint as `event_subset_id`.

The generic probe independently checks the metadata and fingerprint binding, but
the dataset-specific provider remains responsible for truthfully rendering from
that event subset. Matching tensor shapes alone are not alignment evidence.
Labels must not be used to select the interval, filter events, construct a
representation, or create the event-subset identity.

## Checked-in real provider

`ebackbone_v3.providers.n_imagenet_mini:load_sample` is the first real provider.
It reads the archive-native 100-class N-ImageNet mini release, preserves the raw
structured-array dtypes, and renders the three probe tensors once from the whole
stored event array. Its checked-in configs use the verified local dataset root
`/mnt/hdd1/datasets/event/n_imagenet`.

The renderer choices and their probe-only scope are recorded in D011 of
`docs/DECISIONS.md`. They do not select a production B0/B1 model, fusion design,
pooling method, encoder-sharing policy, or training recipe.
