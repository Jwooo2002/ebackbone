# Data and Evaluation Protocol

## Evaluation unit

The classification unit is one raw event recording or one explicitly defined temporal window from a recording.

Each sample has:

- a raw event sequence
- one class label
- a stable sample identifier

The selected dataset and whole-stored-sample policy are recorded in D009-D010.
D011 defines the probe-only renderer; D012 separately defines the production
B0/B1 representations. The probe renderer must not be substituted for the
production renderer or reinterpreted as production evidence.

## Supervised project split protocol

N-ImageNet mini provides dataset-native source splits named `train` and
`validation`, but no independent official test split. D013 adopts this fixed
project-level interpretation:

- project `train`: 124,395 samples remaining after the internal holdout;
- project `validation`: 5,000 official-training samples, exactly 50 per class;
- project `test`: all 5,000 official-validation samples, exactly 50 per class.

The official training source is partitioned once by a deterministic per-class
SHA-256 rank with seed `20260715`. Runtime data loading must consume the checked
manifest assignment and must never create another random split. The official
validation source is not renamed at the storage/provider layer: each final-test
row preserves `source_split: "validation"` and records the project role
separately as `split: "test"`.

All 100 classes occur in every project split. Sample IDs and source locators are
globally unique, and the three project sample-ID sets are pairwise disjoint.
The stable sample ID is based on source identity (`source_split/class/file`), so
changing the derived project assignment does not change sample identity.

The final-test labels may appear in the immutable manifest only for supervised
bookkeeping, protocol verification, and final scoring after the method and
checkpoint-selection procedure are frozen. They must not influence training,
checkpoint selection, early stopping, hyperparameters, normalization,
augmentation, representation choices, fusion, or architecture decisions. Any
decision informed by project-test results invalidates those results as untouched
final-test evidence.

### Manifest and provenance contract

The canonical artifact directory contains only:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `provenance.json`
- `SHA256SUMS`

Every row records schema version, source-stable sample ID, WordNet class ID,
numeric class label/index, project split, dataset-native source split, relative
source archive and nested member paths, exact member byte size, and SHA-256 of
the uncompressed stored NPZ bytes. No representation cache or tensor is read or
created.

Provenance records the selection algorithm and seed, holdout quota, class
mapping, source/project counts, all 11 archive checksums, and checksums for the
two stale bundled path lists marked non-authoritative. It also records exact
manifest hashes, a source-catalog hash, a canonical parameter hash, and a
generation fingerprint. Timestamps, absolute dataset roots, filesystem mtimes,
and archive iteration order are excluded from canonical output.

Generation is atomic and immutable. Repeating the same generation against a
different empty destination is byte-identical. Repeating it against the same
destination verifies and leaves every file untouched. Different parameters,
inputs, incomplete artifacts, or altered bytes are refused rather than
overwritten.

## Raw event contract

Expected raw event fields:

- spatial coordinate `x`
- spatial coordinate `y`
- timestamp `t`
- polarity `p`

For the selected N-ImageNet mini probe, the verified fields are a one-dimensional
packed structured NumPy array with `x:uint16`, `y:uint16`, `t:uint16`, and
`p:bool`. Raw timestamps are microseconds and nondecreasing with ties; coordinates
are zero-based at 480 x 640 resolution. See D010 for evidence boundaries and
remaining acquisition/export `TBD` items.

## Representation generation

For a B1 sample, event frame, voxel grid, and time surface must be generated from the same source sample and the same temporal interval.

The following parameters must be recorded:

- temporal start and end
- event filtering rules
- number of temporal bins
- number of rendered frames
- spatial resolution
- polarity channels
- normalization
- clipping
- padding or truncation

For N-ImageNet mini B0/B1, D012 resolves these representation parameters.
Architecture and augmentation parameters remain separate decisions and must not
be inferred from the renderer.

## B0 protocol

### Train input

Event frame generated from the raw event sample.

The production tensor is the D012 float32 `[2,480,640]` negative/positive
`log1p` count frame.

### Test input

Event frame generated using the same rendering policy as training.

## B1 protocol

### Train input

- event frame
- voxel grid
- time surface

All three must come from the same raw event sample and temporal interval.

Their production shapes are frame `[2,480,640]`, voxel `[2,5,480,640]`, and
time surface `[2,480,640]`; all use the D012 polarity order and provenance.

### Test input

The same three representations using the same rendering policy.

## Fair comparison requirements

B0 and B1 must share:

- dataset split
- raw sample identity
- temporal observation interval
- label set
- spatial crop and resolution
- augmentation policy where applicable
- optimizer
- scheduler
- epoch or optimizer-step budget
- validation and checkpoint-selection rule
- classification head type

Model parameters and compute may differ. These differences must be measured rather than hidden.

## Leakage prevention

Do not use:

- validation or test labels during representation generation
- class-dependent temporal slicing
- class-dependent event filtering
- cached representations generated from a different split
- future events outside the defined sample interval

Any dataset-level normalization statistics must be computed from the training split only.

Here `training split` means project `train` from the immutable D013 manifest;
internal validation and final test are excluded.

## Metrics

Primary:

- top-1 accuracy

Conditional:

- top-5 accuracy when the number of classes makes it meaningful

Efficiency metrics after functional validation:

- parameter count
- preprocessing latency
- forward latency
- peak memory
- throughput

## First required data probe

Before model implementation, inspect at least one real sample and report:

- raw event shape and fields
- event count
- temporal duration
- frame tensor shape
- voxel tensor shape
- time-surface tensor shape
- dtype and value range for each
- confirmation that all three cover the same temporal interval
