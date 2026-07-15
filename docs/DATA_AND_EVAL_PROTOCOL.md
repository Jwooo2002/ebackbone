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

The rank bytes are independently reproducible from provenance:
`UTF8(selection_domain_utf8) || NUL || ASCII(decimal_seed) || NUL ||
UTF8(source_stable_sample_id)`. The stored domain string deliberately excludes
both separator bytes; provenance names both separators explicitly.

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

### Runtime manifest-backed adapter

The read-only runtime adapter consumes one row from one of these immutable JSONL
files. It does not resample membership, derive a split at runtime, or use the
bundled stale path lists. The selected row remains the authority for both the
project `split` and the dataset-native `source_split`; these values are returned
separately with the sample metadata.

The adapter resolves `source_archive` and `source_members` relative to the
configured N-ImageNet root without extracting the full release. A source-train
row is read through its declared ZIP member and nested TAR.GZ member; a
source-validation row is read from its declared ZIP member. Archive handles are
scoped to one resolution operation rather than shared globally, so the adapter
does not retain an open ZIP or TAR handle between accesses.

Before decoding, the adapter validates the manifest schema and row semantics,
including the stable sample ID, synset, class index and label, project/source
split relationship, archive/member paths, and complete source locator. It then
checks the resolved raw NPZ payload's byte length and SHA-256 against
`raw_content_size_bytes` and `raw_content_sha256`. This raw payload hash is the
digest of the exact stored NPZ bytes after archive decompression, not the
decoded-event fingerprint.

The adapter is CPU-only. It decodes the exact `event_data` NPZ contract and
validates its one-dimensional `x:uint16`, `y:uint16`, `t:uint16`, and `p:bool`
arrays before representation rendering. It does not initialize CUDA, construct
or run a model, batch samples, augment samples, or execute a training loop.

Project roles are explicit and fail closed. Runtime callers must request exactly
`train`, `validation`, or `test`; filenames, `train=False`, `eval=True`, and
missing values never infer a role. Final-test access requires both `split="test"`
and `allow_final_test=True` (CLI: `--split test --allow-final-test`). The gate is
checked before any test manifest or archive member is read. A dataset-native
source split remains `validation` for those rows; `test` is only the explicit
project role. Training and checkpoint-selection construction accepts only project
train and internal validation, both backed by source-train manifest rows.

The runtime adapter derives canonical manifest filenames from the explicit role,
validates canonical manifest/provenance/checksum metadata before accepting row
membership, and opens only the archive/member locator named by that immutable
row. It does not call the archive-index builder, glob archives, accept membership
predicates, or derive another split at runtime. Opening train or validation does
not parse `test.jsonl`.

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

### Adapter output and optional cache

The adapter renders the existing D012 production contract; it does not redefine
frame, voxel-grid, or time-surface semantics. Each returned item includes
structured metadata with the stable sample ID, class label/index and synset,
project split, source split, exact raw payload hash, canonical raw-event
fingerprint, temporal start/end and closure, event count, and renderer
fingerprint.

For B0 access, a dedicated frame-only renderer is called and the public output
contains only the event-frame tensor and this metadata; voxel and time-surface
tensors are neither requested nor allocated. For B1 access, it contains the event frame, voxel grid, and time
surface. All B1 tensors are bound to the same verified raw-event fingerprint and
the same closed observed-support interval; a matching shape alone is not
accepted as alignment evidence.

Caching is optional and on-demand for B1. B0 requires cache-off because the
current cache schema stores the full three-representation bundle. Cache-off
performs no cache read or write. B1 cache-on requires an explicit cache root; there is no hidden cache location and
the adapter never precomputes the release. A cache candidate is accepted only
after its provenance matches the production renderer fingerprint and contract
version, project/source split identity, the exact manifest raw-payload SHA-256,
and the validated raw-event identity and interval. A changed renderer/contract
version, split role, or raw payload hash makes the entry stale and prevents its
reuse.

`python main.py inspect-sample` exposes this adapter for one-row inspection.
The command reports metadata and tensor shape/dtype/range summaries only; it
does not invoke a model. Its default dataset root is
`/mnt/hdd1/datasets/event/n_imagenet`, and `--dataset-root` overrides that root
for another local copy of the release.

## B0 protocol

### Train input

Event frame generated from the raw event sample.

The production tensor is the D012 float32 `[2,480,640]` negative/positive
`log1p` count frame.

### Production training and checkpoint selection

D014 uses the complete immutable project `train.jsonl` for optimizer updates
and the complete immutable project `validation.jsonl` only for checkpoint
selection. The selection rule is highest validation top-1 accuracy, breaking a
tie with lower validation cross-entropy and then retaining the earlier epoch.
The production command has no project-test argument and rejects a test manifest
in either input position before constructing a dataset or opening a sample.

Every checkpoint records the production model identity and random initialization,
optimizer and scheduler configurations and states, deterministic seed, exact
train/validation manifest hashes and counts, renderer contract/config/fingerprint,
epoch history, and selection state. `checkpoint_last.pt` is the only resume
source; an exact run-configuration match is required. `checkpoint_best.pt` is
selection output, not a resume source.

### Bounded production-path integration diagnostic

`diagnose-b0-train` is separate from production training. Its API hard-codes
project `train`, `baseline="b0"`, and `cache="off"`; it has no validation/test
selector or final-test override. It selects 8-16 sample IDs deterministically
without labels, requires a real batch size of at least 4, and permits no more
than 200 optimizer steps. Only event frames are decoded, rendered, and collated.
The diagnostic verifies production ResNet-18 loss/backward/update behavior and
strict checkpoint restoration, including BatchNorm running buffers and exact
evaluation logits. Results are pipeline engineering evidence, not accuracy or
checkpoint-selection evidence.

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
