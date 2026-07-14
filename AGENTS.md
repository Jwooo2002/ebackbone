# AGENTS.md

## Project

`ebackbone_V3` develops a tri-representation event encoder for downstream event classification.

A single raw event sample is rendered into three complementary event representations:

- event frame
- voxel grid
- time surface

The initial scope is limited to two supervised baselines:

- **B0:** frame-only classification
- **B1:** frame + voxel grid + time surface classification

Both baselines are trained from random initialization with a linear classification head and cross-entropy loss.

## Core terminology

Use:

- raw event sample
- event representation
- tri-representation event encoder
- multi-representation event classification
- complementary event representations

Do not describe event frame, voxel grid, and time surface as independent sensor modalities unless a document explicitly discusses that terminology. They are derived from the same raw event stream.

## Non-negotiable experimental rules

1. B0 and B1 must use the same raw event samples and the same train/validation/test split.
2. All representations for a B1 sample must be generated from the same raw events and the same temporal interval.
3. Train-time and test-time input availability must match within each baseline.
   - B0: frame at train and test.
   - B1: frame, voxel grid, and time surface at train and test.
4. Labels may be used only for supervised downstream classification and deterministic dataset bookkeeping.
5. Do not use labels to construct event representations, select events, or determine temporal windows.
6. Do not compare B0 and B1 using different optimizer, scheduler, epoch budget, data split, or augmentation policy unless the difference is explicitly documented as an ablation.
7. Do not silently reuse cached representations generated with different rendering parameters.
8. Do not launch long training jobs without an explicit request.
9. Do not commit, push, or modify external repositories unless explicitly requested.
10. Do not guess tensor shapes, event counts, temporal windows, channel counts, or dataset statistics. Inspect the actual implementation and data.

## Current scope

Allowed:

- inspect repositories and datasets
- define data contracts
- implement B0 and B1
- add unit tests and data probes
- run small smoke tests
- document exact commands and outputs

Out of scope until explicitly approved:

- SSMER+ self-supervised losses
- multi-view InfoNCE
- auxiliary reconstruction
- Event2Vec
- EventBind-style prompts or attention fusion
- semantic alignment
- teacher-student distillation
- detection or segmentation heads
- architecture scaling studies
- external pretraining

## Required development sequence

Work in small, verifiable steps.

1. Inspect the current repository state.
2. Read the relevant project documents.
3. Identify the exact raw-event sample contract.
4. Probe one real sample end to end.
5. Verify temporal alignment among frame, voxel grid, and time surface.
6. Record tensor shapes, dtypes, ranges, and normalization.
7. Implement or modify one bounded component.
8. Add tests before broad integration.
9. Run a minimal smoke test.
10. Report changed files, commands, results, and unresolved issues.

Do not combine data loading, rendering, model fusion, training-loop changes, and long-run launch into a single task.

## B0 definition

```text
raw event sample
→ event-frame representation
→ encoder
→ pooled embedding
→ linear classification head
→ class logits
```

Requirements:

- random initialization
- supervised end-to-end training
- cross-entropy loss
- frame-only input at train and test
- no SSL objective

## B1 definition

```text
raw event sample
├─ event frame
├─ voxel grid
└─ time surface
        ↓
tri-representation encoder
        ↓
fused embedding
        ↓
linear classification head
        ↓
class logits
```

Requirements:

- random initialization
- supervised end-to-end training
- cross-entropy loss
- all three representations at train and test
- no SSL objective

The fusion design is not yet fixed. Do not choose a fusion strategy without a dedicated decision task and documented rationale.

## Data contract checks

For every representation, verify and record:

- source raw-event sample identifier
- temporal start and end
- event count before and after filtering
- spatial resolution
- channel count
- tensor shape
- dtype
- value range
- polarity handling
- normalization
- padding or truncation
- deterministic versus stochastic rendering behavior

For B1, additionally verify that all three tensors correspond to the same event subset or explicitly documented equivalent temporal slice.

## Evaluation rules

Primary classification metrics:

- top-1 accuracy
- top-5 accuracy only when meaningful for the class count

Efficiency metrics to add once B0/B1 are functional:

- parameter count
- preprocessing latency
- forward latency
- peak memory
- throughput

Accuracy and efficiency must be reported separately. Higher compute does not invalidate B1, but the cost must not be hidden.

## Testing expectations

At minimum, cover:

- deterministic dataset split
- B0 sample contract
- B1 sample contract
- cross-representation sample identity
- temporal alignment
- shape and dtype validation
- classifier output shape
- finite forward loss
- one optimizer step
- checkpoint save and reload when training infrastructure is introduced

## Reporting format

Every completed task should report:

### Status

`PASS`, `PARTIAL`, or `BLOCKED`

### Changed files

List exact paths and concise descriptions.

### Verification

List exact commands and observed results.

### Data contract

Report confirmed shapes and temporal alignment when relevant.

### Open issues

Separate verified facts from assumptions and unresolved questions.

### Next single task

Recommend only one bounded next action.
