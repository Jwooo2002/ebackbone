# AGENTS.md

## Project

`ebackbone_V3` studies event classification and backbone transfer using
complementary representations derived from one raw event stream.

The implemented repository includes:

- original B0 frame-only and B1 tri-representation foundations
- heterogeneous V1 with controlled baselines
- full-event point-to-voxel-to-frame hierarchy and TS/polarity/structural ablations
- hierarchy-only, latent-only and dual weighted fusion
- SeACT downstream fine-tuning and matched scratch training

Read `docs/README.md` for the document map and the relevant study protocol
before modifying its implementation. CLIP is retired; its historical documents
and `archive/pre-cleanup-20260926` preserve the earlier work.

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
11. Preserve existing runs, checkpoints, manifests, frozen snapshots, queues and live processes. Repository cleanup is not authorization to launch training or resume a paused successor.
12. Read live histories, reports and queue/launch state before reporting current progress. Separate internal validation from final held-out test and bounded engineering evidence from full-run results.
13. Keep authorized studies within their recorded protocol. The SeACT supervisor uses the immutable sibling `ebackbone_v3_seact_artifacts/snapshot_20260926/`; do not edit its source or artifact state during repository maintenance.
14. Mini final-test access remains gated. SeACT's recorded final held-out evaluation is limited to its authorized protocol after checkpoint selection; do not repeat it during cleanup or routine verification.

## Current scope

Allowed within the user's requested task:

- inspect repositories and datasets while preserving existing access boundaries
- maintain the implemented baselines, hierarchy, dual-fusion and SeACT code
- verify data contracts and run bounded unit/smoke checks
- document exact commands, provenance and results

An existing trainer or launch command does not authorize a new run. The recorded
Mini/SeACT study authorization applies to those studies only; inspect their
protocol and live status before acting. Existing legacy hierarchy/TS queue pauses
must remain in force. Preserve dual fusion as a learned weighted combination of
hierarchy and latent features, not a residual correction.

Out of scope until explicitly approved:

- restarting CLIP or adding semantic/text alignment
- SSMER+ self-supervised losses, multi-view InfoNCE or Event2Vec
- auxiliary reconstruction, new distillation, prompts or external pretrained models
- new HARDVS work beyond the retained investigation
- new datasets, detection/segmentation heads or architecture-scaling studies
- new full training, transfer runs, queue successors or final-test evaluations

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

The initial B1 concept is developed in the separate heterogeneous V1 protocol
(`docs/V1_COMPARISON.md`). Later hierarchy and dual-fusion studies have their own
recorded architectures. Do not silently substitute one family or fusion rule for
another; architectural changes require a documented decision and bounded checks.

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

Report efficiency metrics for implemented model comparisons:

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
- checkpoint save/reload, exact resume and backbone-export behavior when relevant

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
