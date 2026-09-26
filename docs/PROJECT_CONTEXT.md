# Project context

`ebackbone_V3` studies event classification using complementary representations
of the same raw event sample and transfer of the resulting event backbone.
Event frames, voxel grids and time surfaces are derived representations, not
independent sensors. Train/test input availability must match each protocol.

## Implemented research paths

1. **B0/B1 foundation:** frame-only supervised classification and the original
   tri-representation contract. B0 production uses its documented ResNet-18;
   synthetic smoke models and the compact debug model are engineering fixtures.
2. **Heterogeneous V1:** representation-specific encoders and controlled
   comparators described in [V1_COMPARISON.md](V1_COMPARISON.md).
3. **Hierarchy family:** all events pass through point features, learned voxel
   aggregation, temporal processing and frame-level features. TS, polarity and
   structural variants are separately documented experiments.
4. **Dual fusion:** hierarchy-only, latent-only and a learned weighted
   combination of the two embeddings. This is a supervised full-event Mini
   study with a learned classifier and cross-entropy, not residual TS fusion.
5. **SeACT transfer:** fine-tune each Mini backbone and compare with the same
   architecture trained from scratch under the matched downstream protocol.
   [SEACT_STUDY.md](SEACT_STUDY.md) records the authorized fallback from the
   earlier HARDVS investigation and the completed Mini reference results.

The [documentation index](README.md) maps each family to its architecture,
protocol and review. Early decisions remain historical context; they do not
imply that later implemented work is still pending.

## Data and evidence boundaries

Mini experiments use immutable train/internal-validation/final-test membership.
Mini internal-validation scores are not final-test accuracy. SeACT uses its
separately documented recording-level splits, checkpoint selection and one-time
held-out evaluation. Labels must never select events or temporal windows.
Preserve full events and common observed temporal support wherever the study
requires them; do not silently change rendering, geometry or cache provenance.

Dated study documents establish protocol and historical evidence. Live status
comes from histories, reports, verified checkpoint/export files and queue state
in the sibling study artifact directories. The active SeACT execution uses
`../ebackbone_v3_seact_artifacts/snapshot_20260926/`, with
`../ebackbone_v3_seact_artifacts/study_status.json` as its status authority.
Repository maintenance must preserve this snapshot and ongoing execution.

## Scope and historical work

The initial SSMER comparison motivated consuming multiple representations at
both training and inference. The implemented studies above use supervised
classification; their implementation does not authorize adding self-supervised
losses, new external pretrained models, or other downstream tasks.

CLIP has been dropped. Its protocol documents remain for traceability and the
removed implementation is preserved at Git tag `archive/pre-cleanup-20260926`
(commit `44d0148`). The old hierarchy/TS queue remains paused. HARDVS investigation
is retained as history; the executable downstream protocol is SeACT.

Cleanup, tests and documentation updates do not authorize new training, queue
successors, final-test access or modification of immutable run artifacts. Follow
[AGENTS.md](../AGENTS.md) and each study's recorded authorization boundaries.
