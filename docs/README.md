# Documentation index

Start with the [repository overview](../README.md),
[project context](PROJECT_CONTEXT.md), and [development rules](../AGENTS.md).
Documents record the protocol and evidence at their stated dates. Use the
corresponding local run artifacts for live status, and keep internal validation,
held-out evaluation and engineering probes separate. The
[repository cleanup audit](REPOSITORY_CLEANUP.md) records the September 2026
maintenance scope and verification.

## Current studies

| Document | Purpose |
| --- | --- |
| [SeACT study](SEACT_STUDY.md) | Downstream fine-tuning / scratch protocol, data contract, completed Mini reference results and local status paths |
| [SeACT review](SEACT_REVIEW.md) | Review and verification evidence for downstream implementation |
| [Dual fusion study](DUAL_FUSION_STUDY.md) | Full-event hierarchy-only, latent-only and learned weighted dual fusion comparison |
| [Dual fusion review](DUAL_FUSION_REVIEW.md) | Implementation review and bounded verification |

## Hierarchy family

| Document | Purpose |
| --- | --- |
| [Hierarchy V1](HIERARCHY_V1.md) | All-event point → voxel → frame architecture and exact tensor/compute contract |
| [Structural ablations](HIERARCHY_ABLATIONS.md) | Early-skip, local interaction and same-architecture controls |
| [Polarity ablation](HIERARCHY_POLARITY.md) | Polarity-separated point-to-voxel aggregation |
| [Time-surface V1](HIERARCHY_TS_V1.md) | Original time-surface integration |
| [Time-surface residual V2](HIERARCHY_TS_RESIDUAL_V2.md) | Residual correction experiment |
| [Time-surface confidence V3](HIERARCHY_TS_CONFIDENCE_V3.md) | Confidence-gated correction experiment |
| [TS handoff](TS_HANDOFF.md) | Dated implementation and experiment handoff |

## Foundation and common contracts

- [Baseline usage](BASELINE_USAGE.md): original B0/B1 commands and renderer/data contracts.
- [B0/B1 definitions](BASELINES_B0_B1.md): historical baseline specification.
- [V1 comparison](V1_COMPARISON.md): heterogeneous V1 and controlled baselines.
- [Data and evaluation protocol](DATA_AND_EVAL_PROTOCOL.md): immutable splits and evaluation boundaries.
- [Probe provider contract](PROBE_PROVIDER_CONTRACT.md): explicit raw-event provider interface.
- [Representation analysis](REPRESENTATION_ANALYSIS.md): representation design evidence.
- [Decision log](DECISIONS.md): dated design decisions; later study protocols may supersede initial-scope entries.
- [W&B tracking](WANDB.md): optional experiment tracking.
- [CPU CI template](ci/README.md): reproducible checks and the GitHub workflow permission needed for activation.

## Historical studies

- [HARDVS investigation](HARDVS_NINE_RUN_STUDY.md): the original downstream plan;
  the authorized executable fallback is documented in the SeACT study.
- [CLIP bounded comparisons](HIERARCHY_CLIP_COMPARISONS.md) and
  [CLIP pretraining](HIERARCHY_CLIP_PRETRAINING.md): retired protocols, preserved
  for traceability. Their commands refer to source removed during cleanup.
  The complete prior source is available at
  [archive/pre-cleanup-20260926](https://github.com/Jwooo2002/ebackbone/tree/archive/pre-cleanup-20260926)
  (commit `44d0148`). Do not execute archived launch instructions in the current
  checkout or restart the retired study as part of maintenance.
