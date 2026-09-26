# ebackbone_V3

Event classification and backbone transfer using complementary representations
of the same raw event stream. The repository contains the production B0 baseline,
B1 representation contracts and synthetic smoke scaffolding, the heterogeneous
V1 comparison, point-to-voxel-to-frame hierarchy
models, dual hierarchy/latent fusion, and the SeACT downstream study.

For the relationship between the local V1/V2/V3 checkouts and experiment folders,
see the [ebackbone version and folder guide (한국어)](docs/EBACKBONE_FAMILY.md).

## Model and experiment map

| Family | Implementation | Protocol |
| --- | --- | --- |
| B0 production / B1 contracts and smoke checks | `main.py`, `ebackbone_v3/b0_*.py`, `representations.py` | [Baseline usage](docs/BASELINE_USAGE.md), [definitions](docs/BASELINES_B0_B1.md) |
| Heterogeneous V1 and controlled baselines | `ebackbone_v3/v1*.py` | [V1 comparison](docs/V1_COMPARISON.md) |
| Hierarchy and TS / polarity / structural ablations | `ebackbone_v3/hierarchy*.py` | [Hierarchy V1](docs/HIERARCHY_V1.md), [family index](docs/README.md#hierarchy-family) |
| Hierarchy-only / latent-only / dual weighted fusion | `ebackbone_v3/dual_fusion*.py` | [Dual fusion](docs/DUAL_FUSION_STUDY.md) |
| SeACT fine-tuning and matched scratch runs | `ebackbone_v3/seact*.py`, `tools/prepare_seact.py` | [SeACT study](docs/SEACT_STUDY.md) |

The CLIP experiment is retired. Its historical protocols remain in
[the documentation archive](docs/README.md#historical-studies); removed source,
tests and configs are recoverable from Git tag `archive/pre-cleanup-20260926`
(commit `44d0148`).

## Install and verify

Python 3.10+, NumPy 1.24+ and PyTorch 2.0+ are declared in
[pyproject.toml](pyproject.toml). From the repository root:

```bash
python -m pip install -e '.[dev]'
python main.py --help
python main.py smoke --baseline b0
python main.py smoke --baseline b1
python -m pytest -q
```

Use this source checkout with an editable installation: study configs, manifests
and preparation tools live outside the Python package and are not included in
the standalone wheel. Production tri-representation models are implemented in V1.

The smoke commands use small synthetic CPU tensors and perform one optimizer
step. They verify execution, not scientific accuracy or a real-data contract.
The default test suite uses synthetic fixtures and bounded CPU checks; local
integration tests that depend on the original machine are skipped. To opt in
when those local resources are available, run
`python -m pytest -q --run-local-integration`. Optional W&B
tracking is available through `python -m pip install -e '.[tracking]'`;
[tracking instructions](docs/WANDB.md) describe its use. Decoding SeACT AEDAT4
files additionally requires the `dv` decoder used by `tools/prepare_seact.py`;
it is separate from the core dependency set.

Inspect available study commands without launching training:

```bash
python -m ebackbone_v3.v1 --help
python -m ebackbone_v3.hierarchy --help
python -m ebackbone_v3.hierarchy_ddp --help
python -m ebackbone_v3.dual_fusion_train --help
python -m ebackbone_v3.seact_train --help
```

## Data, runs and reproducibility

- `ebackbone_v3/`: models, event rendering, data adapters, trainers and verification.
- `configs/`: recorded study configurations, including machine-specific dataset,
  manifest and cache paths. Inspect and adapt paths in a new config for another
  machine; preserve any existing run's hashed configuration.
- `manifests/`: immutable dataset membership and provenance. Mini internal
  validation is distinct from its final held-out test split.
- `tests/`: contract, model, training, resume and export checks.
- `docs/`: architecture and protocol records; begin with the
  [documentation index](docs/README.md).

Raw datasets, caches, checkpoints, generated reports and local tracking output
are not distributed in this source repository. On the original workspace,
study artifacts live in sibling `ebackbone_v3_*_artifacts/` directories. Their
histories, reports and queue status are the authority for live progress; dated
protocol documents do not establish current completion.

The SeACT supervisor uses the frozen
`../ebackbone_v3_seact_artifacts/snapshot_20260926/` source and records live state
in that artifact directory's `study_status.json`. Preserve running processes,
frozen snapshots, checkpoints and the paused legacy hierarchy/TS queue.
Historical checkpoint resumes require their matching archived or frozen source;
cleanup changes source hashes checked by strict resume validation. See the
[cleanup audit](docs/REPOSITORY_CLEANUP.md) for recovery and verification details.

Repository maintenance and verification do not authorize additional training,
queue successors, data-split changes or final-test evaluation. See
[AGENTS.md](AGENTS.md), [project context](docs/PROJECT_CONTEXT.md), and the
[data/evaluation protocol](docs/DATA_AND_EVAL_PROTOCOL.md) before changes.
