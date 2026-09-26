# Repository cleanup — 2026-09-26

This maintenance pass organizes the implemented event-backbone studies for the
existing `Jwooo2002/ebackbone` repository. It does not change model computation,
rendering, optimization schedules, dataset membership or evaluation protocols.

## Recovery point and removals

The previously uncommitted hierarchy, dual-fusion, SeACT and historical CLIP
implementations were first committed as `44d0148`. The tag
[`archive/pre-cleanup-20260926`](https://github.com/Jwooo2002/ebackbone/tree/archive/pre-cleanup-20260926)
preserves that source before removal. Historical CLIP instructions apply to
that revision, not the current tree. Use a separate checkout of the tag for
historical inspection; preserve the matching source/config snapshot when
resuming any existing experiment with source-hash identity checks.

Removed from the current tree:

- 14 retired `ebackbone_v3/hierarchy_clip*.py` modules.
- 12 corresponding `tests/test_hierarchy_clip*.py` test modules.
- 2 `configs/hierarchy_clip*.json` study configurations.
- 7 tracked `.idea/` files; local IDE preferences remain on disk.
- Four unused source imports, three unused test imports, and obsolete CLIP
  bytecode caches.

The 28 CLIP source/test/config files account for 6,136 removed lines. Retained
code has no imports of those modules. Historical CLIP protocol documents remain
available with archive notices. Other baselines and hierarchy ablations remain
because they support implemented comparisons and checkpoint reproducibility.
Public `TrainConfig` and `cpu_state` imports and shared pytest fixtures are
preserved explicitly, despite appearing unused within their defining modules.

## Organization and publication

The root README now maps the implemented model families and runnable commands.
[The documentation index](README.md) groups current studies, common contracts
and historical experiments. Detailed original baseline usage is preserved in
[BASELINE_USAGE.md](BASELINE_USAGE.md). Project context and development rules
now reflect hierarchy, dual fusion and the authorized SeACT study.

`.gitignore` excludes generated reports, checkpoints, run output, datasets,
tracking files, editor state and local credentials. Existing immutable manifests
remain versioned. The [CPU workflow template](ci/README.md) installs PyTorch
2.7.1 and NumPy 1.26.4, checks imports with Pyflakes and runs the portable test
suite. It is not active: the existing GitHub OAuth credential lacks the
`workflow` scope, so GitHub rejected publication under `.github/workflows/`.
The template is retained as a regular documentation file. Runtime dataset
and artifact paths in study configs remain protocol records; they are not
portable dataset distributions.

The supported study setup is an editable install from a source checkout. The
wheel contains the Python package; root-level configs, manifests and preparation
tools are not wheel package data. A successful wheel build verifies packaging,
not a standalone distribution of study resources or datasets.

## Verification

Final result: **PASS — 237 tests passed, 3 local-integration tests skipped** in
56.52 seconds (Python 3.11, PyTorch 2.7.1, NumPy 1.26.4). Pyflakes and
`git diff --check` passed; the wheel built successfully and contains all 51
retained Python modules with no retired CLIP code or checkpoint files. The eight
documented CLI help/synthetic-smoke commands passed. All 108 protected file
hashes remained unchanged. These are local checks; GitHub Actions was not
activated and has no CI result for this cleanup.

The portable suite runs in a fresh copy containing versioned source and bundled
manifests, outside the original workspace's sibling experiment directories,
with CUDA hidden and one OpenMP/BLAS thread. CPU Gloo tests cover distributed
gradients, uneven batches, checkpoint/export reloads and exact resume.

```bash
python -m pyflakes ebackbone_v3 tests tools
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q
python -m pip wheel --no-deps --no-build-isolation . --wheel-dir /tmp/ebackbone-v3-wheels
git diff --check
```

Three machine-local tests require `--run-local-integration`: the real-data B0
probe and two tests of the external legacy queue script. Their actual dataset
or queue assets are never opened by the default suite. If explicitly enabled
without those assets, they skip with a reason. Synthetic contract tests remain
enabled. W&B and AEDAT decoding are optional integrations outside CPU CI.

## Preserved experiments

Before changes, a local backup and SHA-256 inventory were saved under
`../ebackbone_v3_repo_cleanup_20260926/`. The inventory covers 108 immutable
files: bundled manifests, existing repository reports/checkpoints, the SeACT
frozen snapshot and study plan, and legacy queue/pause records. Running histories
are allowed to advance naturally; this maintenance does not rewrite them.

The active SeACT process uses its separate `snapshot_20260926` directory. No
training process, supervisor, checkpoint, dataset, frozen snapshot or queue
state was modified by the cleanup. Verification starts no new experiment and
does not access final-test events.
