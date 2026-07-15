# ebackbone_V3

Tri-representation event encoding for downstream event classification.

## Current question

Given one raw event sample, does jointly using event frame, voxel grid, and time surface improve classification over a frame-only encoder?

## Initial baselines

### B0 — Frame-only supervised baseline

```text
raw event sample
→ event frame
→ encoder
→ linear classifier
```

### B1 — Tri-representation supervised baseline

```text
raw event sample
├─ event frame
├─ voxel grid
└─ time surface
        ↓
tri-representation encoder
        ↓
linear classifier
```

Both baselines use random initialization, end-to-end supervised training, and cross-entropy loss.

## Runnable foundation

From the repository root:

```bash
python main.py --help
python main.py probe --config configs/probe.example.json
python main.py probe --config configs/probe.n_imagenet_mini.train.part1.json
python main.py probe --config configs/probe.n_imagenet_mini.validation.first.json
python main.py smoke --baseline b0
python main.py smoke --baseline b1
```

The two smoke commands are deterministic CPU-only execution checks. They use
small opaque synthetic tensors to exercise a forward pass, linear classification
head, cross-entropy loss, backward pass, and one optimizer step. Their shapes and
toy computation are not a real data contract or a production B0/B1 architecture.
The output keeps fusion, pooling, encoder sharing, normalization, voxel bins, and
time-surface semantics marked `TBD`.

The probe uses an explicitly configured Python provider. It validates raw fields,
event counts, timestamp range, classification bookkeeping, representation tensor
summaries, provenance, and cross-representation temporal alignment. The generic
`probe.example.json` remains an intentional unresolved-contract example. The
resolved N-ImageNet mini configs exercise the verified archive-native provider at
the local dataset root `/mnt/hdd1/datasets/event/n_imagenet`. See
`docs/PROBE_PROVIDER_CONTRACT.md` and D009-D011 in `docs/DECISIONS.md`.

The production representation contract is implemented separately in
`ebackbone_v3/representations.py` and recorded in D012. It fixes the B0 frame
and aligned B1 frame/voxel/time-surface tensors without implementing a model or
training loop.

## Development setup

Python 3.10 or newer, NumPy 1.24 or newer, and PyTorch 2.0 or newer are required.
For an editable development installation:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Not included yet

- SSMER+ self-supervised learning
- Event2Vec
- EventBind-inspired fusion
- semantic alignment
- auxiliary reconstruction
- detection or segmentation

## Document order

1. `AGENTS.md`
2. `docs/PROJECT_CONTEXT.md`
3. `docs/DATA_AND_EVAL_PROTOCOL.md`
4. `docs/BASELINES_B0_B1.md`
5. `docs/PROBE_PROVIDER_CONTRACT.md`
6. `docs/DECISIONS.md`
