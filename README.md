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
python main.py build-splits --config configs/splits.n_imagenet_mini.json
python main.py verify-splits --manifest-dir manifests/n_imagenet_mini/supervised-v1
python main.py inspect-sample \
  --manifest-dir manifests/n_imagenet_mini/supervised-v1 \
  --index 0 \
  --baseline b0 \
  --cache off \
  --split train
python main.py smoke --baseline b0
python main.py smoke --baseline b1
python main.py train-b0-debug \
  --manifest-dir manifests/n_imagenet_mini/supervised-v1 \
  --dataset-root /mnt/hdd1/datasets/event/n_imagenet \
  --output-dir /tmp/ebackbone-v3-b0-tiny-overfit \
  --subset-size 8 --epochs 120 --batch-size 4 --learning-rate 0.01 --seed 20260715
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

The supervised evaluation split protocol is recorded in D013. It derives a
deterministic 50-sample-per-class internal validation subset from the official
training source, keeps the remaining official-training samples as project
`train`, and reserves every official-validation sample as project `test`.
Source and project roles remain separate in the immutable manifests: project
test rows retain `source_split: "validation"` and use `split: "test"`.
The provenance makes the class-stratified rank independently reproducible as
`UTF8(selection_domain_utf8) || NUL || ASCII(decimal_seed) || NUL ||
UTF8(source_stable_sample_id)`.

`build-splits` indexes archive members directly without decoding event tensors.
It writes canonical `train.jsonl`, `validation.jsonl`, `test.jsonl`,
`provenance.json`, and `SHA256SUMS` files. Existing artifacts are verified and
left byte-for-byte untouched; changed inputs, seed, quota, or bytes are a hard
conflict. The stale bundled 1,000-class path lists are checksummed as
non-authoritative provenance inputs and never define membership.

`inspect-sample` is the read-only manifest-backed adapter inspection command.
It uses a row from an immutable manifest; it never resamples a split or extracts
the full dataset. The default dataset root is
`/mnt/hdd1/datasets/event/n_imagenet`, matching the checked-in configurations;
use `--dataset-root <path>` to point at another copy of the same release. The
command resolves the declared ZIP/TAR/NPZ member path, validates the row identity
and exact raw-NPZ payload hash, decodes the production `x`, `y`, `t`, `p`
contract, and renders the D012 production representations. It is CPU-only and
reports metadata plus tensor shapes, dtypes, and ranges only: it does not create
or run a model, batch samples, augment data, or start training.

With `--baseline b0`, inspection invokes the dedicated frame renderer and returns
the frame input only; it does not allocate voxel or time-surface tensors. B0
therefore requires `--cache off`, because the current cache schema is a full
three-representation bundle. With
`--baseline b1`, it returns the frame, voxel grid, and time surface from the
same verified raw-event fingerprint and temporal interval. `--cache off` reads
and writes no cache. `--cache on` requires an explicit `--cache-root <path>`;
cache acceptance is provenance-validated against the production renderer and
contract versions, project/source split identity, the exact raw payload hash,
and the raw-event identity, so a stale entry is never silently reused. Every
inspection requires an explicit project `--split`. Project-final-test rows are
fail-closed: inspecting `test.jsonl` requires both `--split test` and
`--allow-final-test`, and denial occurs before reading the test manifest or an
archive member.

`train-b0-debug` is intentionally limited to the first real-data B0 validation: a
deterministic 4--16 sample CPU overfit run from explicit project `train`. It materializes only
the selected production `[2,480,640]` float32 event frames in memory, uses the
explicit 68,148-parameter `compact_debug` engineering model, and writes a
checkpoint with strict reload/logit-equivalence verification. It does not
provide a full-dataset mode, access validation/test rows, augment inputs, or
read/write the representation cache. PASS requires the requested fixed-subset
accuracy and a final evaluation loss no greater than 25% of the initial loss.
This diagnostic is not the scientific B0 architecture and its results are not
valid B0 accuracy results.

`diagnose-b0-train` is the separate bounded integration check for the scientific
ResNet-18 path:

```bash
python main.py diagnose-b0-train \
  --manifest-dir manifests/n_imagenet_mini/supervised-v1 \
  --dataset-root /mnt/hdd1/datasets/event/n_imagenet \
  --output-dir /path/to/new-diagnostic \
  --subset-size 8 --batch-size 4 --max-steps 100 \
  --evaluation-interval 10 --learning-rate 0.05 \
  --momentum 0.9 --weight-decay 0 --seed 20260715 --device cpu
```

It deterministically selects only project-train samples, decodes their real NPZ
payloads, renders and collates event frames, exercises the production ResNet-18,
and verifies backward/update plus exact BatchNorm and evaluation-logit checkpoint
restoration. CPU is the default. `cuda:1` is the only permitted GPU spelling and
must be checked idle before use; `cuda:0` is rejected. The command accepts 8-16
samples, requires batch size at least 4, caps execution at 200 optimizer steps,
and never exposes a validation/final-test split argument. Its metrics are
engineering evidence only.

Production B0 training is a separate command and architecture:

```bash
python main.py train-b0 \
  --manifest-dir manifests/n_imagenet_mini/supervised-v1 \
  --dataset-root /mnt/hdd1/datasets/event/n_imagenet \
  --output-dir /path/outside-or-inside-worktree/to/new-run \
  --epochs 100 --batch-size 1 --learning-rate 0.05 \
  --momentum 0.9 --weight-decay 0.0001 \
  --seed 20260715 --num-workers 0 --prefetch-factor 2 --device cpu
```

The D014 production model is a randomly initialized 11,224,676-parameter
ResNet-18 adapted to the fixed two-channel native-resolution event frame. It
uses `Conv2d(2,64,7,stride=2,padding=3,bias=False)`, standard ResNet-18 residual
stages and BatchNorm, max-pooling, adaptive global average pooling to a
512-dimensional embedding, exactly one `Linear(512,100)` classifier, and
cross-entropy. The local implementation has no external-weight or network-loading
path. The command
uses only project train samples for optimization and the complete project
validation manifest for best-checkpoint selection. Project test manifests are
not accepted. It writes atomic best/last checkpoints, append-only JSONL batch
and epoch logs, a JSON report, top-1/top-5 metrics, throughput, peak GPU memory,
and strict reload evidence. CPU, batch size 1, and zero loader workers are the
bounded defaults; CUDA requires explicit `--device cuda`. Resume requires `checkpoint_last.pt` and an exact
match of model, optimizer, scheduler, seed, manifest hashes, renderer provenance,
and loader configuration.

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
- B1 models, fusion, or training loops

## Document order

1. `AGENTS.md`
2. `docs/PROJECT_CONTEXT.md`
3. `docs/DATA_AND_EVAL_PROTOCOL.md`
4. `docs/BASELINES_B0_B1.md`
5. `docs/PROBE_PROVIDER_CONTRACT.md`
6. `docs/DECISIONS.md`
