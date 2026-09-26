# Dual hierarchy / latent fusion study

> Status update, 2026-09-26: All three Mini runs completed; recorded results are
> in [SEACT_STUDY.md](SEACT_STUDY.md#completed-mini-results). The launch notes
> below describe the original 2026-09-24 handoff. Use the corresponding study
> artifacts for live status; this document defines the Mini architecture and
> protocol.


Status: **PASS** for implementation and bounded verification (2026-09-24 KST).
No accuracy claim is available for this study. Subsequent launch update:
hierarchy-only started on 2026-09-24 at 09:58 KST after the user's instruction
to continue; epoch 1, step 100 was verified. See
`../ebackbone_v3_dual_fusion_b64_e50_artifacts/launch_status.json` for live state.
Latent-only and dual remain unlaunched; the legacy queue remains paused.

This is a new N-ImageNet Mini supervised study. CLIP is dropped. There is no
HARDVS work, external pretraining, text loss, temporal window splitting, or
successor scheduled in the existing paused queue. All three modes train from
fresh initialization with a learned classifier and one cross-entropy loss.

## Design and comparison

The hierarchy branch reuses `HierarchyV1.forward_features`: all events pass
through the point MLP, learned point-to-voxel aggregation, temporal stages,
ordered temporal collapse and frame stage. Global spatial pooling produces
256 features. Its old internal classification layer is removed.

The latent branch renders event-frame, eight-bin voxel grid and time surface
from the same complete decoded event array. It reuses the V1 frame, temporal
voxel and depthwise surface block families at compact widths 8/16/32.
Each encoder is spatially pooled to 32 features. Concatenation produces 96
features; a small MLP `96 -> 256 -> 256`, with SiLU between layers, fuses them.
The voxel encoder retains ordered temporal collapse before spatial pooling.

Each branch has its own learned `Linear(256,256)` and `LayerNorm(256)`.
LayerNorm uses affine parameters and epsilon 1e-5; it is feature normalization,
not a unit-length constraint. The fusion is exactly

```
lambda = sigmoid(a)                # one global learned scalar, initially a=0
z = (1-lambda) * z_h + lambda * z_l # initially equal branch weights
logits = Linear(256,100)(z)
```

There is no extra normalization or correction addition after this mixture.
`hierarchy_only` and `latent_only` omit the inactive branch and scalar, keeping
the same projection, normalization and learned classifier design. Component
initialization is isolated so corresponding parameters start identically in
the ablations and dual model. No old checkpoint initializes these runs.

The new hierarchy-only control includes projection and LayerNorm. It is not
the historical hierarchy classifier experiment with an unchanged head.
Differences in parameters and compute are reported explicitly; this is a
branch ablation, not a capacity-matched study.

## What prior experiments cover

| Existing experiment | Already tested | Difference in this study |
| --- | --- | --- |
| V1 heterogeneous / same-architecture | Three full-event representation encoders; concatenation of spatial feature maps followed by 1x1 convolution and a CNN trunk | Independently pooled features, concat plus MLP, then a separate complete hierarchy branch |
| Hierarchy TS V1 | TS-derived multiplicative gate on the hierarchy's intermediate frame features | Two independently projected final branch embeddings |
| Hierarchy TS residual V2 | Content-dependent spatial gate multiplying a TS residual correction | Global scalar convex mixture; neither branch is an unchanged additive residual |
| Hierarchy TS confidence V3 | Classifier-confidence modulation of TS residual correction | No confidence controller or sample-dependent gate |

The live batch-64/cosine-50 report records best internal validation top-1 of
75.06% for hierarchy, 76.00% for TS V1 and 73.52% for TS V2. These are contextual
references, not results of this new study. The old queue is paused after V2,
before TS confidence V3. Original 100-epoch studies use another schedule;
the heterogeneous V1 report is partial at 67 epochs and is not a completed
50-epoch matched reference. No final-test result is used here.

Sources: `docs/V1_COMPARISON.md`, `docs/HIERARCHY_V1.md`,
`docs/HIERARCHY_TS_V1.md`, `docs/HIERARCHY_TS_RESIDUAL_V2.md`,
`docs/HIERARCHY_TS_CONFIDENCE_V3.md`, and the read-only artifacts
`../ebackbone_v3_hierarchy_ddp_b64_e50_artifacts/RESULTS.md` and
`../ebackbone_v3_hierarchy_ddp_b64_e50_artifacts/queue_status.json`.

## Data contract

Use the immutable `manifests/n_imagenet_mini/supervised-v1` split: 124,395
training and 5,000 internal-validation examples. Raw cache entries are
content-addressed NPZ bytes, verified against manifest SHA and byte length.
Representations are rendered again; no stale representation cache is used.
There is no test-access override in the study dataset.

| Input | Tensor per sample | Values |
| --- | --- | --- |
| Hierarchy points | `[N,4]`, float32 | x/639, y/479, normalized full-support time, 2p-1 |
| Event frame | `[2,480,640]`, float32 | log1p polarity-separated counts |
| Voxel grid | `[2,8,480,640]`, float32 | log1p linearly interpolated polarity-separated event mass |
| Time surface | `[2,480,640]`, float32 | exp(-(1-latest normalized time)/0.2); empty pixels zero |

All inputs share source identity, event subset and closed observed support.
Eight voxel bins discretize this single support; they do not split it into
independent examples. No raw events are filtered, sampled, padded, truncated,
augmented or assigned label-dependent windows. Packed points preserve every
event; the dense renderings naturally aggregate events at native pixels.

The real first training sample `train/n01440764/n01440764_10029.npz` contains
116,790 events over [0,49913]. Its frame and voxel event-mass sums are both
116,790. Frame values span [0,3.0910425], voxel [0,1.8871734], and TS [0,1].
The recorded payload hash and exact tensors are described in
`reports/dual_fusion_verification_20260923/raw_contract.json`.

## Training protocol

All modes use two ranks, 32 samples per GPU, global batch 64, accumulation 1,
50 epochs, seed 20260908, SGD learning rate 0.05, momentum 0.9, weight decay
0.0001, and cosine decay over the actual 50-epoch horizon. CUDA computation
uses BF16 autocast with FP32 parameters and FP32 hierarchy aggregation.
Shuffling and optimization settings are identical across the three modes.
Exact sharding retains every training example; the final global batch of 43
is split 22/21, with gradients normalized by its actual sample count.
Validation contains each of the 5,000 examples once and runs the unwrapped
model to avoid collectives on unequal rank batch counts.

## Parameters and compute

Executed meta-tensor profiling of the actual model at native 480x640 resolution,
batch one, and N=116,790 events gives:

| Mode | Trainable parameters | Conv/Linear GMAC/sample | Maximum allocated GiB/GPU in native probe | Maximum reserved GiB/GPU |
| --- | ---: | ---: | ---: | ---: |
| hierarchy_only | 2,737,028 | 7.374722816 | 11.697 | 12.725 |
| latent_only | 300,940 | 6.066843648 | 27.253 | 29.850 |
| dual | 3,012,269 | 13.441540864 | 37.953 | 40.586 |

Dual adds 275,241 parameters (+10.06%) and 6.066818048 GMAC (+82.27%) to
the matched hierarchy-only control. The shared classifier is counted once.
The original hierarchy body has 2,645,024 parameters; each projection plus
LayerNorm has 66,304; the compact latent encoder/MLP has 208,936; the learned
classifier has 25,700; the mixing scalar has one.

Hierarchy and dual costs include `1152*N` point-MLP MACs. Latent Conv/Linear
MACs do not vary with event count. These counts exclude interpolation,
scatter, normalization, activations, pooling, elementwise fusion, rendering,
and I/O. One MAC is approximately two FLOPs. They are not measured throughput.
The native probe is a single cold optimization step per model, not a warmed
latency benchmark or an epoch throughput measurement.

The compact latent encoders still retain dense native-resolution activations;
fewer weights do not imply lower activation memory. The observed dual peak is
under the available 49,140 MiB/device, but 64 sampled examples do not establish
the maximum memory requirement over the entire training set. Probe samples
contain 13,859 to 150,332 events. Training records actual event-count summaries.

## Verification and artifacts

Consolidated focused regressions: **41 passed in 18.00 seconds**. Independent
review separately passed all 17 new tests and found no actionable correctness
defects. Preservation check: **393 protected artifacts and 132 existing source
files unchanged**; the existing queue remains `paused` before TS confidence V3.

- Unit and integration checks cover alignment, label independence, closed
  endpoints and zero-duration input, packed isolation, all-stage gradients and
  updates, scalar gradients, weighted-fusion endpoints, absent inactive modules,
  identical component initialization, and the original hierarchy body weights.
- Two-process CPU Gloo checks exercise each mode, uneven 2/1 tail batches,
  global-mean gradient equivalence, exact split coverage, unequal validation
  batch counts, and exact epoch-boundary model/optimizer/scheduler/RNG resume.
- Strict checkpoint and classifier-free backbone reloads are checked against
  live model state and outputs. Exports contain construction and input-contract
  metadata and include the learned mixing scalar for dual.
- Native BF16/NCCL verification used both GPUs and one real optimization step
  per mode: local batch32/global64/accumulation1. Every active component updated;
  rank gradients agreed exactly. Checkpoint logits and backbone embeddings
  were bit-exact after reload on each individual GPU. Dual lambda moved from
  0.5 to 0.49997246. These are disposable verification weights, not pretraining.
- Existing artifact metadata and control-file hashes are compared before/after;
  existing source-file hashes are also compared. Checkpoint preservation checks
  use file size and nanosecond mtime, not a claim to rehash every large checkpoint.

All verification artifacts are under
`reports/dual_fusion_verification_20260923/` (directory named when work began):
`raw_contract.json`, `config_and_compute.json`, `pytest.xml`,
`native_b32/report.json`, `preservation_before.json`, and
`preservation_after.json`. Independent review: `docs/DUAL_FUSION_REVIEW.md`.

Reproduce read-only inspection:

```bash
cd /mnt/ssd1/PycharmProjects/ebackbone_V3
PYTHONNOUSERSITE=1 /home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python \
  -m ebackbone_v3.dual_fusion_train inspect \
  --config configs/dual_fusion.n_imagenet_mini_b64_e50.json
```

Reproduce the bounded native probe with a **new** output directory:

```bash
cd /mnt/ssd1/PycharmProjects/ebackbone_V3
export CUDA_VISIBLE_DEVICES=GPU-25c7cbbd-0ce6-3d44-3397-a0dc30f9b62a,GPU-4fb73c12-68f6-7e96-db96-2ec175350543
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONNOUSERSITE=1
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m ebackbone_v3.dual_fusion_verify \
  --output-dir reports/dual_fusion_verification_repeat/native_b32
```

Focused regression command (Gloo requires host localhost access):

```bash
GLOO_SOCKET_IFNAME=lo PYTHONNOUSERSITE=1 \
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m pytest -q \
  tests/test_dual_fusion_models.py tests/test_dual_fusion_train.py \
  tests/test_hierarchy.py tests/test_v1_models_data.py \
  tests/test_hierarchy_ddp.py tests/test_hierarchy_ddp_queue.py
```

## Prepared launch commands — not executed

Run each variant independently after training authorization, from the repository
directory with the same CUDA and CUBLAS environment shown above. Each starts
fresh and refuses an existing output directory. No command resumes the paused
legacy queue or schedules any other experiment.

```bash
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m ebackbone_v3.dual_fusion_train train \
  --config configs/dual_fusion.n_imagenet_mini_b64_e50.json \
  --variant hierarchy_only \
  --output-dir /mnt/ssd1/PycharmProjects/ebackbone_v3_dual_fusion_b64_e50_artifacts/runs/hierarchy_only

/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m ebackbone_v3.dual_fusion_train train \
  --config configs/dual_fusion.n_imagenet_mini_b64_e50.json \
  --variant latent_only \
  --output-dir /mnt/ssd1/PycharmProjects/ebackbone_v3_dual_fusion_b64_e50_artifacts/runs/latent_only

/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m ebackbone_v3.dual_fusion_train train \
  --config configs/dual_fusion.n_imagenet_mini_b64_e50.json \
  --variant dual \
  --output-dir /mnt/ssd1/PycharmProjects/ebackbone_v3_dual_fusion_b64_e50_artifacts/runs/dual
```

`--resume` is only for one of these study outputs with matching source,
configuration and manifest identity. Best and last checkpoints include model,
optimizer, scheduler, per-rank RNG, epoch/history and provenance. Final exports
are `backbone_best.pt` and `backbone_last.pt`, reloaded with
`dual_fusion_models.load_backbone_export`; their output is `[B,256]` embeddings.

## Changed files and next action

New files only: `ebackbone_v3/dual_fusion_models.py`, `dual_fusion_data.py`,
`dual_fusion_train.py`, `dual_fusion_verify.py`;
`configs/dual_fusion.n_imagenet_mini_b64_e50.json`;
`tests/test_dual_fusion_models.py`, `tests/test_dual_fusion_train.py`;
this study document, the independent review document, and verification reports.
Existing source, runs, checkpoints and paused-queue files are preserved.

Current bounded action: the matched `hierarchy_only` 50-epoch control is running
in its new output directory. No successor is scheduled.
