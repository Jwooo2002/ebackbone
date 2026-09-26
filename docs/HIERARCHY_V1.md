# Hierarchy-only V1 on N-ImageNet Mini

Implemented as an additive experiment on 2026-09-08, under the user's explicit
point -> voxel -> frame hierarchy request. The existing parallel V1 source,
configuration, snapshot, running process and output directory are not modified.
This document defines a separate architecture; it does not replace D017.
GPU verification and a true-batch 8/16/32 benchmark passed on 2026-09-08.
The authorized 100-epoch run is supervised from the isolated snapshot; consult
`/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_v1_artifacts/main_launch_status.json`
for current launch/completion state. All parameters start from random initialization.

## Input and aggregation contract

Use the existing immutable Mini supervised-v1 manifests: 124,395 training
samples and 5,000 internal-validation samples. Each sample uses its complete
stored event array and closed observed temporal support. Labels enter only
the final cross-entropy and existing split bookkeeping. No event filtering,
sampling, augmentation, padding, truncation, resizing or raw-image rendering.
The existing content-addressed raw NPZ cache is read with byte-count and SHA
validation. No representation cache is used, and no final-test examples are
decoded or evaluated by these commands.

Prepare float32 event rows `[x/639, y/479, (t-start)/duration, 2*p-1]`.
Zero-duration samples use normalized time one. Preserve each event's continuous
position/time in the MLP input before aggregation. Polarity is an input feature,
not two separate voxel channels. Packed batching concatenates all rows into
`[M,4]`, where `M=sum_i N_i`, with int64 event counts `[B]`. Routing consists of
lower/upper int64 voxel indices `[M]` and float32 upper weights `[M]`; the
collator offsets indices to keep samples separate. No points cross sample
boundaries during voxel aggregation.

Spatial routing uses `floor(x/4), floor(y/4)` into 120x160 cells. Time uses
linear interpolation between the two neighbors of `7*t_normalized` (eight
bins). The same weights scatter the learned point vectors and unit event mass.
For every cell, output the weighted feature mean (zero for empty cells) and
`log1p` weighted event mass. Accumulation and division use FP32 even under
BF16 autocast. This is a learned feature voxel grid; no independent raw voxel
encoder or raw-frame stem exists. Its 33rd channel preserves local event mass
that would otherwise be lost through mean aggregation.

The point MLP is pointwise and does not include a neighborhood graph or point
attention. Local interactions begin with voxel aggregation and convolutions.

## Exact architecture

Shapes below use batch B and all packed events M. Native input H=480, W=640.

| Stage | Operation | Output | Parameters |
| --- | --- | --- | ---: |
| Points | Linear 4->32, LN(32), SiLU, Linear 32->32, LN(32), SiLU | `[M,32]` | 1,344 |
| Point-to-voxel | Weighted mean plus log1p event mass; spatial cell 4x4, 8 temporal bins | `[B,33,8,120,160]` | 0 |
| Voxel projection | Conv3d 33->32, kernel 1, GN, SiLU | `[B,32,8,120,160]` | 1,120 |
| Voxel stage 1 | TemporalBlock 32->64, spatial stride 2 | `[B,64,8,60,80]` | 82,560 |
| Voxel stage 2 | TemporalBlock 64->128, spatial stride 2 | `[B,128,8,30,40]` | 328,960 |
| Temporal collapse | Flatten C-major/bin-minor to `[B,1024,30,40]`; Conv2d 1024->128, kernel 1, GN, SiLU | `[B,128,30,40]` | 131,328 |
| Frame stage | BasicBlock 128->256, stride 2; BasicBlock 256->256, stride 1 | `[B,256,15,20]` | 2,099,712 |
| Pool | Spatial global average pooling | `[B,256]` | 0 |
| Classifier | Linear 256->100 | `[B,100]` | 25,700 |

Each TemporalBlock contains two spatial/temporal pairs: Conv3d 1x3x3,
GN, SiLU, Conv3d 3x1x1, GN. SiLU separates the pairs; residual addition and
SiLU finish the block. Only the first spatial convolution takes stride 2;
all temporal strides equal one. A 1x1x1 projection plus GN is the shortcut.
Each BasicBlock has two 3x3 Conv2d layers with GN, SiLU after the first and
after residual addition, and a 1x1 projection/GN shortcut when needed.
Convolutions have no bias and symmetric half-kernel padding. GN uses 8 groups
for these widths, epsilon 1e-5 and affine parameters. Point LN uses epsilon
1e-5 and affine parameters. Linear layers include bias. There is no dropout.

The reusable `forward_features` output is `[B,256,15,20]`. Learned ordered
temporal collapse creates the first frame-level feature map; it is not a raw
event frame and does not average temporal bins. A single final classification
head receives CE supervision. No TS, gating, top-down refinement, auxiliary
head/loss, SSL or external weights are included.

## Parameters and MAC accounting

Total trainable parameters: **2,670,724**.

Per sample with N raw events:

`MACs(N) = 7,240,115,200 + 1,152*N`

For the freshly probed first training sample
`train/n01440764/n01440764_10029.npz`, N=116,790, observed interval
[0,49,913] microseconds: **7,374,657,280 MACs = 7.37465728 GMACs**.
The packed point tensor is `[116790,4]` with float32 range [-1,1].
Its raw payload SHA is
`18dc7a5074f0a25f91cb533de83818722be1bd1a25a50be9c9fc857bc92fa69e`.

The profiler executes meta shape propagation through the actual model and
counts all Conv2d, Conv3d and Linear multiply-accumulates (one MAC is roughly
two FLOPs). It excludes normalization, activations, interpolation/scatter,
division, pooling, preprocessing and I/O. The scalar is not a measured latency
or dataset-average cost. Epoch reports record event-count min/max/mean so the
mean counted MACs can be derived. The hierarchical model is not parameter- or
compute-matched to the parallel V1 baseline; the comparison changes both
aggregation structure and capacity/compute.

## Training command and protocol

Use seed 20260908, 100 epochs, true batch 32 and accumulation 1, SGD lr 0.05,
momentum 0.9, weight decay 1e-4, cosine schedule to zero after 100 epochs,
BF16 autocast with FP32 parameters, cuda:1, four loader workers and four CPU
threads. A dedicated seed+epoch generator reproduces baseline sample order.
Accumulation is normalized by actual sample count, including the final partial
group. Select the best checkpoint by complete internal-validation top-1,
then lower CE; report top-5 as well. Only final classifier CE is optimized.

The run uses the frozen snapshot of this configuration. The equivalent source-tree command is:

```bash
cd /mnt/ssd1/PycharmProjects/ebackbone_V3
CUDA_VISIBLE_DEVICES=GPU-4fb73c12-68f6-7e96-db96-2ec175350543,GPU-25c7cbbd-0ce6-3d44-3397-a0dc30f9b62a \
PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -u -m ebackbone_v3.hierarchy train \
  --config /mnt/ssd1/PycharmProjects/ebackbone_V3/configs/hierarchy_v1.n_imagenet_mini.json \
  --raw-cache /mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts/raw_cache \
  --output-dir /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_v1_artifacts/main_seed20260908
```

The user authorized use of an available device. Physical GPU 0 (RTX 6000 Ada)
was idle; the baseline remains on physical GPU 1. CUDA UUID visibility ordering
maps logical `cuda:1` to physical GPU 0 only for the hierarchy processes. The
configuration's logical device and all architecture code remain unchanged.

The user superseded the initial batch-8/accumulation-4 proposal with a true-batch
8/16/32 benchmark. Each candidate passed three runs on the same 320 hash-ranked
training samples (64 warmup, 256 timed), without accumulation. Median end-to-end
throughput was 138.335 / 135.454 / 136.143 samples/s for batches 8 / 16 / 32.
Peak allocated memory was 3,321,812,992 / 6,522,556,416 / 12,695,256,064 bytes;
peak reserved memory was 4,987,027,456 / 9,372,172,288 / 17,838,374,912 bytes.

Select the largest stable candidate within 5% of the best median throughput
while reserving at most 80% of device memory: **batch 32, accumulation 1**.
The learning rate remains 0.05; no scaling or other optimizer/scheduler change.
This is the initial hierarchy experiment, not a requirement to match an existing
hierarchy baseline. Earlier parallel baselines retain their original protocol.
Every candidate produced BF16 logits, FP32 voxel accumulation, finite nonzero
gradients, parameter updates in every stage, and bit-exact checkpoint reload.
Checks cover the observed sample set, not a worst-case full-dataset event count.
Benchmark checkpoints are never loaded by production training.

The detached supervisor launches once, watches the hierarchy process, and then
waits for all four existing baselines to finish. It generates validation tables,
CSV curves, PNG/PDF plots and `comparison/REPORT.md` under the hierarchy artifact
root. Only five verified complete 100-epoch histories permit a final comparison.
Failures are recorded in `main_launch_status.json`; runs are not silently retried
with changed settings. No final-test evaluation is scheduled.

Use `--resume` only with matching source/configuration/manifests, or
`--stop-after-epoch N` to stop at an epoch boundary without changing the
100-epoch cosine budget. Atomic best/last checkpoints store model, optimizer,
scheduler, RNG, history, source hashes, point contract and manifest hashes.
Strict reload checks parameter equality and evaluation logits. Exact resume
state is tested with single-thread CPU execution. Independent multithread CPU
runs showed roundoff-level variation; bitwise multi-thread or CUDA training
reproducibility is not claimed.

## Verification commands and artifacts

Full repository suite: **158 passed in 27.93 seconds**, including eight new
hierarchy tests. Tests cover raw identity/label independence, temporal endpoints,
event-mass conservation, packed sample isolation, all-stage CE gradients and
updates, strict reload, event-dependent MACs, exact single-thread epoch resume,
partial-group accumulation and invalid configuration/final-test rejection.

Run from `/mnt/ssd1/PycharmProjects/ebackbone_V3`:

```bash
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m pytest -q
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m ebackbone_v3.hierarchy profile --events 116790
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m ebackbone_v3.hierarchy probe \
  --raw-cache /mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts/raw_cache
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m ebackbone_v3.hierarchy diagnose \
  --samples 2 --epochs 1 \
  --raw-cache /mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts/raw_cache \
  --output-dir /tmp/hierarchy-v1-new-diagnostic
```

The completed native-resolution CPU diagnostic uses two deterministic training
samples, all 188,557 events, two SGD updates, and a train-only recheck. All seven
parameterized stages have finite nonzero first-update gradient norms. The
checkpoint reload is strict and reproduces evaluation logits bit-exactly.
Peak process RSS was 1,229,983,744 bytes. This is engineering evidence, not an
accuracy/overfit result or a production GPU memory measurement.

Artifacts in `/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_v1_artifacts`:

- `real_sample_probe.json`: raw identity, normalized input/routing statistics,
  exact profile and stage tensor shapes.
- `diagnostic_cpu_seed20260908/report.json`: native real-data optimizer and
  reload results, gradient norms, full source fingerprints and event counts.
- `diagnostic_cpu_seed20260908/checkpoint_last.pt`: verified diagnostic model,
  optimizer, scheduler, RNG and provenance; not initialization for the main run.
- `verification.json`: repository tests and baseline file-preservation audit.

All implementation files are new: `ebackbone_v3/hierarchy_models.py`,
`hierarchy_data.py`, `hierarchy_training.py`, `hierarchy.py`,
`configs/hierarchy_v1.n_imagenet_mini.json`, `tests/test_hierarchy.py`, and this
document. The isolated runner retains the existing optimizer/scheduler protocol
without changing any baseline imports or source hashes.

Next action: verify ongoing training and inspect the automatically generated validation comparison after all five runs complete.
