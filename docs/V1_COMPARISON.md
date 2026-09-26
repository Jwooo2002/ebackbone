# V1: heterogeneous N-ImageNet Mini comparison

Implementation is complete and CPU/GPU engineering checks pass. The full
100-epoch-per-model comparison was **launched on 2026-09-08** on physical GPU 1
(`cuda:1`, RTX 6000 Ada). The earlier unavailable-CUDA finding was caused by
sandbox device isolation; host execution sees both indices 0 and 1 with no
visibility remapping. Full comparison accuracy is pending job completion.
Diagnostic accuracy on two training samples is not a scientific result.

## Selected experiment

The user selected **N-ImageNet Mini, 100 classes** on 2026-09-08. The main
comparison uses all 124,395 project-training samples and all 5,000 internal
validation samples in the existing immutable `supervised-v1` manifests. The
official validation source remains reserved as project final test. Training,
raw-cache preparation, diagnostics, and checkpoint selection never open that
test manifest or its events. No full 1,000-class accuracy claim is intended.

This is a new V1 comparison family. Its frame baseline uses the same spatial
branch and post-branch trunk as V1; it is distinct from the existing D014
ResNet-18 B0. D012/D014, historical results, and their commands remain unchanged.

## Raw events and representations

One complete stored NPZ `event_data` array is one sample. Packed fields are
`x:uint16`, `y:uint16`, `t:uint16`, `p:bool`, with false=negative, true=positive.
Coordinates retain the native 480x640 grid. The temporal window is closed
observed support `[t.min(),t.max()]`. No events are filtered, sampled, cropped,
resized, augmented, padded or truncated. Labels enter classification and the
existing split bookkeeping only.

All three views derive from those exact fields and share their event fingerprint,
payload SHA-256, event count, temporal interval and sample ID:

| View | Shape without batch | Values |
| --- | --- | --- |
| Frame | `[2,480,640]` | `log1p` counts, negative then positive |
| Voxel | `[2,8,480,640]` | Linear temporal interpolation of nonnegative event mass, then `log1p` |
| Surface | `[2,480,640]` | `exp(-(t_end-t_last)/(0.2*duration))`; unoccupied pixels zero |

A zero-duration nonempty sample maps every event to bin 7 and gives occupied
surface pixels value one. V1 rendering has its own version and fingerprint;
D012's 5-bin/0.3 tensors and representation caches are never accepted here.
Frame-only loading does not allocate voxel or surface tensors.

The optional shared cache stores **exact raw NPZ bytes**, keyed by manifest
payload SHA-256, not rendered tensors. Every read verifies size and SHA-256.
`prepare-raw` streams each class TAR once, caches train and internal validation
only, and verifies existing entries on repeat. Missing/corrupted cached payloads
fail closed. Direct archive loading remains available without a cache, but
repeatedly streaming a class TAR is costly for a full run.

## Architecture

All branches have an initial spatial stride-2 stem, followed by three stages
with widths 32/64/128, two residual blocks each, and stage spatial strides 1/2/2.
Convolution padding preserves size except at declared stride changes. All
convolutions are bias-free; normalization is GroupNorm with `gcd(8,C)` groups,
and activation is SiLU. Projection shortcuts handle width/stride changes.

- Frame: 3x3 2D stem; two 3x3 convolutions per basic residual block.
- Heterogeneous voxel: 1x3x3 stem; each block contains two pairs of spatial
  1x3x3 and temporal 3x1x1 convolutions, with no temporal strides. After the
  final stage, flatten `[128,8,H/8,W/8]` in channel-major/bin-minor order and
  project 1024 to 128 channels with 1x1 convolution, GroupNorm and SiLU.
- Heterogeneous surface: 3x3 2D stem; each residual block contains a depthwise
  5x5 convolution and pointwise 1x1 convolution. This lighter local encoder is
  an experimental choice whose benefit must be measured.

Concatenate the three 128-channel spatial maps. Project 384 to 256 with a 1x1
convolution, GroupNorm and SiLU, then apply two standard 256-channel 2D residual
blocks. `forward_features` returns `[B,256,60,80]` at native input resolution.
Global spatial average pooling and one `Linear(256,100)` produce class logits.
The classifier uses a bias. All parameters start randomly; no weight loading,
teacher, SSL, auxiliary loss, alignment loss, attention, or external pretraining
is involved. Multi-view models use all three inputs at inference too.

## Four controlled models

| CLI name | Branches | Parameters | GMAC/sample |
| --- | --- | ---: | ---: |
| `frame` | One 32/64/128 spatial branch; 128-to-256 projection and common trunk | 3,115,364 | 19.3904896 |
| `same_arch` | Three independent spatial branches; voxel `[2,8,H,W]` flattened to 16 input channels | 4,575,012 | 35.8318336 |
| `heterogeneous` | Spatial frame, spatial-temporal voxel, depthwise surface | 4,321,540 | 106.9600000 |
| `frame_matched` | One 112/224/448 spatial branch; 448-to-256 projection and common trunk | 10,991,844 | 108.3678976 |

`same_arch` has identical branch body architecture, independent weights, and
input stems adjusted only for channel count. All models retain the same
256-channel post-projection trunk and classifier. The matched frame width is
chosen before training from multiples of 8 in [32,160], minimizing relative
MAC error against heterogeneous V1. Error is **1.3163%**, below the required 5%.
This is compute matching, not parameter matching or latency matching.

MAC counts use meta-tensor shape propagation and hooks over every Conv2d,
Conv3d and Linear at `[H,W]=[480,640]`, batch one. One multiply-accumulate is one
MAC (approximately two FLOPs). Normalization, activation, pooling, rendering
and archive I/O are excluded. Their costs are not claimed to be matched.

## Shared training and selection

The single checked-in configuration applies to every architecture:

- Seed 20260908; random initialization; 100 complete epochs.
- Batch 8, accumulation 4, effective batch 32 except the final partial group.
- SGD: learning rate 0.05, momentum 0.9, weight decay 0.0001.
- CosineAnnealingLR to zero across 100 epochs; step after each epoch.
- Cross-entropy on fused logits only; no augmentation, label smoothing or clipping.
- CUDA device 1, BF16 autocast, FP32 parameters and optimizer state; four workers.
- A dedicated epoch-seeded loader generator, independent of model RNG. All
  samples are used, with `drop_last=False`; accumulation normalizes by actual
  group sample count, including partial batches/groups.
- Select highest complete internal-validation top-1; ties use lower CE loss;
  remaining ties retain the earlier epoch. Also report top-5.

Training and validation record ordered sample-ID SHA-256 and complete counts.
The comparison rejects mismatched settings, renderer/manifests, or training
sample order. Train/validation sample identities are checked disjoint; any
identical raw payload hashes across them are reported without silently changing
the established split. Final-test evaluation is deferred until training and
checkpoint selection complete; there is no V1 final-test CLI in this change.

Atomic last/best checkpoints include model, optimizer, scheduler, RNG, epoch,
history, complete run configuration, source-code hashes and manifest hashes.
Resume requires an exact identity match and restores full continuation state.
At completion or an explicit epoch boundary stop, strict checkpoint loading
must reproduce every model tensor and evaluation logits exactly. This is
tested against uninterrupted training, including optimizer and scheduler state.

Efficiency reports separate decoding, rendering, model forward time, and
end-to-end epoch throughput. GPU peak allocated memory is reset per model.
CPU peak RSS is explicitly labeled process-lifetime, not isolated model memory.
Timings from tiny diagnostics are engineering observations, not benchmark
latency claims. A single seed does not establish statistical significance.

## Commands

Run from `/mnt/ssd1/PycharmProjects/ebackbone_V3`. The installed environment
`/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python` provides PyTorch 2.7.1+cu128
and NumPy 1.26.4 and passes the repository tests. Its name does not imply any
SSMER/Event2Vec model or weights are used.

```bash
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m pytest -q
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m ebackbone_v3.v1 profile
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m ebackbone_v3.v1 probe
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m ebackbone_v3.v1 diagnose \
  --samples 2 --epochs 1 --output-dir /tmp/v1-new-diagnostic
```

For full training, first restore and verify idle CUDA device 1. Raw caching is
optional but recommended for archive throughput; choose a cache with sufficient
space for the complete training-source NPZ payloads. Neither command overwrites
dataset archives or resplits samples.

```bash
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m ebackbone_v3.v1 prepare-raw \
  --raw-cache /mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts/raw_cache
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m ebackbone_v3.v1 compare \
  --raw-cache /mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts/raw_cache \
  --output-dir /mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts/main_seed20260908
```

Use `--resume` with the same configuration and output to continue completed
epoch checkpoints. `--stop-after-epoch N` preserves the original 100-epoch
scheduler budget while returning after epoch N. All four models run sequentially
in the same process. Existing nonempty run directories are refused without
resume. Interrupted comparisons resume existing model checkpoints and start
the remaining models under the same configuration.

## Verification and limitations (2026-09-08)

- `/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m pytest -q`:
  **150 passed in 23.97 seconds**, including all 17 new V1 tests.
- Authoritative native-resolution diagnostic artifacts:
  `/mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts/diagnostic_torch271_20260908/comparison.json`.
  Per-model directories contain complete settings, history, checkpoints, source
  hashes and strict reload verification. The source hashes identify the tested
  implementation before the later progress-logging addition. The shared
  diagnostic training-order SHA-256 is
  `8c557f627c128479da5c74dd56cae1769cd7db731496002409f4c804f3b3797c`.
- Real sample `train/n01440764/n01440764_10029.npz`: 116,790 events, 0 to 49,913 us,
  payload SHA-256 `18dc7a5074f0a25f91cb533de83818722be1bd1a25a50be9c9fc857bc92fa69e`.
  All three V1 tensors are finite float32. Frame tensors are bit-identical
  between single-view and three-view loading, with the same raw fingerprint.
- Renderer tests check independent numerical expectations for interpolation,
  count conservation, polarity, last-event recency, zero background and
  zero-duration samples. Identity mismatches are rejected.
- Model tests check gradients and updates in every branch, fusion, trunk and
  classifier; strict reload and spatial output; independent branch weights;
  input rejection; MAC matching without consuming model initialization RNG.
- Data tests exercise real archive/manifest fixture decoding, train/validation
  isolation, final-test rejection, raw cache idempotence and corruption checks.
- Runner tests check complete four-way execution, identical order, exact epoch
  resume, partial accumulation equivalence and fail-closed configuration/resume.
- Native-resolution, two-train-sample diagnostics pass for all four models.
  These are not overfit or representation-quality claims.
- Full comparison launch stops before dataset access or run creation when
  `cuda:1` is unavailable in the execution environment. A sandbox check alone
  cannot establish host CUDA availability. No final-test accuracy exists yet.
- The base Python environment has older PyTorch 2.0.1: V1 checks pass there,
  but 16 existing repository tests fail on unsigned dtype/itemsize/CPU autocast
  compatibility. Use the verified PyTorch 2.7.1 environment for repository work.

## Host launch and live artifacts (2026-09-08)

Both physical indices are visible outside the sandbox. Logical `cuda:1` was
verified by PyTorch UUID as physical GPU 1,
`GPU-4fb73c12-68f6-7e96-db96-2ec175350543`; it had no compute process before
launch. `CUDA_VISIBLE_DEVICES` and `NVIDIA_VISIBLE_DEVICES` were unset.
Allocation, finite gradients, a real optimizer update at batch 8/accumulation 4,
and bit-exact checkpoint reload passed for all four models. Heterogeneous V1
peaked at 28,994,975,744 allocated bytes in that GPU preflight.

The raw cache contains 129,395 verified NPZ payloads (43,819,681,928 bytes);
preparation did not access final-test samples. The detached comparison started
with supervisor PID 3078931 and training PID 3078981. Initial verified progress
was frame epoch 1, update 100, 3,200 samples, finite batch CE 4.688055, and 93%
GPU utilization. These are early progress observations, not a completed epoch
or accuracy result; consult the logs for current state.

Artifacts under `/mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts`:

- `main_launch_status.json`: supervisor stage, PID, exact command and device mapping.
- `main_seed20260908.log`: optimizer progress and epoch metrics.
- `main_seed20260908/`: per-model configuration, checkpoints and comparison results.
- `main_snapshot_seed20260908/`: frozen Python sources and configuration used by
  the job. Only the config's manifest location was resolved to the identical
  absolute path; all scientific training settings remain unchanged.
- `gpu_preflight_20260908/report.json`: four-model native GPU step/reload evidence.
- `raw_cache_prepare_host.log`: successful complete raw-cache preparation.

Progress logging was added to `v1_training.py` at the first, every 100th and final
optimizer update of each epoch. The five runner tests, including exact resume
and accumulation equivalence, passed after this logging-only change. The run
snapshot captures it. Supervisor state becomes `complete` only after all four
models report 100 completed epochs; a nonzero process exit records `failed`.

Next action: check the first completed epoch and checkpoint in the running job.

## Changed files

All paths below are within `/mnt/ssd1/PycharmProjects/ebackbone_V3`.

| Path | Change |
| --- | --- |
| `ebackbone_v3/v1_models.py` | Four architectures and MAC matching |
| `ebackbone_v3/v1_data.py` | Versioned renderer, manifest adapter and verified raw cache |
| `ebackbone_v3/v1_training.py` | Common training, comparison, metrics and full-state resume |
| `ebackbone_v3/v1.py` | Separate profile/probe/diagnose/cache/compare CLI |
| `configs/v1.n_imagenet_mini.json` | Shared scientific run settings |
| `tests/test_v1_models_data.py` | Model, rendering, raw cache and split tests |
| `tests/test_v1_training.py` | Runner, identical-order, accumulation and resume tests |
| `docs/V1_COMPARISON.md` | Architecture, protocol, evidence and commands |
| `docs/DECISIONS.md` | D017 records the accepted V1 design |
| `README.md` | Exposes the new V1 entry point |

Pre-existing uncommitted changes in `ebackbone_v3/b0_production.py` and
`ebackbone_v3/cli.py` were preserved and are not part of this implementation.
