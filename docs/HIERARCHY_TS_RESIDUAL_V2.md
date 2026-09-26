# Content-aware residual TS fusion V2

Status: CPU/GPU preflight PASS; 100-epoch training launched on 2026-09-11, PID 3710484. First optimizer update verified. See main_launch_status.json for live status. Existing hierarchy-only, multiplicative TS V1, and parallel baselines remain unchanged.

## Decision and evidence

The user requested changing the approach and explicitly chose to preserve the point-to-voxel-to-frame hierarchy while redesigning TS integration. At epoch 55 the TS model and hierarchy-only model both have about 93.05% training accuracy, but validation top-1 is 68.14% vs 69.80%. The TS gate is active: its first validation batch changes the feature norm by about 28%, with about 1% of gate values near the bounds. These measurements do not establish why generalization differs, do not describe all validation batches, and are not final V1 results.

Hypothesis: let the hierarchy content control the use of a separate TS correction, rather than deriving multipliers solely from TS. This changes both the fusion operation and its conditioning; a gain would validate this combined V2 design, not isolate either change individually. Learned input-dependent fusion weights are an established approach; see [Gated Multimodal Units for Information Fusion](https://arxiv.org/abs/1702.01992). Our inputs remain representations of the same raw events, not independent sensor modalities. This implementation is a residual TS variant, not a reproduction of that paper.

## Exact architecture

The original hierarchy is unchanged. It produces F [B,128,30,40] after its learned ordered eight-bin temporal collapse. The existing TS renderer and four-block TS CNN are reused unchanged, producing T [B,64,30,40]. TS uses separate negative/positive channels, latest timestamp per pixel, exp(-(1-last_normalized_time)/0.2), background 0, and occupied value 1 for zero duration. Both inputs use exactly the same decoded raw sample and time interval.

Correction D = Conv1x1(T), 64->128 with bias: [B,128,30,40].
Weight W = sigmoid(Conv1x1(SiLU(GN8(Conv1x1(concat(F,T)))))):
concat [B,192,30,40] -> bias-free Conv192->32 -> GN8 -> SiLU -> biased Conv32->1 -> sigmoid [B,1,30,40].
Output = F + W * D, broadcasting W over the 128 correction channels.
The existing frame residual CNN, spatial pooling, Linear256->100, and single supervised cross-entropy follow.

The weight is a learned spatial fusion coefficient, not calibrated uncertainty. D can add or subtract channel features, including where F is zero. F is an explicit unchanged summand; the correction can still learn to harm useful features, so a performance gain is not guaranteed. No top-down refinement, new representation, multi-scale decay, auxiliary loss, SSL, external pretraining, or extra raw-frame stem is added.

## Initialization and gradient checks

Initialize correction projection weight/bias to zero. Initialize the confidence network normally, avoiding a doubly zero branch. The initial output is exactly F for every TS input. The hierarchy and TS encoder initial weights and global CPU RNG state equal their V1 counterparts at the same seed. During first backward only the correction projection receives data gradients within the new fusion branch; confidence network and TS encoder receive data gradients after the correction update. Tests explicitly verify this two-step behavior.

## Compute

2,745,829 parameters: 6,241 more than multiplicative TS V1, 75,105 more than hierarchy-only V1. At 116,790 events, 7.63520128 GMACs: 0.0074112 more than TS V1 and 0.260544 more than hierarchy-only. Executed meta shape propagation counts Conv2d/Conv3d/Linear MACs only; rendering, scatter, activations, normalization, pooling and elementwise fusion are excluded. Three BF16 batch32 repetitions passed. Median 128.572 samples/s; allocated 11.897 GiB, reserved 16.104 GiB. Each used 64 warmup and 256 timed real samples. BF16 autocast keeps normalization, scatter and residual feature tensors FP32; classifier logits are BF16. Both TS encoder and confidence network receive gradients after the residual update; all stages update and reload logits are bit exact.

## Implementation

- ebackbone_v3/hierarchy_ts_residual_models.py: residual fusion, model, and profile.
- ebackbone_v3/hierarchy_ts_residual_training.py: isolated runner with original optimizer/checkpoint protocol.
- ebackbone_v3/hierarchy_ts_residual.py: profile, probe, diagnose and train CLI.
- configs/hierarchy_ts_residual_v2.n_imagenet_mini.json: unchanged training settings.
- tests/test_hierarchy_ts_residual.py: identity, two-input conditioning, additive capacity, two-step gradients, packing, shapes/cost, checkpoint and exact resume tests.

## Proposed training protocol and commands

Same manifests, seed20260908, no augmentation, fresh initialization, 100 epochs, true batch32, accumulation1, BF16, SGD lr0.05 momentum0.9 weight_decay0.0001, cosine schedule, workers4, CPUthreads4. GPU preflight passed; full training is authorized. The main run starts from fresh initialization and never reuses benchmark/diagnostic/V1 checkpoints. Source is frozen in main_snapshot_seed20260908 and original experiment snapshots are hash-checked at launch.

From /mnt/ssd1/PycharmProjects/ebackbone_V3:

```bash
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m ebackbone_v3.hierarchy_ts_residual profile
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m ebackbone_v3.hierarchy_ts_residual diagnose --samples 2 --epochs 1 --raw-cache /mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts/raw_cache --output-dir /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ts_residual_v2_artifacts/diagnostic_cpu
```

The diagnostic already exists; the runner refuses to overwrite it. Verification artifacts are stored in /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ts_residual_v2_artifacts. No training checkpoint from the diagnostic will initialize a main run.

## Exact main training command

```bash
CUDA_VISIBLE_DEVICES=GPU-4fb73c12-68f6-7e96-db96-2ec175350543,GPU-25c7cbbd-0ce6-3d44-3397-a0dc30f9b62a PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ts_residual_v2_artifacts/main_snapshot_seed20260908 /home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -u -m ebackbone_v3.hierarchy_ts_residual train --config /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ts_residual_v2_artifacts/main_snapshot_seed20260908/configs/hierarchy_ts_residual_v2.n_imagenet_mini.json --raw-cache /mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts/raw_cache --output-dir /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ts_residual_v2_artifacts/main_seed20260908
```

172 tests passed. Local curves refresh after each epoch. The stopped TS V1 run remains explicitly labeled stopped_early. W&B uploads are enabled through a separate V2 logger in the same project.
