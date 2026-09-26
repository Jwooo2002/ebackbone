# TS confidence-controlled residual V3

Status: preflight and launch verification PASS; 100-epoch fresh full-model training launched 2026-09-13 on physical GPU0, PID20940. First optimizer update verified. W&B uploader launched after explicit user approval and verified through API readback: https://wandb.ai/jwoh3923-/ebackbone-v3-n-imagenet-mini/runs/v1abd01d991aad41. Live launch status is in main_launch_status.json. No V1/V2 checkpoint initializes this model.

## Design

The frozen two-model router reached 81.62% versus hierarchy-only 80.00% and V2 79.08% on internal validation. This motivates a single-model confidence-control ablation; it does not predict V3 accuracy. All hierarchy and TS parameters train from scratch, as requested. The point→voxel→frame hierarchy, TS renderer/CNN, spatial correction network and final supervised CE remain as in V2. No raw-frame stem, auxiliary loss, SSL or separate classifier is added.

Let F be hierarchy features after temporal collapse; T the TS CNN output; H the existing shared frame stage, spatial pooling and classifier. Compute:

```
z_base = stopgrad(H(F))
m = top1(softmax(z_base)) - top2(softmax(z_base))
tau = sigmoid(theta)                  # theta initialized 0; tau starts at 0.5
s = sigmoid((tau - m) / 0.1)          # [B,1,1,1]
W = sigmoid(q(concat(F,T)))           # [B,1,30,40]
D = Conv1x1_64_to_128(T)              # [B,128,30,40], initialized zero
F_corrected = F + s * W * D
z_final = H(F_corrected)
loss = CE(z_final, labels)
```

The threshold learns from training CE; no validation-selected threshold is transferred. The temperature is fixed at 0.1. Lower confidence permits more correction; higher confidence suppresses correction smoothly. This is not hard routing and does not guarantee protection of every correct prediction. Confidence logits are detached, while the hierarchy, shared frame stage and classifier receive gradients through the final corrected output. The shared tail/head run twice with identical weights, and GroupNorm has no running statistics. BF16 autocast weight caching is disabled only for the no-grad confidence pass to preserve gradients in the second pass.

The residual projection starts at zero, so initial outputs equal the same-seed hierarchy-only model exactly. First-step branch gradients reach the residual projection; subsequent updates also reach the TS encoder, spatial gate and learned threshold. All parameters remain trainable. During end-to-end training, uncorrected logits are a confidence signal, not a separately supervised hierarchy-only model.

## Tensor contract (native 480×640)

| Stage | Encoder/operation | Output |
|---|---|---|
| Event points | packed normalized x,y,t,polarity | [sum_N,4] |
| Point encoder | Linear4→32, LayerNorm, SiLU, Linear32→32, LayerNorm, SiLU | [sum_N,32] |
| Learned voxel construction | FP32 weighted scatter mean + log1p event mass, 8 temporal bins, 4×4 cells | [B,33,8,120,160] |
| Voxel projection | 1×1×1 Conv33→32, norm, SiLU | [B,32,8,120,160] |
| Voxel stage 1 | existing temporal residual block, spatial stride2 | [B,64,8,60,80] |
| Voxel stage 2 | existing temporal residual block, spatial stride2 | [B,128,8,30,40] |
| Frame aggregation F | ordered temporal flatten to1024; Conv1×1 1024→128, norm, SiLU | [B,128,30,40] |
| TS input | same raw sample and interval; negative/positive recency, decay0.2 | [B,2,480,640] |
| TS encoder | four Conv3×3 stride2 + GN8 + SiLU blocks, 2→16→32→64→64 | [B,64,30,40] |
| Spatial gate W | concat192→Conv1×1→32, GN8, SiLU, Conv1×1→1, sigmoid | [B,1,30,40] |
| Confidence control s | detached shared-classifier probability margin; learned scalar threshold | [B,1,1,1] |
| Corrected features | F+sWD | [B,128,30,40] |
| Shared frame stage | residual blocks128→256 stride2;256→256 | [B,256,15,20] |
| Shared classification head | global average pooling, Linear256→100 | [B,100] |

## Compute and verification

2,745,830 parameters (V2 +1). 8.26437248 GMACs per sample at 116,790 events (V2 7.63520128). The extra 0.6291712 GMACs comes from the second shared tail/head evaluation. Counts include Conv2d/Conv3d/Linear only, excluding rendering, scatter, softmax, normalization, activations, pooling and elementwise operations. No conditional compute savings are claimed.

23 relevant tests passed: identity, monotonic/detached confidence, full-model gradients, CPU BF16 shared-tail regression, data contracts, strict reload and exact resume, plus unchanged V1/V2 hierarchy tests. The GPU preflight used 320 training samples: 64 warmup +256 measured, true batch32 without accumulation, BF16, physical GPU0. It passed with finite/nonzero later gradients for all stages, all stages updated, FP32 scatter, and bit-exact GPU checkpoint reload. Throughput: 128.562 samples/s; peak allocated11.897 GiB, reserved16.402 GiB. These are observed preflight values, not worst-case memory across the entire dataset.

An initial preflight caught an autocast-cache gradient issue; the failed attempt is preserved under gpu_benchmark/failed_initial_autocast_cache. The fix passed both CPU regression and fresh GPU preflight before main launch. No diagnostic checkpoint is reused.

## Exact training settings and command

N-ImageNet Mini: same 124,395 training and 5,000 internal validation samples. No test-set access. 100 epochs; seed20260908; fresh full-model initialization; batch32; accumulation1; BF16; SGD lr0.05, momentum0.9, weight_decay0.0001; cosine decay to0; no augmentation; workers4, CPUthreads4. Physical GPU0 maps to logical cuda:1. Existing physical GPU1 baseline remains untouched.

```bash
CUDA_VISIBLE_DEVICES=GPU-4fb73c12-68f6-7e96-db96-2ec175350543,GPU-25c7cbbd-0ce6-3d44-3397-a0dc30f9b62a PYTHONPATH=/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ts_confidence_v3_artifacts/main_snapshot_seed20260908 PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -u -m ebackbone_v3.hierarchy_ts_confidence train --config /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ts_confidence_v3_artifacts/main_snapshot_seed20260908/configs/hierarchy_ts_confidence_v3.n_imagenet_mini.json --raw-cache /mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts/raw_cache --output-dir /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ts_confidence_v3_artifacts/main_seed20260908
```

Source is frozen under main_snapshot_seed20260908. The supervisor verifies all source/config hashes and 164 existing protected files before launching. Checkpoints, atomic histories and final strict-reload report are written to main_seed20260908. W&B uses a separate artifact follower in the existing project, including first-batch threshold, strength and correction statistics; first-batch metrics are not epoch averages. Only previously authorized metrics, settings and hashes are uploaded.

## Files and next check

New repository files: ebackbone_v3/hierarchy_ts_confidence_models.py, hierarchy_ts_confidence_training.py, hierarchy_ts_confidence.py; configs/hierarchy_ts_confidence_v3.n_imagenet_mini.json; tests/test_hierarchy_ts_confidence.py; docs/HIERARCHY_TS_CONFIDENCE_V3.md. Existing experiment code/configurations/checkpoints are unchanged.

Next check: verify the first complete train/validation epoch and live W&B curves. Treat all intermediate accuracy as partial; compare final best V3 against completed hierarchy/V2 runs after the budget finishes. The validation-selected router result is exploratory motivation, not a target V3 has already achieved.
