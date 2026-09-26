# Hierarchy + TS gating V1

Status: stopped early at user request after 55 completed epochs; epoch 56 was interrupted. Best checkpoint: epoch 49, validation top-1 69.68%. Both best and latest checkpoints are preserved and strict-load verified. GPU 0 released. See main_seed20260908/stopped_report.json.

## Architecture

Same raw sample: packed points [sum_N,4] -> MLP 4/32/32 -> weighted point-to-voxel [B,33,8,120,160] -> 1x1x1 projection [B,32,8,120,160] -> factorized temporal residual stages [B,64,8,60,80] and [B,128,8,30,40] -> ordered flatten [B,1024,30,40] -> 1x1 temporal collapse [B,128,30,40].

TS: latest normalized timestamp per pixel and polarity, exp(-(1-last)/0.2), background zero, zero-duration occupied pixels one. Same raw events, interval and identity; no labels in rendering. Input [B,2,480,640]. Four bias-free 3x3 stride-2 padding-1 Conv/GN8/SiLU blocks: 2->16->32->64->64. Shapes [B,16,240,320], [B,32,120,160], [B,64,60,80], [B,64,30,40]. Biased 1x1 Conv 64->128 gives A [B,128,30,40].

F_gated = F * (1 + 0.5*tanh(A)). Final projection weights and bias initialize to zero. Original backbone weights and RNG state are preserved. Encoder first data gradient is zero; second backward receives a nonzero gradient. Multiplier bounded 0.5 to 1.5.

Existing frame residual CNN -> [B,256,15,20] -> mean pooling [B,256] -> Linear256/100. Single final cross-entropy, all stages trainable. No raw-frame stem, top-down refinement, auxiliary loss, SSL or text encoder.

## Verified cost and tests

Parameters: 2,739,588 (68,864 added). At 116,790 events: 7.62779008 GMAC (0.2531328 added). Conv/linear MACs only; excludes TS rendering, scatter, normalization, activations, pooling and elementwise gating. Tests: 168 passed. Native real-data CPU diagnostic and BF16 GPU strict reload were bit exact. Three repeat true-batch-32 benchmarks: [127.58213757337286, 127.71730776372847, 128.66925195188958] samples/s; median 127.717. Peak allocated 11.896 GiB; reserved 16.104 GiB. Each repetition: 64 warmup + 256 measured real train samples, one batch per optimizer update. This is observed memory, not an exhaustive maximum-event-count bound.

## Training

Fresh initialization, 100 epochs, true batch32, accumulation1, BF16, SGD lr0.05 momentum0.9 weight_decay0.0001, cosine decay to0, seed20260908, workers4, CPUthreads4. Settings exactly equal hierarchy V1; same manifests and sample order. Physical GPU0 maps to logical cuda:1 through UUID visibility. Existing baseline on physical GPU1 remains unchanged. No benchmark checkpoint is reused.

```bash
CUDA_VISIBLE_DEVICES=GPU-4fb73c12-68f6-7e96-db96-2ec175350543,GPU-25c7cbbd-0ce6-3d44-3397-a0dc30f9b62a PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ts_v1_artifacts/main_snapshot_seed20260908 /home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -u -m ebackbone_v3.hierarchy_ts train --config /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ts_v1_artifacts/main_snapshot_seed20260908/configs/hierarchy_ts_v1.n_imagenet_mini.json --raw-cache /mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts/raw_cache --output-dir /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ts_v1_artifacts/main_seed20260908
```

## Artifacts

Training log: main_seed20260908.log. Epoch metrics/gate diagnostics: main_seed20260908/history.json. Best/last checkpoints and verified final report are written in that directory. Local comparison CSV/PNG/PDF and REPORT.md refresh after each epoch. Final reporting requires completed verified runs; incomplete baselines remain labeled partial. No final-test access.

Source is frozen under main_snapshot_seed20260908. Preexisting repository files were hash-checked unchanged; both original experiment snapshots are verified again by the launcher. W&B upload remains separate from this local training launcher.
