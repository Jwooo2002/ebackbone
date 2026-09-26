# TS research handoff

Snapshot checked on **2026-09-11 at 10:35 KST**. Training continues, so refresh the local histories before quoting current results.

## Research objective and user decisions

Build a reusable event-vision backbone from different representations of the **same raw event stream**. The main hierarchy is **point -> voxel -> frame-level features**, fine to coarse. A separate raw-frame stem is deliberately excluded. Time surface (TS) is an optional complementary recency prior outside the hierarchy.

The user prioritizes improved classification accuracy and explicitly chose to **keep the hierarchy and redesign TS integration** after the first TS experiment underperformed. Discuss TS integration here; do not assume the user wants a new backbone, CLIP/text branch, top-down refinement, SSL, or auxiliary losses. None of those are implemented in these runs.

The user explicitly stopped multiplicative TS V1 and subsequently authorized and launched residual TS V2. V2 is currently running; keep its implementation/settings/snapshots and the parallel baselines intact while discussing alternatives.

## Workspace and first reads

Repository: `/mnt/ssd1/PycharmProjects/ebackbone_V3`.

1. Read `/mnt/ssd1/PycharmProjects/ebackbone_V3/AGENTS.md` and inspect git status. There are pre-existing uncommitted files; do not reset them.
2. Read `/mnt/ssd1/PycharmProjects/ebackbone_V3/docs/HIERARCHY_TS_RESIDUAL_V2.md` for the current TS design, exact command, tensor shapes and preflight.
3. Read `/mnt/ssd1/PycharmProjects/ebackbone_V3/docs/HIERARCHY_TS_V1.md` for the stopped first TS design.
4. Read `/mnt/ssd1/PycharmProjects/ebackbone_V3/docs/HIERARCHY_V1.md` for the hierarchy-only reference.
5. Inspect the corresponding model code and real training artifacts below. Historical discussion and this handoff are context; frozen source and run artifacts establish what actually ran.

## Backbone and evaluation

All events are packed as `[sum_N,4]`: normalized x, y, timestamp, signed polarity. Point encoder: MLP `4->32->32`, LayerNorm and SiLU. Spatial routing uses 4x4 native-pixel cells; temporal routing linearly interpolates across eight bins. Weighted means of learned features plus log1p event mass produce `[B,33,8,120,160]`.

A 1x1x1 projection gives 32 channels. Factorized spatial/temporal residual stages produce `[B,64,8,60,80]`, then `[B,128,8,30,40]`. Ordered channel/time flatten gives `[B,1024,30,40]`; a learned 1x1 projection with normalization/activation gives **F `[B,128,30,40]`**. This is where both TS variants connect.

The frame CNN has residual blocks 128->256 (stride2), then 256->256, producing `[B,256,15,20]`. Global average pooling gives `[B,256]`; `Linear(256,100)` gives classification logits.

Dataset: **N-ImageNet Mini**, 100 classes, 124,395 training samples and 5,000 internal validation samples. The backbone and classification head train jointly from scratch with one final cross-entropy loss. Validation runs after every epoch without gradients. Report top-1/top-5 and CE; select best by validation top-1, then lower CE. This is supervised classification, **not** frozen-backbone linear probing, transfer evaluation, SSL or final-test evaluation.

Settings shared by hierarchy-only, TS V1 and TS V2: 100-epoch configured budget, true batch32, accumulation1, BF16 autocast, SGD lr0.05, momentum0.9, weight_decay1e-4, cosine decay to0, seed20260908, workers4, CPUthreads4, no augmentation. First hierarchy experiment benchmarked true batches8/16/32 and selected32; subsequent TS versions retain this setting. Do not confuse this with the separate parallel baselines' original batch8/accumulation4 protocol.

## TS representation: identical for V1 and V2

Read `/mnt/ssd1/PycharmProjects/ebackbone_V3/ebackbone_v3/hierarchy_ts_data.py`.

Decode one raw sample once; use exactly the same events and observed interval for points and TS. At each pixel/polarity, retain the most recent timestamp. For occupied pixels:

`S = exp(-(1 - latest_normalized_timestamp) / 0.2)`.

Background is0; zero-duration occupied pixels are1. Two channels, negative/positive polarity, `[B,2,480,640]`, float32. No event filtering, sampling, label-conditioned rendering, or extra raw-frame rendering. This is an explicit recency prior, not additional sensor information or recovery of the complete event ordering.

Both TS encoders use four bias-free 3x3 stride2 padding1 Conv/GN8/SiLU blocks: `2->16->32->64->64`. Outputs: `[B,16,240,320]`, `[B,32,120,160]`, `[B,64,60,80]`, **T `[B,64,30,40]`**.

## Implemented TS variants

**TS V1: multiplicative modulation.** `A = Conv1x1(64,128)(T)`; `F_out = F * (1 + 0.5*tanh(A))`. Each channel/location gets a multiplier in[0.5,1.5], conditioned only on TS. Projection weight/bias initialize to zero, giving exact identity. Read `/mnt/ssd1/PycharmProjects/ebackbone_V3/ebackbone_v3/hierarchy_ts_models.py`.

**TS V2: content-aware residual fusion.** `D = Conv1x1(64,128)(T)` is a signed correction. `W = sigmoid(q(concat(F,T)))`, where q is bias-free Conv1x1(192,32), GN8, SiLU, biased Conv1x1(32,1). W has shape `[B,1,30,40]` and is shared across the128 correction channels. `F_out = F + W*D`. The fusion weight sees both hierarchy and TS features; it is not calibrated uncertainty. D's weight/bias start at zero; W is initialized normally. Original backbone and TS encoder initial weights/RNG are preserved. TS encoder and weight network receive zero data gradients on the first backward and nonzero gradients after the correction projection updates. Read `/mnt/ssd1/PycharmProjects/ebackbone_V3/ebackbone_v3/hierarchy_ts_residual_models.py`.

V2 changes both additive fusion and gate conditioning; any gain would not isolate which of those two changes caused it. It is a hypothesis, not a demonstrated improvement yet.

Model parameters / GMAC at116,790 events:

| Model | Parameters | GMAC |
|---|---:|---:|
| Hierarchy-only | 2,670,724 | 7.37465728 |
| TS V1 | 2,739,588 | 7.62779008 |
| TS V2 | 2,745,829 | 7.63520128 |

MACs count convolution/linear layers only, excluding rendering, scatter, normalization, activations, pooling and elementwise fusion. V2 preflight:172 tests passed; three BF16 batch32 repetitions; median128.57 samples/s; peak allocated11.90GiB, reserved16.10GiB; strict GPU reload bit-exact. Mixed precision retains FP32 scatter/normalization/residual outputs and BF16 classifier logits.

## Current evidence, not final V2 results

| Model | State at handoff | Best validation top-1 | Latest validation top-1 |
|---|---|---:|---:|
| Hierarchy-only | Completed100 epochs | 80.00%, epoch75 | 79.44%, epoch100 |
| TS V1 | User-stopped after55 epochs; epoch56 interrupted | 69.68%, epoch49 | 68.14%, epoch55 |
| TS V2 |31 epochs completed; epoch32 running |70.36%, epoch29 |69.46%, epoch31 |

At matched epoch31: hierarchy-only71.54%, TS V1 67.18%, TS V2 69.46%. V2 improves over the first TS design at that point but remains below hierarchy-only. Comparing incomplete V2 directly with hierarchy's final80% cannot establish its eventual outcome.

At epoch55, hierarchy-only and TS V1 both had about93.05% training accuracy, while validation differed. V1's TS gate was active, not disconnected: first-validation-batch feature change was about28%, with about1% of multipliers near bounds. Those are sampled gate diagnostics, not an identified cause of underperformance.

## Exact artifacts to inspect

The three artifact roots are:

- `/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_v1_artifacts`
- `/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ts_v1_artifacts`
- `/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ts_residual_v2_artifacts`

Under each root:

- `main_launch_status.json`: supervisor state; stored PIDs alone are not proof of liveness.
- `main_seed20260908/history.json`: complete per-epoch metrics; refreshed atomically.
- `main_seed20260908.log`: current optimizer progress between epochs.
- `main_seed20260908/config.json`: architecture, settings, manifests and source hashes.
- `main_seed20260908/checkpoint_best.pt` and `checkpoint_last.pt`: saved training states.
- `main_seed20260908/report.json`: verified completed/partial report when generated. TS V1 instead has `stopped_report.json`; do not label it a completed100-epoch run.
- `main_snapshot_seed20260908/ebackbone_v3/`: exact frozen source used by that experiment.

V2 also has `RUN.md`, `verification.json`, `launch_verification.json`, `gpu_benchmark/batch32*/report.json` and `comparison/REPORT.md`, `comparison/learning_curves.csv`, `comparison/learning_curves.png`.

Training/evaluation implementations: `/mnt/ssd1/PycharmProjects/ebackbone_V3/ebackbone_v3/hierarchy_ts_residual_training.py` and `/mnt/ssd1/PycharmProjects/ebackbone_V3/ebackbone_v3/hierarchy_ts_residual.py`. Config: `/mnt/ssd1/PycharmProjects/ebackbone_V3/configs/hierarchy_ts_residual_v2.n_imagenet_mini.json`. Tests: `/mnt/ssd1/PycharmProjects/ebackbone_V3/tests/test_hierarchy_ts_residual.py`.

Parallel-baseline artifacts are separate at `/mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts`; these jobs must remain unchanged. V2 uses physical GPU0, exposed as logical cuda:1 using UUID visibility. At handoff its trainer PID was3710484; verify live state before using any PID. GPU1 serves the separate parallel-baseline experiment.

## W&B

Project: https://wandb.ai/jwoh3923-/ebackbone-v3-n-imagenet-mini

- Hierarchy without TS: https://wandb.ai/jwoh3923-/ebackbone-v3-n-imagenet-mini/runs/v1b7f999604423ca
- Stopped multiplicative TS V1: https://wandb.ai/jwoh3923-/ebackbone-v3-n-imagenet-mini/runs/v1afd46bfcc1e1aa
- Running residual TS V2: https://wandb.ai/jwoh3923-/ebackbone-v3-n-imagenet-mini/runs/v14b321cc787f0b9

Compare `validation/top1_pct` against `epoch`. Uploads occur through separate CPU artifact-following processes every30 seconds, not directly from trainers. User explicitly authorized aggregate metrics/settings/provenance hashes to this project. No raw events, labels, source files, credentials, or checkpoint uploads are part of the uploader scope.

V2 uploader state: `/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ts_residual_v2_artifacts/wandb_sync/state/status.json`; registry and launch metadata are in that `wandb_sync` directory. Other runs have a separate uploader under `/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_v1_artifacts/wandb_sync`. Do not start duplicate uploaders. TS V1's W&B native run state is finished, but its custom `training/status` is `stopped_early`, not completed training.

## Suggested starting point for the new chat

Refresh V2 and matched-epoch reference metrics from the history files. Read both TS fusion implementations. Discuss what TS should contribute beyond the existing point/voxel temporal encoding, and distinguish observed failure patterns from hypotheses. Preserve the main hierarchy and keep current runs intact while assessing the next TS experiment. Future accuracy claims require real validation results; a functioning gate or decreasing training loss is not sufficient.

## Scheduled post-training diagnostic (added 2026-09-11)

The user requested a fixed-checkpoint correction-strength sweep **after V2 completes all100 epochs**: alpha in {0,0.25,0.5,0.75,1,1.25}, F_out=F+alpha*Delta, with Delta equal to the full gated TS correction. A queue is already running; do not launch a duplicate. It selects checkpoint_best.pt from the completed run, evaluates all5,000 validation samples without retraining, and reports top1 plus correct->wrong and wrong->correct counts versus alpha1. Alpha1 must reproduce recorded accuracy.

Inspect `/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ts_residual_v2_artifacts/alpha_sweep_best_after100/status.json` and `request.json`. Results will be in its `results/REPORT.md`, `alpha_summary.csv`, `report.json`, and a separate W&B validation-alpha-sweep run. Per-sample predictions remain local. Six evaluator tests passed; model source and training snapshots are unchanged.
