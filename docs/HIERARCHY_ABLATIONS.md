# Independent hierarchy ablations: local events, self control, early skip

Status: preflights and launch verification PASS. Local interaction running on physical GPU0, PID305585; self control and early skip queued. W&B uploader is prepared but blocked by automatic approval review pending authorization for these three new runs. This suite queues three fresh 100-epoch runs on physical GPU0. GPU1's existing baseline is untouched. No combined local+skip variant is included.

## Questions and controls

| Variant | Single change from hierarchy-only | Primary comparison |
|---|---|---|
| hierarchy_local | Residual local-event interaction before original voxelization | hierarchy-only and hierarchy_self_control |
| hierarchy_self_control | Same-size residual per-event network, no neighbor information | hierarchy_local |
| hierarchy_early_skip | First-voxel-stage features skip to frame aggregation output | hierarchy-only |

Local vs self control matches trainable parameter counts and executed Linear/Conv MACs, not sorting/gather overhead or full runtime. Its purpose is to distinguish neighborhood use from adding a per-event network. It is not a claim that optimization or effective function classes are identical. Early skip has its own cost; it is neither added to local interaction nor trained with TS. All use one final CE, no auxiliary supervision, no raw-frame stem, no top-down module, no event subsampling or test-set access. Existing TS and baseline runs remain unchanged.

## Exact local event block

The original point MLP 4→32→32 is unchanged and produces h_i [M,32]. A new Linear32→8 produces q_i. Bucket IDs are the existing lower voxel indexes, including packed sample offsets: original 4×4 spatial cell, floor(7*t_normalized). Consequently neighborhoods never cross sample boundaries. Events are sorted stably by normalized timestamp within bucket; equal timestamps retain input order. Each event can use up to two preceding and two following events in that bucket. This is a bounded local neighborhood, not a global k-nearest search; it cannot cross bucket boundaries.

Each neighbor message receives concat(q_i[8],q_j[8],relative[x,y,t,polarity][4]) [M,20], then Linear20→16, LayerNorm16, SiLU. Relative offsets are scaled to cell/bin/polarity units: dx*(W-1)/4, dy*(H-1)/4, dt*7, dp/2. Invalid neighbors are masked; messages are averaged over available neighbors. Linear16→32 produces a residual added to h_i. Both projection weights and bias start at zero. Isolated events get zero correction. The corrected [M,32] features go to the original weighted voxel aggregation; every original event remains, in its original order and with its original interpolation weights. There are no labels in neighborhood construction.

The self control uses identical trainable modules and four executed message evaluations with concat(q_i,q_i,x_i,y_i,t_i,p_i), rather than neighbor features/relative offsets. Its messages use only the event itself. The duplicated evaluations intentionally match neural MACs; this is an experimental compute control, not an optimized deployment proposal.

The two added blocks have 1,176 trainable parameters each, and 2,048 Linear MACs per event in addition to the original point MLP. The initial output exactly matches same-seed hierarchy-only. Initially the residual projection receives data gradients; the reducer/message network receives gradients after its first update.

## Exact early skip

The original first voxel stage E [B,64,8,60,80] feeds both the unchanged main second stage and the new skip:

```
E → fixed-order 2×2 spatial mean → [B,64,8,30,40]
  → ordered temporal flatten → [B,512,30,40]
  → Conv1×1 512→128, bias, initialized zero → S [B,128,30,40]
F = original temporal collapse after second voxel stage [B,128,30,40]
F + S → original frame CNN → GAP → Linear256→100 → CE
```

The fixed-order mean has deterministic CUDA backward and preserves the eight time bins. It was verified against average pooling in both forward/backward. The skip bypasses the second voxel encoder/collapse; it does not recover information discarded during point pooling. Zero initialization makes initial logits match hierarchy-only exactly.

## Unchanged downstream tensor contract

Packed points [M,4] → point features [M,32] → weighted voxel mean+log1p mass [B,33,8,120,160] → projection [B,32,8,120,160] → voxel stage1 [B,64,8,60,80] → stage2 [B,128,8,30,40] → ordered temporal collapse [B,128,30,40] → frame CNN [B,256,15,20] → GAP [B,256] → logits [B,100]. All input representations come from the same full raw-event sample and stored interval.

## Params, GMACs and GPU preflight

GMACs below are per sample at 116,790 events, matching the baseline reference. The preflight's hash-selected reference contains 90,370 events; its profile stores that actual event count and exact per-event/fixed cost, used to normalize the table. Conv/Linear MACs exclude sorting, scatter, gather, normalization, activation, pooling and I/O.

| Variant | Parameters | GMACs at116790 | Samples/s | Allocated GiB | Reserved GiB |
|---|---:|---:|---:|---:|---:|
| hierarchy-only reference | 2,670,724 | 7.37465728 | — | — | — |
| hierarchy_local | 2,671,900 | 7.61384320 | 93.34 | 15.13 | 21.41 |
| hierarchy_self_control | 2,671,900 | 7.61384320 | 98.46 | 14.47 | 20.09 |
| hierarchy_early_skip | 2,736,388 | 7.45330048 | 126.02 | 12.24 | 16.91 |

Each variant used the exact same ordered 320 training samples, true batch32 with no accumulation, 64 warmup +256 measured. All passed finite/nonzero gradients by stage, expected zero-initialization gradient behavior, parameter updates, initial BF16 baseline-logit identity, and strict bit-exact GPU checkpoint reload. Observed memory is not a worst-case guarantee for all event counts. 17 CPU tests passed, including BF16 gradients, sample isolation, neighbor sensitivity vs self independence, zero-neighbor behavior, parameter/MAC equality, and exact training resume for all variants.

The initial skip preflight exposed non-deterministic CUDA avg_pool3d backward. No long run was started with that implementation. It was replaced by an equivalent fixed-order spatial mean, tested, and all three GPU preflights rerun under the final frozen source. Earlier preflight artifacts/logs remain for provenance.

## Training protocol and exact commands

Same 124,395 training +5,000 internal-validation samples; seed20260908; 100 epochs each from scratch; batch32, accumulation1; BF16; SGD lr0.05, momentum0.9, weight_decay0.0001; cosine lr to0; no augmentation; workers4, CPUthreads4; physical GPU0 exposed as logical cuda:1. Train and validation loader order is independent of model initialization/branch count. One shared immutable source snapshot contains all three definitions; each run selects exactly one model and writes to its own directory. No preflight or existing experiment checkpoint initializes a main run.

Queue order: local interaction, self control, early skip. Each starts only after the previous process exits and GPU0 is free. Failures are recorded separately; remaining independent variants proceed without changing settings. No existing process is stopped or restarted.

### hierarchy_local

```bash
CUDA_VISIBLE_DEVICES=GPU-4fb73c12-68f6-7e96-db96-2ec175350543,GPU-25c7cbbd-0ce6-3d44-3397-a0dc30f9b62a PYTHONPATH=/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ablation_artifacts/snapshot_seed20260908 PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -u -m ebackbone_v3.hierarchy_ablation train --model hierarchy_local --config /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ablation_artifacts/snapshot_seed20260908/configs/hierarchy_local.n_imagenet_mini.json --raw-cache /mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts/raw_cache --output-dir /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ablation_artifacts/hierarchy_local/main_seed20260908
```

### hierarchy_self_control

```bash
CUDA_VISIBLE_DEVICES=GPU-4fb73c12-68f6-7e96-db96-2ec175350543,GPU-25c7cbbd-0ce6-3d44-3397-a0dc30f9b62a PYTHONPATH=/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ablation_artifacts/snapshot_seed20260908 PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -u -m ebackbone_v3.hierarchy_ablation train --model hierarchy_self_control --config /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ablation_artifacts/snapshot_seed20260908/configs/hierarchy_self_control.n_imagenet_mini.json --raw-cache /mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts/raw_cache --output-dir /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ablation_artifacts/hierarchy_self_control/main_seed20260908
```

### hierarchy_early_skip

```bash
CUDA_VISIBLE_DEVICES=GPU-4fb73c12-68f6-7e96-db96-2ec175350543,GPU-25c7cbbd-0ce6-3d44-3397-a0dc30f9b62a PYTHONPATH=/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ablation_artifacts/snapshot_seed20260908 PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -u -m ebackbone_v3.hierarchy_ablation train --model hierarchy_early_skip --config /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ablation_artifacts/snapshot_seed20260908/configs/hierarchy_early_skip.n_imagenet_mini.json --raw-cache /mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts/raw_cache --output-dir /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_ablation_artifacts/hierarchy_early_skip/main_seed20260908
```

W&B follows the three independent run directories in the existing project using the previously authorized metrics/settings/hash scope. A queued run appears in W&B when its trainer creates the run configuration; queue_status.json records all states locally from launch. The existing W&B runs are not modified. End-of-run comparison exports summary.csv and learning_curves.png under comparison/; queued or partial runs are explicitly labeled. No early stopping or combining variants is performed based on interim validation.

## Interpretation

Compare each variant with the completed 80.00% hierarchy baseline. If local improves over both hierarchy-only and the self control, that supports the value of this specific neighborhood interaction. If local and self improve similarly, added per-event capacity is a plausible explanation. If skip improves, first-stage features help bypass later transformation/compression, without identifying which exact clue is responsible. These are single-seed validation ablations, not proof of general superiority. Combine mechanisms only in a later separate experiment if these individual results justify it.

## Changed files

New modules: hierarchy_ablation_models.py, hierarchy_ablation_training.py, hierarchy_ablation.py. New configs: hierarchy_local.n_imagenet_mini.json, hierarchy_self_control.n_imagenet_mini.json, hierarchy_early_skip.n_imagenet_mini.json. New tests: tests/test_hierarchy_ablations.py. Documentation: docs/HIERARCHY_ABLATIONS.md. Snapshot, preflights, queue supervisor, commands, W&B sidecar and run outputs are isolated under this artifact directory. 206 existing source/config/checkpoint files were hash-verified unchanged before launch.
