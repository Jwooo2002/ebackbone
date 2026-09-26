# Polarity preservation: isolated voxel-aggregation ablation

Status: implementation and bounded verification PASS. No full training launched or queued; no GPU used. The existing local/control/skip queue continues unchanged. This is preparation for the next ablation, not a new accuracy result.

## Where polarities mix today

1. `hierarchy_data.prepare_points` supplies signed polarity -1/+1 alongside x,y,t. The unchanged 4→32→32 point MLP processes each event independently; it can encode polarity but does not mix separate events.
2. `hierarchy_models.PointToVoxel` scatters both signs into the same spatial/temporal cell, divides feature sums by combined event mass, and appends one log1p(total mass) channel. This is the first explicit mixing of positive and negative events. It is a mean of learned features, not necessarily cancellation of signed raw counts. Separate polarity means/counts are no longer explicitly available.
3. The 33→32 voxel projection and later spatial (1×3×3)/temporal (3×1×1) residual convolutions mix latent channels. Temporal collapse compresses eight ordered time bins into a frame-level feature map, followed by the spatial frame CNN. No explicit polarity axis remains.
4. Existing TS rendering already keeps negative/positive recency in two channels. Its first TS Conv2d2→16 learns to combine those channels. TS rendering, encoding and integration are unchanged by this ablation.

## Exact change

Keep the original shared point MLP and the original spatial cells, temporal interpolation weights, event order and complete raw sample. For each polarity s and voxel v separately:

```
m_s(v) = sum_i w_iv * 1[p_i=s]
mu_s(v) = sum_i w_iv * 1[p_i=s] * h_i / max(m_s(v), 1e-6)
G_s(v) = concat(mu_s(v)[32], log1p(m_s(v))[1])
G = concat(G_negative[33], G_positive[33])
V = SiLU(existing_GroupNorm(Conv3d_1x1x1_66_to_32(G)))
```

Empty sign/cell summaries are zero. A cell with one polarity does not divide its features by events of the opposite polarity. The first learned cross-polarity fusion is the widened 1×1×1 convolution. Its output shape remains exactly the original voxel-projection output. No labels are used in aggregation, routing or fusion; no events are dropped.

Only `point_to_voxel` and the input channel count/weights of `voxel_projection[0]` change. This projection replaces the existing 33→32 projection, rather than adding a second stem. The existing projection GroupNorm and SiLU, voxel blocks, temporal collapse, frame CNN, pooling and classifier remain unchanged.

The new factory constructs a fresh instance of the selected original model class and reuses its forward implementation verbatim. Supported bases are hierarchy-only, TS V1, TS V2 and TS V3; compatibility is tested without changing their TS path. The primary prepared ablation is hierarchy-only. It is not combined with local interaction or early skip.

## Tensor shapes (480×640 input)

| Stage | Shape |
|---|---|
| Packed events | [M,4] |
| Unchanged point MLP | [M,32] |
| Separate polarity summaries, conceptual | [B,2,33,8,120,160] |
| Concatenated negative/positive grids | [B,66,8,120,160] |
| Learned fusion / voxel projection | [B,32,8,120,160] |
| Existing voxel stage1 | [B,64,8,60,80] |
| Existing voxel stage2 | [B,128,8,30,40] |
| Existing ordered temporal collapse | [B,128,30,40] |
| Existing frame CNN | [B,256,15,20] |
| Pool and classifier | [B,100] |

The fusion is pointwise in the voxel lattice: it does not pool across spatial locations or temporal bins. This preserves the input to the existing spatial/temporal processing contract. Explicit polarity separation ends at learned fusion; the model does not maintain two independent polarity hierarchies afterward. It still loses individual-event ordering within aggregation cells, and it does not add local event interaction.

## Parameters, compute and initialization

Hierarchy-only total: **2,671,780 parameters**, **7.53685888 GMACs** per sample at116,790 events. Relative to the original: **+1,056 parameters**, **+0.1622016 GMACs**. The extra MACs are 33×32×8×120×160. The point MLP cost is unchanged. Conv/Linear counts exclude scatter, normalization, activations and memory traffic.

The unfused voxel tensor grows from33 to66 FP32 channels: an additional20,275,200 bytes (19.34MiB) per sample for that tensor alone. Peak training-memory and runtime impact have not been benchmarked; no GPU memory estimate is claimed.

Point, downstream and selected TS initial tensors are identical to their originals at the same seed. The global CPU RNG state is preserved. The widened fusion convolution is initialized normally inside a forked RNG context; its weights and input statistics necessarily differ from the old projection. Initial logits are NOT claimed to equal baseline logits. An unmodified baseline checkpoint must fail strict loading on the widened projection shape; the bounded new checkpoint reload is bit exact.

## Bounded verification

35 tests passed in6.77s across polarity, hierarchy-only and existing TS tests. Coverage includes opposite-polarity collisions, independent weighted means/counts, fractional temporal interpolation, count conservation, terminal bin, empty polarity banks, polarity swaps, negative-only gradient isolation, packed sample isolation, CPU BF16 CE gradients to both fusion halves, same-seed point/downstream/TS preservation, strict reload, and full native-resolution meta shapes/MACs for all four supported bases.

One real TRAIN sample contains116,790 events:61,246 negative and55,544 positive. Interpolated mass sums reproduce both counts exactly. Two CPU SGD steps using only final classification CE gave finite gradients to the point MLP, both polarity fusion halves and every downstream stage. Strict reload reproduced logits exactly. This tiny diagnostic has no scientific accuracy interpretation.

No validation/test samples were loaded, no GPU was used, and no full-training process was launched. All255 recorded existing source/config/checkpoint/queue files match their pre-task hashes. The running queue definition is unchanged.

## Training protocol and commands

No existing configuration or training loop was edited. A future full run retains the established100 epochs, seed20260908, true batch32, BF16, SGD lr0.05/momentum0.9/weight_decay0.0001, cosine schedule, same train/validation split, no augmentation and one final CE. For any selected TS base, use its existing TS path without redesign. A full-run launcher has intentionally not been added to this bounded-verification CLI.

From `/mnt/ssd1/PycharmProjects/ebackbone_V3`:

```bash
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m pytest -q tests/test_hierarchy_polarity.py tests/test_hierarchy.py tests/test_hierarchy_ts.py tests/test_hierarchy_ts_residual.py tests/test_hierarchy_ts_confidence.py
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m ebackbone_v3.hierarchy_polarity profile --base hierarchy --events 116790
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m ebackbone_v3.hierarchy_polarity verify --base hierarchy --raw-cache /mnt/ssd1/PycharmProjects/ebackbone_v3_v1_artifacts/raw_cache --output-dir /mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_polarity_artifacts/diagnostic_cpu
```

The diagnostic directory already exists; reruns require a new output path. `verify` always performs exactly two CPU steps on the first training sample. It has no `train` subcommand. Full machine-readable evidence is in diagnostic_cpu/report.json and verification.json.

## New files and next step

New files: ebackbone_v3/hierarchy_polarity_models.py (aggregation and factory), ebackbone_v3/hierarchy_polarity.py (profile/bounded verification only), tests/test_hierarchy_polarity.py, and docs/HIERARCHY_POLARITY.md. Existing source files and experiments remain unchanged.

After the current experiment queue finishes, the next bounded step is a GPU BF16 batch32 memory/throughput preflight before considering a separately authorized full run. Accuracy improvement remains untested.
