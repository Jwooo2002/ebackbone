# Independent dual-fusion implementation review

Reviewed 2026-09-24 02:49 KST by an independent implementation reviewer. Status:
**PASS for the reviewed model/data/runner contracts and bounded CPU tests**.
No actionable correctness defect remains in this review. Full training was not
launched. The separate native GPU report was also independently inspected; it
confirms a bounded batch-32/GPU step, subject to the limits below.

## Prior coverage

The existing V1 `same_arch` and `heterogeneous` backbones already encode event
frame, voxel grid and time surface from one full raw event array. They concatenate
spatial feature maps, apply a 1x1 convolution and use a shared spatial CNN tail.
This covers multi-representation encoding and concatenation, but not the new
pooled-latent concat/MLP or hierarchy/latent dual-branch mixture.

TS V1 multiplies post-collapse hierarchy features by a TS-derived gate. TS V2
adds a spatially gated TS residual. TS V3 adds a confidence-dependent scale to
that residual. None implements two normalized branch vectors combined by one
learned global sigmoid scalar.

Reports were read directly, separately from launch-era documentation:

- The older V1 heterogeneous report is partial at 67 epochs, with best internal
  validation 68.80%; `same_arch` completed 100 epochs, best 67.12%.
- The original hierarchy report completed 100 epochs, best 80.00%.
- The separate batch-64/cosine-50 study completed hierarchy, TS V1 and TS V2 at
  best 75.06%, 76.00% and 73.52%, respectively. Its queue is paused before TS V3.

These protocols are not interchangeable. All inspected reports explicitly state
`final_test_accessed=false`. The new hierarchy-only control adds projection and
LayerNorm before its classifier, so it is a new matched control rather than an
exact repeat of the historical plain linear-head hierarchy.

## Reviewed contracts

- `prepare_inputs` reuses unchanged point preparation and V1 rendering on one
  decoded event array and source identity. No temporal window splitting,
  filtering, augmentation or representation cache is introduced. Eight voxel
  bins interpolate the same complete interval.
- Latent branches use widths 8/16/32, independent frame/voxel/surface encoders,
  spatial pooling, concatenation to 96 channels and MLP 96->256->256. The voxel
  encoder retains ordered eight-bin collapse before spatial pooling.
- Each active branch has Linear256->256 followed by affine LayerNorm. This is
  LayerNorm, not unit-L2 normalization. Learned affine scales mean sigmoid(a)
  should not be interpreted as a calibrated contribution or confidence score.
- Dual output is exactly `(1-sigmoid(a))*z_h + sigmoid(a)*z_l`, with scalar
  `a=0` initially. There is one learned Linear256->100 classifier and CE loss.
- Independent seeded construction preserves identical shared initial weights
  across all three modes and does not advance caller CPU RNG. The original
  hierarchy feature weights match the original same-seed hierarchy.
- Inactive branches and the original hierarchy classifier are absent. The
  scalar exists only in dual mode. Every active parameter receives finite
  gradients; each parameterized component and scalar updates.
- Global batches are sharded without duplicate or missing samples. DDP sums
  local CE and scales by `world_size / actual_global_count`, including uneven
  tails. Validation uses the unwrapped model to avoid uneven-forward deadlocks.
- Production config fixes two ranks, 32/rank, global 64, accumulation 1, 50
  epochs, SGD .05/.9/1e-4, BF16 and identical manifests/seed across modes.
- New output directories are exclusive. Resume requires identical config,
  manifest and source provenance and restores optimizer, scheduler and per-rank
  RNG. No old queue or run path is used by the new runner.
- Checkpoints strictly reload the actual trained logits. Classifier-free
  exports reconstruct from validated metadata and reproduce embeddings exactly.
  Export metadata includes architecture and renderer identity.
- The native verification command performs one disposable real-data optimizer
  step per selected mode. It checks equal two-rank gradients, stage updates,
  live checkpoint reload and classifier-free export; it never initializes a
  production study from probe weights.

## Independent verification

Executed from `/mnt/ssd1/PycharmProjects/ebackbone_V3`:

```bash
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m pytest -q \
  tests/test_dual_fusion_models.py tests/test_dual_fusion_train.py
```

Sixteen tests passed. The remaining distributed test failed before model code
because the sandbox could not resolve/bind localhost for Gloo. The exact test
was rerun with host local-socket access:

```bash
GLOO_SOCKET_IFNAME=lo /home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m pytest -q \
  tests/test_dual_fusion_train.py::test_two_rank_gloo_gradients_tail_resume_and_export
```

**1 passed in 6.46 seconds**. Thus all 17 focused tests passed independently.
This includes all three modes, an independent full-batch reference for unequal
2/1 rank batches, exact two-epoch resume, fusion endpoints, label-independent
rendering, full-event mass preservation, and malformed export rejection.

Read-only CLI inspection also passed:

```bash
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m ebackbone_v3.dual_fusion_train inspect
```

| Mode | Parameters | GMAC/sample at 116,790 events |
| --- | ---: | ---: |
| hierarchy_only | 2,737,028 | 7.374722816 |
| latent_only | 300,940 | 6.066843648 |
| dual | 3,012,269 | 13.441540864 |

MACs include Conv2d/Conv3d/Linear only. They exclude rendering, scatter,
normalization, activation, pooling, elementwise fusion and I/O. These are
architecture comparisons, not parameter- or compute-matched models.

## Native two-GPU evidence inspected independently

`reports/dual_fusion_verification_20260923/native_b32/report.json` records PASS
for all three modes, each with one actual optimizer step, 32 samples per GPU,
global batch 64 and accumulation 1. The reviewer checked that both ranks have
identical gradient digests, positive gradients and updates for all recorded
components, and bit-exact live-to-checkpoint logits and exported embeddings.
The same 64 distinct real training samples were used across modes. Observed
sample event counts span 13,859 through 150,332. The dual scalar changes from
0.5 to 0.49997246265411377 after the disposable step.

Maximum observed dual memory across ranks was 40,751,524,864 bytes allocated
(37.95 GiB) and 43,578,818,560 bytes reserved (40.59 GiB). This is one batch's
feasibility evidence, not a worst-case bound for the full dataset or a throughput
benchmark. The artifact explicitly records no full-training launch and no
final-test access. The reviewer inspected the report and verification code;
the main task executed this GPU probe.

## Reviewed source fingerprints

| File | SHA-256 |
| --- | --- |
| `ebackbone_v3/dual_fusion_models.py` | `cc09b24d41b2c3127675cbff5237da0dc42315f5467e091097a0897ba549734b` |
| `ebackbone_v3/dual_fusion_data.py` | `995aaf1743943ce601d8cb00809d1fcc1fe21c4d6915f95ccd0b5ffe8a4d032c` |
| `ebackbone_v3/dual_fusion_train.py` | `c4ebb34ed622bbf31bcbbaa59c493c79c0e143ff7794aa42c78a3ab61cff2860` |
| `ebackbone_v3/dual_fusion_verify.py` | `3a2b8292bd4b1a9e7c3570c202fc6652bf3f3a2573c87f132c766d29f76e2c04` |
| `tests/test_dual_fusion_models.py` | `bdc3864a641cb2b853f92d73a1905abd46e6b0506708e6c61ba5e3a71b55c6b6` |
| `tests/test_dual_fusion_train.py` | `41a9efa6193de5dc3c9fdee0594d9881711ea70f2f32510f74e6039b5449a042` |
| `configs/dual_fusion.n_imagenet_mini_b64_e50.json` | `bd0290d3ecdd0ad0e513c2da68acca26ec67e0fab0c396b6dc90f399e610259b` |

The reviewer changed only this review document. Native GPU measurements,
preservation fingerprints and concrete launch commands are supplied by the main
study documentation/report. No CLIP experiment, HARDVS run or full training was
performed by this review.

## Final trainer verification addendum

The reviewer independently re-read the final `verify_exports` implementation on
2026-09-24 and refreshed the trainer fingerprint above. The final change adds
exact equality of every live-model tensor against the last checkpoint, plus
exact state-key and tensor equality between each classifier-free export and its
corresponding checkpoint. These checks strengthen the existing logit/embedding
comparisons and also reject stale exports. They do not change optimization,
data access, model architecture, checkpoint contents or the launch path. No
actionable issue was found in the final change. This addendum is a source review;
it does not claim an additional GPU test or authorize a launch.
