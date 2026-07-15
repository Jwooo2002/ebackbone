# Production representation analysis

## Reproducible selection

The bounded analysis uses the sorted 100-class archive catalog and the following
fixed sample IDs from train class indices `0,18,36,54,72,90` and different
validation class indices `9,27,45,63,81,99`:

- `train/n01440764/n01440764_3236.npz`
- `train/n01582220/n01582220_549.npz`
- `train/n01667778/n01667778_1087.npz`
- `train/n01729322/n01729322_1862.npz`
- `train/n01773157/n01773157_2207.npz`
- `train/n01820546/n01820546_5937.npz`
- `validation/n01518878/ILSVRC2012_val_00001031.npz`
- `validation/n01631663/ILSVRC2012_val_00000258.npz`
- `validation/n01692333/ILSVRC2012_val_00001475.npz`
- `validation/n01748264/ILSVRC2012_val_00003143.npz`
- `validation/n01796340/ILSVRC2012_val_00000860.npz`
- `validation/n01855672/ILSVRC2012_val_00000547.npz`

This yields six samples per available split and twelve different classes.
Event counts span `81,425-131,468`, observed durations span `49,912-50,333 us`,
and positive-event fractions span `0.454-0.532`.

## Candidate sweep

The executed NumPy harness evaluated 44 candidates: native `480x640` and
coordinate-scaled `224x224`; raw and `log1p` frames; voxel bins `4,5,8` crossed
with collapsed, signed, or separated polarity and hard or linear temporal
binning; and time surfaces with zero or exponential-floor backgrounds. It used
three renderer-only repetitions per candidate/sample after one warm-up. All
real candidate tensors were finite.

Key rejection evidence:

- `224x224` saves 6.122 times dense storage but merges 56.53% of occupied native
  polarity-pixel locations overall.
- Signed B5/hard/224 voxels lost up to 10,390 events from absolute mass through
  opposite-polarity cancellation. Collapsed voxels discard polarity by design.
- Linear interpolation preserved mass within `3.5e-4` events in float32 and is
  continuous at bin boundaries; hard assignment is discontinuous.
- A decay-floor time surface has zero sparsity and gives an absent polarity
  branch a nonzero value; the occupancy-masked surface leaves it exactly zero.

## Checked-in production output statistics

These values come from `render_production_representations`, not from the probe
renderer. Percentiles are over nonzero values; latency is reported for the full
bundle separately because the three tensors are generated together.

| Tensor | Shape | MiB | Zero fraction | Min | Max | Mean | Std | p50 | p90 | p99 | p99.9 | p99.99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| event frame | `[2,480,640]` | 2.344 | 0.85426 | 0 | 3.61092 | 0.11455 | 0.28821 | 0.69315 | 1.09861 | 1.60944 | 1.79176 | 2.48491 |
| voxel grid | `[2,5,480,640]` | 11.719 | 0.93644 | 0 | 2.56612 | 0.02704 | 0.11918 | 0.43531 | 0.67724 | 1.02767 | 1.29794 | 1.58219 |
| time surface | `[2,480,640]` | 2.344 | 0.85426 | 0 | 1 | 0.04504 | 0.14967 | 0.20283 | 0.75825 | 0.96126 | 0.98693 | 0.98700 |

The full production call took 29.764 ms median (p90 34.626 ms, range
22.384-37.982 ms). The retained tensors total 16.406 MiB. On one 81,425-event
sample, tracemalloc measured 44.224 MiB peak host allocation. Warm OS-cache
archive catalog/read/decode was measured separately at 18.321 ms/sample.

Frame integer counts conserve the raw event count exactly. After `log1p` and
`expm1`, the maximum per-polarity frame discrepancy was `8.49e-5` events. The
interpolated voxel accumulator conserves global and per-polarity mass to
float32 tolerance; the maximum selected-output per-polarity discrepancy after
`expm1` was `4.72e-4` events. The opposite polarity channel remained exact zero
in the empty-branch test.

Pearson correlations between per-sample tensor mean and event count were
`0.9491` (frame), `0.9846` (voxel), and `0.8331` (surface). Correlations with
observed duration were `0.3824`, `0.4350`, and `0.2955`, respectively. Duration
varied by less than one percent in this set, so the latter values document the
bounded set but do not support a general duration-sensitivity conclusion.

## Degenerate synthetic checks

Focused synthetic cases supplement, but do not replace, the real-data evidence.
They verify closed interval endpoints, a zero-duration nonempty sample, an empty
polarity branch, untouched pixels, repeated timestamps, deterministic repeated
output, mass conservation, shared raw identity, and cache invalidation. Empty
full samples are rejected rather than populated with fabricated events.
