# Baselines B0 and B1

## Shared objective

Both baselines perform supervised event classification from random initialization.

```text
loss = cross_entropy(class_logits, class_label)
```

Both use a single linear classification head after the encoder output has been reduced to one sample-level embedding.

The B1 pooling and fusion details are not fixed yet. D014 fixes only the B0
production architecture and pooling choice.

## B0 — Frame-only baseline

### Input

One event-frame representation generated from a raw event sample.

For N-ImageNet mini, D012 selects one accumulated native-resolution,
polarity-separated `log1p` count frame. A frame sequence is not part of B0.

### Model contract

```text
frame tensor
→ frame encoder
→ sample embedding
→ linear classification head
→ class logits
```

### Selected production architecture

D014 selects a standard randomly initialized ResNet-18 for production training,
validation, and scientific B0 results. Its native input is `[B,2,480,640]`; its
two-channel stem is `Conv2d(2,64,7,stride=2,padding=3,bias=False)`. It retains
the standard BasicBlock stages, BatchNorm, max-pooling, and adaptive average
pooling to a 512-dimensional embedding, followed by exactly one
`Linear(512,100)`. It has 11,224,676 trainable parameters and no external-weight
or network-loading path.

The 68,148-parameter `compact_debug` CNN remains available only to the
`train-b0-debug` engineering diagnostic. Its results are not scientific B0
results and must not be substituted into a B0/B1 comparison.

### Initialization

Random.

### Training

End-to-end supervised classification.

### Test

Frame-only input.

## B1 — Tri-representation baseline

### Inputs

- event frame
- voxel grid
- time surface

All inputs are generated from the same raw event sample and temporal interval.

### Model contract

```text
frame tensor ───────┐
voxel tensor ───────┼─ tri-representation encoder ─ sample embedding
surface tensor ─────┘
                                         ↓
                              linear classification head
                                         ↓
                                    class logits
```

### Initialization

Random.

### Training

End-to-end supervised classification.

### Test

All three representations.

## Unresolved architecture decisions

Do not resolve these implicitly during implementation:

1. one accumulated frame versus frame sequence
2. input-level, feature-level, or embedding-level fusion
3. shared, partially shared, or separate encoder weights
4. B1 pooling method (B0 global average pooling is fixed by D014)
5. representation-specific stems
6. representation-specific model-side normalization beyond the fixed D012 input transforms
7. compute-matched versus parameter-matched auxiliary comparisons

Each decision requires a bounded design task and an entry in `DECISIONS.md`.

## Minimum B0/B1 comparison

| ID | Input | Initialization | Head | Loss | Train input | Test input |
|---|---|---|---|---|---|---|
| B0 | Frame | Random | Linear | CE | Frame | Frame |
| B1 | Frame + Voxel + Time Surface | Random | Linear | CE | All three | All three |

## Interpretation

`B1 - B0` measures the effect of the tri-representation system as a whole. It
does not by itself isolate fusion design, parameter count, or compute. The
future comparison must report parameter-capacity and efficiency differences;
any capacity-matched control is a separate documented auxiliary comparison.
