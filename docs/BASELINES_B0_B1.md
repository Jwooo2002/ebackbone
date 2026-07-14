# Baselines B0 and B1

## Shared objective

Both baselines perform supervised event classification from random initialization.

```text
loss = cross_entropy(class_logits, class_label)
```

Both use a single linear classification head after the encoder output has been reduced to one sample-level embedding.

The pooling and fusion details are not fixed yet.

## B0 — Frame-only baseline

### Input

One event-frame representation generated from a raw event sample.

Whether this is one accumulated frame or a sequence of event frames is `TBD` and must be decided from the target dataset and downstream convention.

### Model contract

```text
frame tensor
→ frame encoder
→ sample embedding
→ linear classification head
→ class logits
```

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
4. pooling method
5. representation-specific stems
6. normalization per representation
7. compute-matched versus parameter-matched auxiliary comparisons

Each decision requires a bounded design task and an entry in `DECISIONS.md`.

## Minimum B0/B1 comparison

| ID | Input | Initialization | Head | Loss | Train input | Test input |
|---|---|---|---|---|---|---|
| B0 | Frame | Random | Linear | CE | Frame | Frame |
| B1 | Frame + Voxel + Time Surface | Random | Linear | CE | All three | All three |

## Interpretation

`B1 - B0` measures the effect of the tri-representation system as a whole. It does not by itself isolate fusion design, parameter count, or compute.
