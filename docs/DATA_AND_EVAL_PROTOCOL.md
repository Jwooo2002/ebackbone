# Data and Evaluation Protocol

## Evaluation unit

The classification unit is one raw event recording or one explicitly defined temporal window from a recording.

Each sample has:

- a raw event sequence
- one class label
- a stable sample identifier

The precise dataset and slicing policy are currently `TBD` and must be confirmed from the actual data pipeline.

## Raw event contract

Expected raw event fields:

- spatial coordinate `x`
- spatial coordinate `y`
- timestamp `t`
- polarity `p`

Exact storage layout, dtype, timestamp unit, ordering, and polarity encoding are `TBD` until inspected.

## Representation generation

For a B1 sample, event frame, voxel grid, and time surface must be generated from the same source sample and the same temporal interval.

The following parameters must be recorded:

- temporal start and end
- event filtering rules
- number of temporal bins
- number of rendered frames
- spatial resolution
- polarity channels
- normalization
- clipping
- padding or truncation

Do not invent unresolved values. Mark them `TBD`.

## B0 protocol

### Train input

Event frame generated from the raw event sample.

### Test input

Event frame generated using the same rendering policy as training.

## B1 protocol

### Train input

- event frame
- voxel grid
- time surface

All three must come from the same raw event sample and temporal interval.

### Test input

The same three representations using the same rendering policy.

## Fair comparison requirements

B0 and B1 must share:

- dataset split
- raw sample identity
- temporal observation interval
- label set
- spatial crop and resolution
- augmentation policy where applicable
- optimizer
- scheduler
- epoch or optimizer-step budget
- validation and checkpoint-selection rule
- classification head type

Model parameters and compute may differ. These differences must be measured rather than hidden.

## Leakage prevention

Do not use:

- validation or test labels during representation generation
- class-dependent temporal slicing
- class-dependent event filtering
- cached representations generated from a different split
- future events outside the defined sample interval

Any dataset-level normalization statistics must be computed from the training split only.

## Metrics

Primary:

- top-1 accuracy

Conditional:

- top-5 accuracy when the number of classes makes it meaningful

Efficiency metrics after functional validation:

- parameter count
- preprocessing latency
- forward latency
- peak memory
- throughput

## First required data probe

Before model implementation, inspect at least one real sample and report:

- raw event shape and fields
- event count
- temporal duration
- frame tensor shape
- voxel tensor shape
- time-surface tensor shape
- dtype and value range for each
- confirmation that all three cover the same temporal interval
