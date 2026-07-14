# Project Context

## Project name

`ebackbone_V3`

## Research direction

The project studies downstream event classification using multiple complementary representations generated from the same raw event sample.

The initial encoder receives:

- event frame
- voxel grid
- time surface

Train and test use the same representation set for each baseline.

## Relation to earlier SSMER work

Earlier SSMER-style work used multiple event representations primarily as self-supervised views for pretraining and then transferred a backbone to a downstream model.

`ebackbone_V3` starts from a different downstream assumption:

> the downstream encoder itself may consume all three representations at both train and test time.

This means the project is not initially a backbone-only transfer study. It is first a supervised multi-representation classification study.

## Immediate research question

Does a tri-representation encoder improve event classification over a frame-only encoder when both operate on the same raw event samples and temporal intervals?

## Current scope

- sample-level event classification
- B0 frame-only supervised baseline
- B1 tri-representation supervised baseline
- random initialization
- linear classification head
- cross-entropy loss

## Deferred questions

- optimal fusion location
- shared versus separate encoder weights
- self-supervised pretraining
- auxiliary reconstruction
- Event2Vec alignment
- EventBind-inspired prompts
- detection transfer
- efficiency-oriented architecture compression

These must not be introduced before B0 and B1 are operational and comparable.
