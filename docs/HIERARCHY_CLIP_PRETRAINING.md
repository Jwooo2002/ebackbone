# N-ImageNet Mini hierarchy / CLIP pretraining

> Archived, 2026-09-26: CLIP was dropped. This document preserves the historical
> protocol; its source, tests and configs have been removed from the active
> checkout. Recover them from
> [archive/pre-cleanup-20260926](https://github.com/Jwooo2002/ebackbone/tree/archive/pre-cleanup-20260926)
> (commit `44d0148`). Commands and authorization statements below describe that
> historical study and do not authorize restarting it. See the
> [current documentation index](README.md).


This is the explicitly authorized Mini pretraining study, following the bounded
implementation in `HIERARCHY_CLIP_COMPARISONS.md`. It is **EventBind-inspired
event–text alignment**, not a full EventBind reproduction. HARDVS is the future
downstream dataset; this study does not load or train on HARDVS. There is no
paired-RGB input, loss, teacher or extra contrastive objective.

## Protocol

| Setting | Value |
| --- | --- |
| Runs, sequentially | `hierarchy_temporal`, `hierarchy_text`, `hierarchy_clip_vit` |
| Input | All native Mini events; four ordered equal-duration windows; no sampling or augmentation |
| Splits | Existing immutable 124,395 train / 5,000 internal-validation samples |
| Initial hierarchy | Same hierarchy-only batch-64 study best export, epoch 35 |
| CLIP | Official-checksum-verified pretrained OpenAI ViT-B/16 |
| Text | Fixed `a photo of a {class_name}.` for all 100 object classes, manifest-label order |
| Distribution | Two-GPU DDP, one model at a time |
| Batch | 32 examples per rank, global 64, one optimizer step per batch; no accumulation |
| Epochs | 50 per comparison |
| Stage 1 | Epochs 1–5: new temporal/head/adapter components only |
| Stage 2 | Epochs 6–50: additionally hierarchy temporal collapse/frame stage and final two CLIP visual blocks where present |
| Optimizer | AdamW, weight decay 0.01; new components LR 1e-4, selected pretrained parameters LR 1e-5 |
| Schedule | One continuous 50-epoch cosine; no reset at the stage boundary |
| Precision | BF16 autocast forward/loss, FP32 parameters, backward outside autocast |
| Determinism | Deterministic PyTorch/CUDA operations, TF32 disabled, fixed seed 20260908 |

The temporal baseline uses ordinary classifier CE. Both alignment models use
CE over similarities to the fixed normalized class text vectors. Text encoding
and the pretrained logit scale remain frozen. No learned prompts are added.

The three architectures are comparisons, not compute-matched models. All use
the same samples, window contract, initialization checkpoint, epoch count and
optimizer policy. Pretraining-selected Mini checkpoints are reused on Mini;
these internal-validation results are not independent downstream or zero-shot
performance estimates.

## Execution and exact resume

The new runtime processes up to 16 hierarchy windows and 8 CLIP windows per
execution chunk. It preserves each full batch and all events. Activation
checkpointing recomputes selected hierarchy/CLIP activations during backward.
The spatial adapter implements the same adaptive-average bins with fixed
separable matrices, enabling deterministic CUDA backward. The original
hierarchy point MLP and point→voxel→frame source remain unchanged.

The trainer rebuilds DDP when stage 2 unfreezes parameters. Connector Adam
moments and step counts are retained; newly unfrozen parameters receive new
Adam state. Epoch LR is calculated from the absolute epoch on the original
50-epoch cosine horizon.

Training sharding neither duplicates nor drops examples. The final 43-sample
global batch is split 22/21 and its summed loss is scaled by its actual sample
count. Validation bypasses DDP forward collectives, sums loss/correct/count
across ranks, and verifies exact sample-ID coverage without padding.

Full `checkpoint_best.pt` and `checkpoint_last.pt` include unwrapped model
weights, Adam state and named parameter groups, scheduler state, stage, epoch,
global step, sampler position, per-rank Python/NumPy/PyTorch/CUDA RNG state,
history, selection result and full configuration/provenance identity. The
trainer resumes at the last completed epoch boundary. It does not promise
mid-accumulation or in-flight DataLoader resume. Strict identity guards reject
changed class order/prompts, source, configuration, world size or split hashes.

Every committed checkpoint has a `transfer_best.pt` or `transfer_last.pt`
bundle containing separately keyed hierarchy backbone, temporal head, projection,
CLIP adapter and visual weights where present. Object classifier/text state is
separated as source-task data. `load_transfer_bundle(..., include_task_head=False,
allow_new_classes=True)` explicitly transfers reusable components while keeping
a future downstream model's new class head/text bank. HARDVS sensor mapping,
class prompts and official split still require a separate downstream task.

## Launch gate and artifacts

The new study lives at
`/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_clip_ddp_b64_e50_artifacts`.
The existing batch-64 hierarchy/TS study remains paused before V3, and its
checkpoints, queue and control files are hash-protected.

The supervisor launches only after all three final preflight reports pass and
match the configuration and snapshotted source. Before each run it rechecks the
old paused queue and idle GPU UUIDs. It stops the new queue if a run fails; it
does not automatically overwrite or restart existing runs.

Preflight covers true batch 32/rank, large-payload training samples, the uneven
tail, explicit gradient averaging, identical synchronized gradients across
ranks, unchanged frozen weights, serial-reference validation, transfer reload,
and bit-exact next updates after reconstructing both model and optimizer. Resume
checks simulate the 5→6 stage boundary and a 6→7 selective-stage boundary; they
are bounded tests rather than completed epochs. Memory stress uses the 67
largest stored training payloads as a label-independent proxy, not a proof of
the maximum decoded event count across the dataset.

Key new modules:

- `ebackbone_v3/hierarchy_clip_train.py`: full trainer and exact sharding/metrics.
- `ebackbone_v3/hierarchy_clip_runtime.py`: memory-controlled model execution.
- `ebackbone_v3/hierarchy_clip_checkpoint.py`: exact state and transfer exports.
- `ebackbone_v3/hierarchy_clip_preflight.py`: bounded DDP verification and launch gate.
- `ebackbone_v3/hierarchy_clip_supervise.py`: separate sequential study supervisor.

The study `config.json`, `snapshot_hashes.json`, `training_gate.json`,
`cpu_tests.xml`, `preflight_final/*/report.json`, `queue_status.json`, and
`runs/*/history.json` are the reproducible configuration, verification and live
status evidence. The study README contains launch commands and current status.
