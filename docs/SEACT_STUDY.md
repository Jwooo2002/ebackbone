# SeACT downstream study: fine-tune three, then train three from scratch

Authorized scope: nine full runs in total. N-ImageNet Mini hierarchy-only,
latent-only and dual fusion are complete. HARDVS raw data was not located in
the bounded local-storage search; the user explicitly authorized SeACT as the
fallback. All six downstream runs therefore use SeACT. No CLIP or RGB input is
used, and the old hierarchy/TS queue remains paused.

## Completed Mini results

| Mode | Best internal validation top-1 | Best epoch | Epoch-50 top-1 |
| --- | ---: | ---: | ---: |
| hierarchy_only | 74.42% | 35 | 73.86% |
| latent_only | 47.34% | 50 | 47.34% |
| dual | 71.38% | 38 | 70.60% |

Each completed run passed strict checkpoint and backbone-export reload checks.
These are 5,000-example Mini internal-validation scores, not SeACT results.

## Data and split

Raw root: `/mnt/ssd1/PycharmProjects/datasets/ExACT/raw/SeAct`, 580 AEDAT4
recordings. The event-stream sensor header is 260 high by 346 wide. Decode only
its event stream; raw-file hashing does not imply using other sensor streams.
The separate existing PNG preprocessing applies adaptive sampling and is not
used. `tools/prepare_seact.py` provides a lossless structured event cache with
int64 microsecond timestamps, int16 coordinates and binary polarity. Decoder
source/version, raw hashes and cached-array hashes are recorded.

Membership comes from the bundled ExACT `SeAct_train.txt` and `SeAct_val.txt`.
The latter 116 recordings are reserved as final held-out evaluation. A fixed
hash-selected recording per class is removed from the released 464-example
training membership to form 58 internal-validation examples, leaving 406
training recordings. Mapping uses subject plus the four-digit recording code
and full timestamp to handle local removal of the `khao`/`xu` prefixes.
The older project's subject-disjoint 464/116 transfer split is not reused.

There are 58 classifier classes. All 58 occur in train/internal validation;
the released 116-example held-out set covers **53 of 58**. It must not be
reported as a balanced 58-class evaluation. Internal validation alone chooses
the checkpoint. Final held-out evaluation occurs once after all 50 epochs,
using the selected best checkpoint. A saved test identity prevents inadvertent
re-evaluation on resume. Held-out events were not decoded during preparation
or preflight; their verified cache is created lazily during final evaluation.

## Full-event representations and model

Every recording retains every event over the same closed observed support.
Coordinates stay native. The canvas is padded only at bottom/right to 288x352
for the hierarchy's spatial strides. Points are `[x/345,y/259,t_normalized,2p-1]`.
Frame `[2,288,352]` is log1p counts; voxel `[2,8,288,352]` is log1p interpolated
event mass; surface `[2,288,352]` uses latest normalized time, decay ratio0.2
and empty-pixel zero. No temporal windows, event sampling, resizing or label-
dependent representations are introduced.

Some recordings contain millions of events: the largest training recording has
3,705,975. To bound activation memory, the hierarchy computes its point MLP and
weighted voxel sums in 65,536-event chunks with activation checkpointing. All
chunk sums and event masses combine **before** division/log1p. Chunks do not
become examples or temporal windows. CPU forward/gradient checks against the
original aggregation passed; FP32 summation order can change roundoff.

Fine-tuning loads the corresponding Mini **best** backbone export, including
both branches and the learned mixing scalar when present, and creates a fresh
58-class classifier. All parameters are trainable. Scratch uses the identical
architecture with fresh backbone weights. The new classifier's seed and initial
weights match the corresponding fine-tuned run. Strict initialization hashes,
head hashes, checkpoint tensors and classifier-free export tensors are checked.

## Matched six-run optimization

All six use two GPUs, **one recording per GPU and accumulation4**, effective
global batch8; the final group uses its actual sample count. This downstream
batch differs from the Mini protocol and is shared by all six downstream runs.
All six use 50 epochs, SGD learning rate0.01, momentum0.9, weight decay0.0001,
cosine decay over50, seed20260908, BF16 autocast and FP32 parameters/aggregation.
No variant-specific tuning or auxiliary loss is used. This is one controlled
fine-tune-versus-scratch schedule, not a claim of independently optimized peaks.

## Verification and execution

CPU verification: **18 tests passed in11.12s**, including data identity/mass,
closed endpoints/zero duration, padding, corruption rejection, held-out gating,
chunked gradient equivalence, strict transfer/head initialization, two-process
Gloo accumulation and tails, exact resume and checkpoint/export reloads.
Three supervisor tests cover run order, duplicate-launch rejection and stopping
before training when a preflight or Mini dependency fails. Independent review:
`docs/SEACT_REVIEW.md`.

The detached supervisor runs six native preflights before any full downstream
run. Each uses the eight largest training recordings, one optimizer group,
two internal-validation examples and strict reload verification. It requires
finite connected gradients, matching rank gradients, component updates and at
least10% observed GPU memory headroom. Preflight checkpoints never initialize
production runs. Any failure stops the queue without automatic retries.

Execution order after preflights:
`finetune hierarchy_only -> latent_only -> dual -> scratch hierarchy_only -> latent_only -> dual`.

Artifacts: `/mnt/ssd1/PycharmProjects/ebackbone_v3_seact_artifacts/`.
`study_status.json` is the live authority, `study_plan.json` records authorization
and protocol, `snapshot_20260926/` contains the frozen source/config/manifests,
and `snapshot_sha256.json` guards their contents before and after every child.
`runs/{regime}_{variant}/` contains histories, best/last checkpoints, backbone
exports and the final report. `nine_results.csv` separates Mini validation from
SeACT internal-validation and held-out metrics. It updates after each verified
full run. No SeACT accuracy claim should be inferred from a preflight.

## Launch verification, 2026-09-26

All six native two-GPU preflights passed before the first full run started.
The detached supervisor is training `finetune_hierarchy_only`, then continues
with the other two fine-tuning runs and all three scratch runs. The startup
audit verified all frozen snapshot hashes, Mini checkpoint/export identities,
and the unchanged paused legacy queue. See
`../ebackbone_v3_seact_artifacts/startup_verification.json` for evidence and
`../ebackbone_v3_seact_artifacts/study_status.json` for live progress.
