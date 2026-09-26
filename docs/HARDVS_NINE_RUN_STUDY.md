# Nine-result study: Mini pretraining, HARDVS fine-tuning, HARDVS scratch

> Update, 2026-09-26: All three Mini runs completed. The user-authorized SeACT fallback is now active because HARDVS raw events were not located. All six SeACT preflights passed and downstream training started. See [SEACT_STUDY.md](SEACT_STUDY.md) for the executable protocol. The HARDVS investigation below is retained as historical evidence.

The user authorized this order on 2026-09-24:

1. Finish hierarchy-only, latent-only and dual fusion on N-ImageNet Mini.
2. Fine-tune each corresponding Mini backbone on HARDVS.
3. Train the same three architectures from scratch on HARDVS.

This is nine trained models in total, including the already completed Mini
hierarchy-only control. A machine-readable plan is stored at
`../ebackbone_v3_dual_fusion_b64_e50_artifacts/nine_run_plan.json`.
The original no-HARDVS boundary is superseded by this explicit instruction.
CLIP remains excluded. The existing old hierarchy/TS queue remains paused.

## Current evidence and dependencies

Mini hierarchy-only completed 50 epochs: best internal validation 74.42% at
epoch35; final73.86%. Its checkpoints and exports passed strict reload checks.
The remaining Mini variants use the same unchanged frozen source and configuration
and are supervised sequentially by `supervise_remaining.py` in the study artifact
directory. Consult its `remaining_status.json` for actual launch state.

HARDVS raw event data has not yet been resolved to an accessible file. The user
has been asked for its directory or archive path while mounted-storage searches
continue. Split lists and loaders are metadata evidence, not proof of a usable
raw dataset. No HARDVS model has been launched or connected as an automatic
successor. The six runs are authorized but still require a verified data adapter.

## Split decision

The local ExACT checkout contains two incompatible partitions:

| Metadata scheme | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Raw-list 80/20 | 86,168 | 21,542 | none |
| Sampled-list recording IDs 60/10/30 | 64,626 | 10,771 | 32,313 |

The original [HARDVS paper](https://arxiv.org/pdf/2211.09648) describes a
60%/10%/30% train/validation/test protocol, but reports counts
64,526/10,734/32,386. Therefore the local ExACT list counts must not be described
as exact official-paper split counts. Its released-list variant needs explicit
provenance in the eventual report. All six HARDVS runs must share one immutable
recording-level partition; do not mix the 80/20 and 60/10/30 lists.

The intended local approach is to map the disjoint 60/10/30 recording IDs to
raw event files, rather than use their sampled PNG frames. Membership and actual
file availability must be checked before this becomes an executable protocol.
The ExACT example configuration uses a test list under its `Val` key; that
configuration is not reused. Validation chooses checkpoints; held-out test
evaluation is separate, after the protocol is fixed.

## Transfer contract to implement after the raw probe

- Fine-tuning loads each variant's own best Mini backbone export, checks its
  hash and architecture, and creates a new learned 300-class classifier.
  All backbone parameters, including the dual mixing scalar, remain trainable.
- Scratch constructs the identical HARDVS model without Mini weights. The new
  classifier initialization is matched between fine-tuned and scratch controls.
- Optimizer, schedule, epoch budget, sample order, preprocessing and evaluation
  must be matched across the six HARDVS runs to make initialization the intended
  fine-tune-versus-scratch difference. The Mini global64/50-epoch setup is a
  candidate, not yet a validated HARDVS memory or optimization configuration.
- Preserve full event arrays and their full observed temporal support. No CLIP,
  RGB input, temporal-window splitting or event sampling is added.
- Existing Mini rendering/routing is hardcoded for 640x480 and cannot be reused
  blindly. The paper describes a 346x260 sensor. Verify actual x/y bounds,
  timestamp units, polarity convention and event counts on a real training file.
  A separate geometry-aware adapter must preserve every event and document any
  spatial padding needed by the hierarchy's stride contract.
- Verify one real sample, tensor alignment, one bounded optimization step,
  strict Mini-backbone initialization, classifier replacement, scratch isolation,
  and checkpoint/export reload before adding HARDVS to the execution queue.

The [official HARDVS repository](https://github.com/Event-AHU/HARDVS) lists
compact event-file downloads separately from rendered event images. This study
needs raw event arrays capable of reproducing all three representations.
