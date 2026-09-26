# Hierarchy / CLIP bounded comparisons

Implements the September 20 CLIP proposal in task `0- 총괄`
(`01a093eb-cd6d-7843-8233-563c2dcded12`). The proposal is preserved in
`/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_clip_artifacts/proposal_source.md`.
The downstream action-recognition dataset has **not been selected**. Mini is
used only to verify integration; these probes establish no HAR accuracy,
zero-shot ability, or benefit from CLIP. No full training is launched.

## Comparisons

| Name | Event path | Class prediction |
| --- | --- | --- |
| `hierarchy_temporal` | Shared pretrained hierarchy per window → spatial mean → temporal transformer (256D) | New linear head, CE |
| `hierarchy_text` | Same hierarchy → spatial mean → same temporal architecture (256D) → learned 256→512 projection | Fixed normalized CLIP text vectors, similarity CE |
| `hierarchy_clip_vit` | Same hierarchy spatial maps → token adapter → pretrained CLIP ViT-B/16 → temporal transformer (512D) | Same fixed CLIP text vectors, similarity CE |

All three start from the same hierarchy-only `backbone_best.pt` from the
global-batch-64, 50-epoch study. Loading drops only the old classifier, verifies
all backbone keys/shapes and uses strict loading. It does not substitute TS V2
for the hierarchy-only backbone. Mini-selected pretraining checkpoints overlap
the Mini probe's data; this is an engineering check, not an independent
downstream evaluation.

The CLIP loader checks the official OpenAI file SHA256 and strictly loads the
complete checkpoint before reusing its visual components. This workspace uses
`/mnt/ssd1/PycharmProjects/model_zoo/pretrained_CLIP/ViT-B-16.pt` and the compatible
local CLIP architecture/tokenizer under
`/mnt/ssd1/PycharmProjects/tranfer_candidate/ExACT/model/clip`. Source and BPE
hashes are recorded. No weights are downloaded and no random CLIP fallback is
accepted. Tiny test fixtures are explicitly separate from pretrained evidence.

## Data and tensor contracts

- Immutable Mini split: 124,395 train and 5,000 internal validation samples,
  100 classes. Both originate from the official training source. Full manifest
  identity, source-locator and stored-NPZ SHA256 overlap checks pass.
- One deterministic train sample and one validation sample are decoded per
  integration probe. Validation is forward-only. No final-test event payloads
  or evaluation are used; protected-file checks hash existing manifest bytes.
- Four equal-duration windows cover each stored event stream. Interior bounds
  are left-closed/right-open; the final endpoint belongs to the last window.
  Every event is retained once, with original within-window order and polarity.
  Labels do not select events or determine window boundaries.
- Coordinates stay native 480×640. Points remain `[x/639,y/479,t_local,2*p-1]`,
  float32. Each window uses the existing point MLP and 8-bin point→voxel→frame
  hierarchy unchanged. Time is normalized against the window's fixed bounds.
- Packed counts have shape `[B*K]` in sample-major/window-minor order. Empty
  windows have zero events and an explicit false mask. A zero-duration stream
  places its events in the last window at temporal bin 7. Padding does not
  affect temporal aggregation.
- Hierarchy maps: `[B,K,256,15,20]`. Spatial adapter: adaptive spatial pooling to
  14×14, learned 256→768 pointwise projection and LayerNorm, yielding
  `[B*K,196,768]` patch tokens. Original pretrained CLS and positional embeddings
  create 197 tokens. Original CLIP pre-norm, Transformer, post-norm and 768→512
  projection produce `[B,K,512]`. The RGB patch convolution is bypassed.
- Temporal aggregation uses learned ordered positions, one 4-head Transformer
  layer, padding masking and a masked mean. Its dimensions differ across the
  visual and direct-text comparisons; this is not a compute-matched study.

Mini's ordered class IDs are mapped directly to readable first-lemma names from
a local timm synset table. All 100 names are prepared once in manifest-label
order. Both CLIP comparisons use exactly `a photo of a {class_name}.` for every
candidate class. The text encoder, text vectors and pretrained logit scale stay
fixed. Action prompts must be chosen once when a HAR dataset is selected; no
per-sample or ground-truth-conditioned prompt selection is implemented.

## Fine-tuning stages

1. `connectors`: freeze the hierarchy and CLIP; train the new adapter, temporal
   module and classifier/output projection where present. Frozen visual layers
   still propagate gradients to the adapter.
2. `selective`: retain those trainable connectors and unfreeze hierarchy
   `temporal_collapse` plus `frame_stage`; for the visual comparison, also
   unfreeze only the last two CLIP visual Transformer blocks. These pretrained
   parameters use 0.1× the connector learning rate. The point MLP, voxel stages,
   CLIP patch convolution, CLS/positions, norms/projection and text stay frozen.

`configure_stage` returns disjoint optimizer groups; recreate the optimizer
after a stage change. `set_staged_train` holds fully frozen modules in eval
mode. The bounded runner uses AdamW with connector LR 1e-4 and pretrained LR
1e-5, with exactly **one update per stage per comparison**. These settings are
engineering checks, not a chosen downstream training schedule.

Comparison checkpoints validate variant, class/prompt order, split/sequence
identity, pretrained sources, stage and trainable names before strict model and
optimizer loading. The probe corrupts a trainable tensor after saving, reloads
the checkpoint, and checks every loaded tensor and its evaluation output
exactly. CUDA evaluation enables deterministic algorithms (and the required
cuBLAS workspace setting): the original hierarchy's CUDA scatter otherwise
introduces small accumulation-order differences even without a reload.
Training-step kernels remain at their normal settings. Scratch downstream
checkpoints are removed after successful verification; existing pretrained
checkpoints are read-only. This is checkpoint-loading support, not a claim of
a complete long-run trainer or RNG-exact training resume.

## Commands and artifacts

Run from `/mnt/ssd1/PycharmProjects/ebackbone_V3`. No command accepts an epoch
budget or launches a full training loop.

```bash
export PYTHONPATH=/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_clip_artifacts/tokenizer_dependencies
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
CUDA_VISIBLE_DEVICES='' /home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m pytest -q tests/test_hierarchy_clip_temporal.py tests/test_hierarchy_clip_models.py tests/test_hierarchy_clip_staging.py tests/test_hierarchy_clip_gate.py tests/test_hierarchy.py
CUDA_VISIBLE_DEVICES='' /home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m ebackbone_v3.hierarchy_clip_probe --config configs/hierarchy_clip.engineering.json --output-dir /tmp/hierarchy_clip_cpu_check --device cpu
/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python -m ebackbone_v3.hierarchy_clip_probe --config configs/hierarchy_clip.engineering.json --output-dir /tmp/hierarchy_clip_gpu_check --device cuda:0
```

Output directories must be new or empty. The isolated tokenizer directory
contains local copies of pure-Python `ftfy==6.2.0` and `wcwidth==0.2.13`, with
source hashes. The existing training environment is unchanged; its `regex`
package is reused.

Before any CUDA allocation, the runner verifies V2's completed 50-epoch report,
best/last checkpoint and backbone hashes, a paused/completed queue, a released
supervisor lock, dead study process IDs and both named GPUs without compute
processes. Failed or unavailable host visibility blocks the probe. Devices are
mapped explicitly by study UUID. The gate neither stops nor resumes the old
queue, and is an observation rather than an exclusive GPU reservation.

Reports are under
`/mnt/ssd1/PycharmProjects/ebackbone_v3_hierarchy_clip_artifacts`:
`data_provenance_audit.json`, `mini_class_names.json`, `cpu_tests.xml`,
`cpu_verified/report.json`, `gpu0_verified/report.json`, and
`gpu1_verified/report.json`. See `README.md`
there for final verification status.

POKER polarity/motion semantics, learned prompts, LLM descriptions, paired-RGB
alignment, extra contrastive losses, LoRA and a random-hierarchy ablation remain
deferred. The next research task is to select a downstream dataset and official
split, then specify its sensor-coordinate mapping, raw-event loader and fixed
class prompts against these contracts.
