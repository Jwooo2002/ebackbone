# Independent SeACT implementation review

Status: PASS for bounded implementation review and CPU verification. Native
two-GPU preflights remain a separate gate for full training.

Reviewed `ebackbone_v3/seact_data.py`, `ebackbone_v3/seact_models.py`, `ebackbone_v3/seact_train.py`,
`tests/test_seact_models.py`, `tests/test_seact_train.py`, and the downstream
supervisor at
`/mnt/ssd1/PycharmProjects/ebackbone_v3_seact_artifacts/supervise_seact.py`.

## Static findings

- Chunked point processing accumulates weighted feature sums and event mass
  across every event before division and `log1p`. It does not create temporal
  windows. The point MLP uses per-event LayerNorm, so chunk boundaries do not
  change batch statistics. Floating-point addition order differs from the
  original unchunked implementation; tolerance-based forward/gradient tests
  are required.
- Fine-tuning loads the exact selected Mini backbone with matching variant,
  dimension and latent width, and leaves the new 58-class head initialized
  identically to the corresponding scratch model. Scratch forbids a source
  checkpoint. All parameters, including the dual global scalar, train.
- With two ranks, local batch 1 and accumulation 4, each optimizer group uses
  the sum of actual sample counts. Scaling summed cross-entropy by
  `world_size / group_samples` compensates for DDP gradient averaging. The
  406-sample training split therefore produces 50 groups of 8 and one of 6,
  without repeating or dropping samples.
- Evaluation uses the unwrapped model and combines metric sums globally, so
  unequal rank lengths do not create DDP forward collectives.
- Resume checks source/config/manifest/pretrained identity. Checkpoints save
  model, optimizer, scheduler and every rank's RNG state. Exports exclude the
  classifier and are checked against checkpoint tensors and embeddings.
- Held-out event access occurs only after contiguous completed epochs 1..50
  and verification of the selected checkpoint's identity and epoch. A saved
  final-test result is reused after a report-write failure.
- The supervisor verifies all six bounded preflights before full training,
  orders fine-tune3 before scratch3, checks Mini export and frozen-source
  hashes, preserves the legacy queue fingerprint, and stops after failure.

## Integration findings

- The configuration now references the prepared `released-v1` manifests.
- The data adapter checks manifest hashes, split identity, recording/raw-hash
  disjointness, raw size/mtime/content, cache content hash, event count, temporal
  endpoints and geometry. Deferred held-out decoding checks the decoder source
  hash before executing it. The packed collator preserves per-recording voxel
  routing offsets.
- All representations use the same full structured event array. Native sensor
  coordinates occupy the upper-left 260x346 region of the padded 288x352 canvas;
  point normalization uses the native 345/259 coordinate maxima. No events are
  dropped or temporally partitioned.
- No model, trainer or supervisor defect requiring an edit was found. Existing
  trainer/model tests passed without modification.

## Verification

Using `/home/iulab3/anaconda3/envs/ssmerpp-e2v/bin/python`:

```sh
python -m pytest tests/test_seact_models.py tests/test_seact_train.py -q
python -m pytest tests/test_seact_train.py::test_two_rank_six_regimes_global_mean_accumulation_and_resume -q
```

The first command passed 14 tests; its one DDP test could not bind loopback
inside the sandbox (`Cannot resolve 127.0.0.1 to a (local) address`). The second
command reran that exact test with approved host socket access and passed in
9.53 seconds. Total: all 15 distinct tests passed. Tests opened synthetic data
only and did not use CUDA or launch training.

Verified behavior includes original-versus-chunked forward and gradient
agreement, exact fine-tune initialization and fresh classifier matching,
scratch isolation, export reloads, configuration pinning, output collision and
resume identity guards, complete-training test access, global-mean gradients
across all six regimes, unequal rank evaluation tails, production accumulation
groups of 8 and 6, and bit-exact uninterrupted/resumed training state.

The largest prepared training recording contains 3,705,975 events; the largest
internal-validation recording contains 2,404,392. The largest-eight training
preflight therefore covers the observed validation event-count range, although
the held-out event files have deliberately not been decoded at this stage.
