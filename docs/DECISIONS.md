# Decisions

## D001 — Create a separate project

- **Status:** Accepted
- **Decision:** Develop the new direction in `ebackbone_V3` rather than continuing inside the previous SSMER++ project.
- **Rationale:** The downstream model now consumes three event representations at train and test time. This changes the system boundary from backbone transfer to a multi-representation downstream encoder.
- **Alternatives:** Continue in the prior repository on a new branch.
- **Consequences:** Existing rendering and dataset code may be reused, but prior training code and assumptions must not be copied blindly.

## D002 — Start with B0 and B1 only

- **Status:** Accepted
- **Decision:** Implement and validate frame-only B0 and tri-representation B1 before introducing self-supervised objectives or advanced fusion.
- **Rationale:** The first question is whether the three-representation downstream input is useful at all.
- **Alternatives:** Start directly from SSMER+, EventBind, or Event2Vec.
- **Consequences:** Initial experiments remain interpretable and bounded.

## D003 — Match train-time and test-time inputs

- **Status:** Accepted
- **Decision:** B0 uses frame at train and test. B1 uses frame, voxel grid, and time surface at train and test.
- **Rationale:** The project evaluates a downstream tri-representation encoder rather than training-only privileged representations.
- **Alternatives:** Train with three representations and test with frame only.
- **Consequences:** B1 requires representation generation during inference and must later report preprocessing and model cost.

## D004 — Use a linear classification head initially

- **Status:** Accepted
- **Decision:** Use one linear layer from the sample embedding to class logits.
- **Rationale:** A stronger head would obscure whether gains come from the encoder or the classifier.
- **Alternatives:** MLP or task-specific nonlinear head.
- **Consequences:** Encoder quality remains the main variable.

## D005 — Defer fusion design

- **Status:** Accepted
- **Decision:** Do not fix input-level, feature-level, or late fusion before the real data contract is inspected.
- **Rationale:** Fusion depends on actual tensor shapes, temporal structure, and encoder compatibility.
- **Alternatives:** Immediately choose concatenation or attention.
- **Consequences:** The next task is a data probe, not model implementation.

## D006 — Defer SSMER+, EventBind, and Event2Vec

- **Status:** Accepted
- **Decision:** Exclude SSL, prompt fusion, and semantic alignment from the first baseline implementation.
- **Rationale:** Their effects cannot be interpreted before B0 and B1 are stable.
- **Alternatives:** Integrate them from the start.
- **Consequences:** They become later experimental increments rather than hidden baseline components.

## D007 — Use an explicit provider for the first real-data probe

- **Status:** Accepted
- **Decision:** Keep the generic probe independent of dataset and rendering code. A configured, importable provider must return raw-event fields, all three rendered tensors, complete rendering metadata, and sample/interval provenance. Representation event-subset attestations must equal a canonical fingerprint computed from the observed post-filter raw fields.
- **Rationale:** No dataset, slicing policy, or representation definition has been selected for V3. An explicit provider makes those choices inspectable and prevents file-extension or sibling-project assumptions from becoming hidden defaults.
- **Alternatives:** Auto-detect a nearby dataset or copy an earlier SSMER renderer into V3.
- **Consequences:** The generic checked-in example remains intentionally unresolved. D009-D011 add one approved dataset-specific provider and resolved configs. A successful probe verifies metadata consistency and raw-subset fingerprint binding while explicitly retaining the provider's responsibility for renderer correctness.

## D008 — Keep synthetic smoke computation outside the baseline architecture

- **Status:** Accepted
- **Decision:** Use a private deterministic CPU harness to exercise B0/B1 input names, a linear head, cross-entropy, backward, and one optimizer step. Report all fixture shapes and toy flatten/project/concatenate operations as synthetic-only assumptions.
- **Rationale:** Executable plumbing is needed before the real data contract is available, while fusion, pooling, encoder sharing, stems, and normalization remain deferred by D005.
- **Alternatives:** Implement a production B1 model before the data and fusion decisions, or omit execution checks entirely.
- **Consequences:** `smoke` proves environment and autograd health only. It is not accuracy evidence and does not make production architecture or rendering decisions.

## D009 — Select the archive-backed 100-class N-ImageNet mini release for the first real probe

- **Status:** Accepted
- **Decision:** Use `/mnt/hdd1/datasets/event/n_imagenet/mini_zenodo/archives` as the real-probe source. Archive membership, not the bundled full-dataset text manifests, defines the mini split.
- **Verified facts:** This is the only N-ImageNet candidate below `/mnt/hdd1/datasets/event/`. Ten `train_Part_*.zip` files contain 100 class tar payloads and 129,395 training NPZ samples. `mini_validation_split.zip` contains the same 100 WordNet synsets and 5,000 validation samples, 50 per class. The train/validation class-relative sample-ID intersection is empty. No independent test archive, manifest, or directory is present. MD5 values for all ten training ZIPs, the validation ZIP, and both text manifests match Zenodo record `6388221`.
- **Class mapping:** The dataset-native class identifier is the WordNet synset directory. Numeric indices are deterministic bookkeeping: lexicographically sort the 100 synsets and enumerate from zero (`n01440764 -> 0`, `n01855672 -> 99`). The provider verifies that train and validation expose the same class set before creating the mapping.
- **Manifest caveat:** Local `train_list.txt` and `val_list.txt` contain 1,281,167 and 50,000 stale absolute paths for the full 1,000-class release. Filtering them to the archive's 100 synsets gives 129,395 and 5,000 entries, but the provider does not use those manifests as its split source.
- **Consequences:** Available supervised splits are train and validation only. An independent test split and human-readable synset names remain `TBD`; validation must not be silently aliased as test.

## D010 — Define one probe sample as the whole stored N-ImageNet event array

- **Status:** Accepted
- **Decision:** One probe sample is the complete `event_data` array from one archive NPZ. Use its observed event support `[min(t), max(t)]` with closed endpoints. Do not slice, resize, augment, filter, pad, truncate, or cache events in the V3 provider.
- **Verified storage contract:** `event_data` is a packed one-dimensional NumPy structured array with field order and dtype `[('x','<u2'), ('y','<u2'), ('t','<u2'), ('p','?')]`, itemsize 7 bytes. The selected sensor grid is 480 x 640 with zero-based `x` as width and `y` as height. Stored timestamps are nondecreasing microseconds with ties; the official N-ImageNet loader at commit `6815335a35b5bdf65623b3e6b1acbe3fe74b2e33` divides raw `t` by `1,000,000` to obtain seconds. Stored polarity is boolean; the official loader interprets `False` as negative and `True` as positive.
- **Validation policy:** The provider fails on an empty or non-positive-duration sample, a changed dtype/layout, unsorted timestamps, invalid polarity storage, or out-of-bounds coordinates instead of silently discarding events. Provider event counts before and after filtering are both the stored array length because provider filtering is `none`.
- **Evidence boundary:** `[min(t), max(t)]` is the observed event-support interval, not a proven acquisition-window boundary. Acquisition/export-time filtering, hot-pixel suppression, and the exact nominal capture-window wording are not encoded in the local NPZ and remain `TBD`.

## D011 — Use an explicit probe-only tri-representation renderer

- **Status:** Accepted
- **Decision:** Render all three tensors together from the same validated full-sample `x`, `y`, `t`, and `p` arrays. Use four voxel bins in the resolved configs. This establishes the real-probe contract only; production B0/B1 renderer adoption remains `TBD`.
- **Event frame:** Float32 `[2,480,640]`, channel order `[False/negative, True/positive]`, one accumulated per-pixel count frame. No value normalization or clipping. Empty pixels are zero.
- **Voxel grid:** Float32 `[4,480,640]`, polarity-collapsed per-pixel counts. With `u=(t-t_min)/(t_max-t_min)`, the bin is `floor(u*(B-1))` clamped to `[0,B-1]`, with no interpolation. Counts are not normalized or clipped. Under this inherited code-derived formula, the final bin contains events at `u=1`; this is a probe choice, not N-ImageNet metadata.
- **Time surface:** Float32 `[2,480,640]`, channel order `[False/negative, True/positive]`. First compute the latest normalized timestamp per polarity/pixel with an explicit maximum reduction, then apply the official N-ImageNet exponential formula `exp(-(1-latest)/tau)` with `tau=0.3` normalized-interval units. The explicit maximum avoids the duplicate-index assignment defect found in the prior sibling renderer. The V3 probe initializes missing latest timestamps to zero, so pixels without events use `exp(-1/0.3)`. No output clipping, padding, or truncation is applied.
- **Alignment proof:** The provider calls one renderer with one field mapping, checks that frame and voxel count sums each equal the raw event count, computes the canonical raw-event fingerprint once, and attaches the same sample ID, observed interval, event counts, and fingerprint to every representation. The generic probe still correctly describes this as provider evidence rather than an independent re-rendering proof.
- **Deferred production choices:** Fusion, pooling, encoder sharing, representation-specific stems, augmentations, caching, training normalization, and whether production B0/B1 should retain these probe renderer choices all remain `TBD`.
