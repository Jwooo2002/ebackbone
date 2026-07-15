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
- **Consequences:** Dataset-native source splits are train and validation only. An independent official test split and human-readable synset names remain `TBD`. The provider must not fabricate a source split named `test`; D013 separately and explicitly reserves the complete official-validation source as the project-level final `test` role while preserving `source_split=validation` in every record.

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

## D012 — Fix the production B0/B1 representation contract at native resolution

- **Status:** Accepted
- **Scope:** This decision fixes representation generation only. It does not select a model, stem, encoder-sharing policy, fusion, pooling, classifier, augmentation, or training recipe. `ebackbone_v3/representations.py` is separate from the D011 probe renderer, which remains the raw-contract validation path.
- **Shared source and provenance:** Render all tensors in one call from the same complete, validated N-ImageNet mini `x,y,t,p` arrays. Every tensor shares the dataset release, split, source sample ID, canonical raw-event SHA-256 fingerprint, closed observed-support interval `[t_min,t_max]`, and event count. Labels are not accepted by the renderer. Empty samples are rejected; timestamp ties are valid.
- **Spatial size:** Keep native `[H,W]=[480,640]`; do not rescale event coordinates. In a deterministic 12-sample analysis, `224x224` used 6.122 times less dense memory but merged 56.53% of occupied native polarity-pixel locations overall (50.13%-71.37% per sample). Native resolution therefore wins under the required information-preservation-first priority. The pinned official loader uses a `224x224` network input, but that implementation choice is not evidence that rescaling is information-optimal. Consequence: a later bounded model task must address the larger input cost without silently changing this renderer contract.
- **Canonical polarity order:** Use axis order `[False/negative, True/positive]` for the frame, voxel polarity axis, and time surface. This follows raw boolean order and the verified probe convention. The official renderer's `[positive,negative]` order is rejected to avoid remapping the stored boolean values; neither order is dataset metadata, so consistency is the controlling requirement.
- **B0 event frame:** Float32 `[2,480,640]` polarity-separated accumulated counts transformed elementwise with `log1p`; no clipping or per-sample/per-branch scaling. Counts are accumulated as unsigned integers before conversion. `log1p` is monotone and controls the observed count tail while retaining recoverable event mass through `expm1`. Across 12 real samples, raw count maximum was 36 and active-pixel percentiles p50/p90/p99/p99.9/p99.99 were `1/2/4/5/11`; `log1p` reduced the maximum to 3.611 and p99.99 to 2.485. Raw counts are rejected for their wider scale; per-sample max and event-count normalization are rejected because they remove absolute activity and can fail on empty branches.
- **B1 voxel grid:** Float32 polarity-major `[2,5,480,640]` (`P,T,H,W`), with five bins, polarity separation, and linear interpolation between adjacent temporal bins. Normalize time as `u=(t-t_min)/(t_max-t_min)` and coordinate as `z=u*(B-1)`; endpoints lie exactly in bins 0 and 4. Apply `log1p` after accumulation and do not clip. Polarity separation preserves both branch counts; signed voxels are rejected because opposite polarities cancel, and collapsed voxels discard polarity. Linear interpolation is selected over hard bins because it is continuous in time and preserves event mass; the measured float32 mass error before `log1p` was at most `2.91e-4` events. Five bins are a measured cost/information compromise, not an accuracy claim: native B4/B5/B8 dense voxel costs are 9.375/11.719/18.750 MiB. The candidate B5 separated/interpolated accumulator took 12.347 ms median before the selected `log1p`; the actual full production bundle timing is reported separately below. No official N-ImageNet voxel renderer or bin recommendation exists.
- **B1 time surface:** Float32 `[2,480,640]`, the same polarity order, latest normalized timestamp per occupied polarity-pixel, and `exp(-(1-latest)/0.3)`. Empty pixels and a wholly empty polarity branch are exactly zero via a separate occupancy mask. No additional normalization or clipping is applied. On the 12 samples it was finite in `[0,1]`, 85.43% zero, with 6.385 ms median rendering. The probe/official-style decay-floor background is rejected because it makes every empty pixel nonzero and aliases emptiness with a real earliest event.
- **Degenerate time:** When a nonempty sample has `t_min==t_max`, define every normalized timestamp as `u=1`: all voxel mass goes to the last bin and occupied time-surface pixels equal one. This avoids division by zero and treats each event as occurring at the shared interval end. Do not fabricate events. The real provider may continue to reject this case in its stricter raw-contract validation path.
- **Determinism, normalization, and clipping:** All outputs are deterministic, finite, unclipped float32. Count-like frame and voxel tensors use only elementwise `log1p`; the time surface is already bounded. Empty pixels remain exactly zero in every representation. Tests verify frame mass and interpolated voxel mass through `expm1` within floating-point tolerance.
- **Cache format and invalidation:** Caching is optional and on-demand; do not precompute the full release implicitly. Use an uncompressed, pickle-free NPZ with fixed tensor names and a canonical UTF-8 JSON manifest. The content-addressed SHA-256 key covers cache schema, renderer contract version, dataset release/raw dtype and resolution contract, split/sample ID, canonical raw fingerprint, interval/closure, event count, and every renderer parameter. Writes are atomic. Reads validate the expected key, source, parameters, fixed entry set, shapes, dtypes, finiteness, and tensor SHA-256 values. Any parameter or raw identity change is a cache miss/error, never silent reuse.
- **Measured cost:** Dense output storage is 2.344 MiB/sample for B0 and 16.406 MiB/sample for all B1 tensors. The checked-in full production renderer, including validation, fingerprint verification, allocation, interpolation, and transforms, took 29.764 ms median per sample (p90 34.626 ms; range 22.384-37.982 ms). A tracemalloc check on one 81,425-event sample measured 44.224 MiB peak host allocation versus 16.406 MiB retained output. Warm OS-cache archive catalog/read/decode was measured separately at 18.321 ms/sample. These are NumPy CPU measurements on the bounded analysis host, not training throughput or GPU memory.
- **Evidence set:** The fixed sample IDs listed in `docs/REPRESENTATION_ANALYSIS.md` contain six train classes at sorted class indices `0,18,36,54,72,90` and six different validation classes at `9,27,45,63,81,99`. Event counts ranged `81,425-131,468`, durations `49,912-50,333 us`, and positive polarity fractions `0.454-0.532`. The analysis evaluated 44 candidates spanning both spatial sizes, raw/log frame, voxel bins 4/5/8, collapsed/signed/separated polarity, hard/interpolated binning, and zero/decay-floor surface backgrounds. All real-data candidate outputs were finite.
- **Rejected alternatives and consequences:** `224x224` is cheaper but loses native spatial occupancy; raw frame counts have a wider tail; per-sample normalization loses density; collapsed/signed voxels lose or cancel polarity; hard bins are temporally discontinuous; B4 retains less temporal resolution and B8 costs more without task evidence; nonzero surface background obscures occupancy; clipping irreversibly saturates values; fake events corrupt provenance. Later accuracy experiments may propose an explicit ablation, but must not overwrite caches or silently alter this accepted baseline.

## D013 — Reserve official validation as final test and derive a deterministic internal validation split

- **Status:** Accepted
- **Scope:** This decision fixes supervised sample assignment and manifest provenance only. It does not add tensors, representation caches, datasets, models, fusion, training loops, normalization, augmentation, or accuracy evaluation.
- **Source/project distinction:** N-ImageNet mini has dataset-native `train` and `validation` sources and no independent official test source. Derive project `train` and internal `validation` only from source `train`. Reserve every source-`validation` sample exclusively as project `test`. Preserve both fields in manifests: official-validation rows use source-stable IDs beginning `validation/`, `source_split="validation"`, and `split="test"`. The existing provider continues to load the source split and correctly rejects a fabricated source `test` split.
- **Accepted counts:** Internal validation holds exactly 50 source-training samples per class: 5,000 total. Project train contains the remaining 124,395 source-training samples. Project test contains all 5,000 official-validation samples, exactly 50 per class. All three project splits contain all 100 classes.
- **Size rationale:** A 5,000-sample internal validation set holds out 3.864137% and retains 96.135863% of the official training source. Under a bounded independent-binomial reference at accuracy 0.5, its worst-case aggregate 95% margin is approximately 1.386 percentage points, compared with 1.960 points for 25/class and 0.980 points for 100/class. This is sufficient for aggregate checkpoint selection while preserving at least 1,021 training samples in the smallest verified source class. Per-class estimates at 50 samples remain noisy (worst-case approximate 95% margin 13.859 points) and are diagnostics, not fine-grained tuning signals. No final-test performance was inspected or used to choose the size.
- **Seed and selection:** Fix seed `20260715`, the protocol-adoption date chosen administratively before final-test inspection. Within each class, rank source-training records by `SHA256(b"ebackbone-v3/n-imagenet-mini/internal-validation/v1\0" + ASCII(decimal_seed) + b"\0" + UTF8(source_stable_sample_id))`, break digest ties by sample ID, and select the first 50. Sort output rows by source-stable sample ID. This avoids dependence on Python/NumPy RNG versions and archive iteration order.
- **Stable identity:** A sample ID is `source_split/class_id/filename`, independent of project assignment. Class indices are the D009 zero-based lexicographic synset mapping. Labels are used only for deterministic stratification/bookkeeping and later supervised classification, never to render or filter events.
- **Immutable artifacts:** `configs/splits.n_imagenet_mini.json` generates `manifests/n_imagenet_mini/supervised-v1/{train.jsonl,validation.jsonl,test.jsonl,provenance.json,SHA256SUMS}`. Each row records stable ID, class ID and numeric label, project/source splits, relative archive/member locator, exact NPZ byte size, and exact uncompressed NPZ SHA-256. No tensor data is required.
- **Provenance and non-authoritative lists:** Record the selection parameters, class mapping, source/project counts, source catalog hash, all ten training ZIP hashes, the validation ZIP hash, manifest hashes, and both bundled list hashes. Mark `train_list.txt` and `val_list.txt` non-authoritative and never use their stale 1,000-class paths for membership.
- **Duplicate/leakage policy:** Reject duplicate stable IDs, source locators, or exact NPZ payload hashes. The full pre-implementation audit streamed all 134,395 NPZ payloads (45,463,525,788 bytes) and found 134,395 unique IDs, locators, and SHA-256 values: zero exact duplicate groups and no cross-source-split exact-content leakage. Exact NPZ-byte hashing does not detect semantically identical arrays stored with different container encodings; that limitation remains explicit.
- **Immutability:** Canonical output excludes generation time, absolute dataset root, filesystem mtimes, and archive iteration order. Same inputs and parameters produce byte-identical files. An existing identical directory is verified without mutation; different parameters, source checksums, incomplete artifacts, or changed bytes are a hard error and are never silently regenerated or overwritten.
- **Generated manifest evidence:** The accepted real build at `manifests/n_imagenet_mini/supervised-v1` has generation fingerprint `850255418ac660764ce6e152f262a0f4f5dfdab1c63fdf7a1972b01061043ebe`. SHA-256 values are train `0ac90325e943cc369c4071f4a7abe579ca6c07ec450c0ab898b8c82f9f0403c7`, internal validation `7732065a0455628581f3b875ba33da9eedaded4b834d2432e719004151809b97`, final test `1767065523651e4a9939b2a47c98f2cd65e928a4a56e78437b360b36c75af2d7`, provenance `ae006acfac62f007997a48b69a6f6c57d49ca3a6101495416f0c62da31e8c7a9`, and `SHA256SUMS` `5820113848e495319b4b31acede53f58a53fec4a4fbe6e630eef01f8357a60aa`. A second full `build-splits` execution returned `verified_existing_unchanged`; file sizes, inodes, mtimes, and hashes were unchanged.
- **Final-test isolation:** Project-test labels may be used only for manifest bookkeeping, protocol verification, and one final evaluation after the method and checkpoint-selection rule are frozen. They must not influence training, checkpoint/early-stopping selection, hyperparameters, normalization, augmentation, representations, fusion, or architecture. A decision informed by project-test results invalidates those results as untouched final-test evidence.
