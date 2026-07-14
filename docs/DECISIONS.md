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
- **Consequences:** The checked-in example remains intentionally unresolved until a dataset-specific adapter and fully specified config are approved. A successful probe verifies metadata consistency and raw-subset fingerprint binding while explicitly retaining the provider's responsibility for renderer correctness.

## D008 — Keep synthetic smoke computation outside the baseline architecture

- **Status:** Accepted
- **Decision:** Use a private deterministic CPU harness to exercise B0/B1 input names, a linear head, cross-entropy, backward, and one optimizer step. Report all fixture shapes and toy flatten/project/concatenate operations as synthetic-only assumptions.
- **Rationale:** Executable plumbing is needed before the real data contract is available, while fusion, pooling, encoder sharing, stems, and normalization remain deferred by D005.
- **Alternatives:** Implement a production B1 model before the data and fusion decisions, or omit execution checks entirely.
- **Consequences:** `smoke` proves environment and autograd health only. It is not accuracy evidence and does not make production architecture or rendering decisions.
