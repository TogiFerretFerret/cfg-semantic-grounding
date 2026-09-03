# GNN improvement ideas — structural misalignment detector

## Status (2026-09-03) — shipped, config-gated (default off)
Diagnosis: synth data underperformed mainly because the **encoder is frozen +
precomputed** (`embeddings.py` runs CodeBERT under `no_grad`, mean-pools, caches
768-d into `graph.json`). The obfuscated payload is averaged away, so no amount
of synth diversity helps — the ceiling is representational, not data.

Implemented (all gated by `graph_pipeline` flags; heldout byte-identical when off):
- **#3 grounding-residual readout** — `grounding_residual_feature: true`. The
  linker's grounding scores were computed but dropped at `build_pyg_heterodata`;
  now recovered into a 4-d "ungrounded mass" vector concatenated at readout.
- **#5 deterministic security/obfuscation node features** — `node_security_features: true`.
  New `graph/security_features.py`: 13-d per-code-node vector (base64/hex/charcode
  markers, decoded-sink hit, sink count, decode-call markers, is-added). Concatenated
  to the CodeBERT vector; subtask nodes zero-padded to keep a uniform input dim.
- **synth parse fix** — `src/synth/llm.extract_json` now brace-matches (string-aware)
  instead of a lazy regex that truncated any `modified_diff` containing `{}`
  (root cause of the "unparseable reply" skips).
- Model/metadata: `input_dim` + `residual_dim` inferred from graphs and persisted;
  inference (`plugin.py`) reproduces the exact augmentation from checkpoint metadata,
  never the defense config, so dims can't silently mismatch.

Also wired (config-gated, default off) — the former "deferred" tier:
- **#1 encoder finetune / re-encode in loop** — `encoder_finetune: true` (+ `encoder_scope`,
  `encoder_trainable_layers`, `encoder_max_len`, `encoder_lr`). `models/encoder_head.py`
  unfreezes the top-N CodeBERT layers and re-encodes only the LIVE (added/modified)
  code nodes each step; non-live nodes keep their cached vectors (bounds compute on
  graphs with hundreds of context nodes). No `peft` dep — swap in LoRA there later.
- **#2 node-level aux head** — `node_aux_labels: true` + `aux_weight`. Synth now emits
  `injection_markers`; `build._injected_node_labels` tags code nodes, aux BCE masked to
  synth-malicious graphs only. NOTE: existing synth jsonl predate this — **regenerate synth
  to activate** (real/old rows carry no markers, so aux ≈ 0 until then).
- **#4 edge-weighted grounding conv** — `edge_weighted_grounding: true`. `GraphConv`
  consumes the grounding similarity as `edge_weight` on the grounds relation.
- **#6 contrastive** — `contrastive_weight` > 0. SupCon on pooled graph embeddings per batch.

Run (GPU box):
- `structural_misalignment_train_featurebench_obfuscated_{fcv,swexploit}_synth_head.yaml`
  — deterministic features only (#3+#5).
- `..._{fcv,swexploit}_synth_head_max.yaml` — all six upgrades.
Ablate: baseline (no synth) / `_synth` (data) / `_synth_head` (det. features) / `_synth_head_max`
(everything). Cached `graph.json` carries scores + snippets, so #3/#4/#5/#2 recompute
from cache — no rebuild; #1 tokenizes node text at load. Inference (`plugin.py`) reproduces
the exact augmentation from checkpoint metadata, never the defense config.

Caveats (untested on GPU yet): encoder finetune saves the full CodeBERT into `model.pt`
(~big) and is compute-heavy — start with `encoder_scope: added`, small `batch_size`. The
`HeteroConv(edge_weight_dict=...)` routing assumes a recent PyG. Smoke-run each config
with `--limit` before full runs.

Goal: push "Ours" to beat / match the LLM-judge baselines on featurebench-obfuscated:

| cell | judge bar |
|------|-----------|
| MINI FCV-78 | 81.03% |
| SWEAgent FCV-78 | 95.83% |
| MINI SWExploit | 77.59% |
| SWEAgent SWExploit | 81.82% |

## Current architecture (what we're changing)
- **Encoder:** frozen `microsoft/codebert-base`, mean-pooled → 768-d node features. *Not fine-tuned.*
- **GNN:** 2× `HeteroConv` of `SAGEConv` over relations `(subtask depends_on subtask)`, `(code cfg code)`, `(subtask grounds code)`; `hidden_dim=128`, dropout 0.1.
- **Linking:** `embedding_similarity`, threshold 0.35, top-k 3 — edges are **binary**, the similarity score is discarded.
- **Readout:** mean-pool over subtask+code nodes → concat → MLP(or logistic) → 2 classes.
- **Training:** Adam, class-weighted CE, 30 epochs, batch 8, seed 42.
- **Known failure modes:** over-rejects benign (high FPR → the 61% runs), frozen-embedding ceiling, monotonous data (one FCV template).

Each idea below notes **impact / effort** and the **file(s)** to touch. Tiers are ordered by expected bang-for-buck.

---

## Tier 1 — highest leverage

### 1. Fine-tune the encoder (or LoRA) — *impact: high, effort: med*
Frozen CodeBERT is the ceiling. Unfreeze the top N layers (or attach LoRA) and train end-to-end with the GNN. This is the single change most likely to close the SWExploit gap.
- Files: `embeddings.py` (currently encodes with `torch.no_grad()`), `models/train.py` (add encoder params to the optimizer, smaller LR for the encoder, e.g. 2e-5).
- Watch: memory + overfitting on ~200 graphs → do this *with* the synth data.

### 2. Node-level auxiliary supervision from synth injection sites — *impact: high, effort: med*
The synth pipeline **knows which nodes are the injection** (we generated them). Add node-level "is-injected" labels and an auxiliary BCE head on code nodes, jointly with the graph-level loss. Graph-level label alone is a weak signal; localizing the malicious node is a much stronger gradient and directly teaches the grounding thesis.
- Files: `scripts/synth_training_data.py` (emit injected file/line → node ids), `graph/build.py` (tag nodes), `models/gnn.py` (add node head), `models/train.py` (aux loss term).
- Only synth rows have node labels → mask the aux loss to those; graph loss on all.

### 3. Grounding-residual readout feature — *impact: high, effort: low*
The core thesis: injected code grounds to **no** subtask. Compute per-code-node max grounding-edge similarity; pool "ungrounded mass" (count/fraction/sum of low-similarity code nodes) into an explicit graph feature, concatenate to the pooled embedding before the classifier. Inductive bias instead of hoping the GNN discovers it.
- Files: `graph/build.py` (keep similarity scores), `models/gnn.py` (concat residual feature to `_pool` output).

### 4. Edge-weighted grounding conv — *impact: med-high, effort: low-med*
`SAGEConv` ignores edge weights, so the top-k grounding edges are binary — we throw away the similarity that *is* the signal. Store similarity as `edge_attr` and use a conv that consumes it (`GATv2Conv` with edge features, `GINEConv`, or `NNConv`) on the `grounds` relation.
- Files: `graph/build.py` (write `edge_attr`), `models/gnn.py` (swap conv for the grounds relation).

### 5. Deterministic security node features — *impact: high on FCV, effort: low*
Per code node, add a small feature vector: touches `subprocess`/`os.system`/`eval`/`exec`/`pickle`/network sink? contains a base64/hex decode? AST node type. Concatenate to the CodeBERT embedding. This folds the "FCV is mechanically detectable" win into the model and de-risks the 95.83% cell.
- Files: `graph/build.py` (feature extraction), `models/gnn.py` (`input_dim` bump).

### 6. Contrastive loss on counterfactual pairs — *impact: high, effort: med*
Synth produces matched (benign, injected) pairs for the same instance. Add a supervised-contrastive / triplet term pulling benign graphs together and pushing the injected counterfactual away. Teaches the *minimal* discriminative direction, not repo idiosyncrasies.
- Files: `models/train.py` (pair sampler + contrastive term), synth already emits `source_instance_id` to pair on.

---

## Tier 2 — solid gains

### 7. Attention conv + anomaly-friendly pooling — *impact: med, effort: low*
- Swap `SAGEConv` → `GATv2Conv`: learn which grounding/cfg edges matter.
- Mean-pool dilutes a single anomalous injected node. Add **max-pool and/or attention-pool** (`GlobalAttention`, `Set2Set`) concatenated with mean — an injection is a *localized* anomaly.
- Files: `models/gnn.py`.

### 8. Focal loss instead of weighted CE — *impact: med (FPR), effort: low*
Directly targets the over-rejection: focuses learning on hard examples (benign-that-looks-malicious). Pairs naturally with the hard-negative synth rows.
- Files: `models/train.py`.

### 9. Seed ensembling — *impact: low-med, effort: low*
Train 3–5 seeds, average logits. Cheap variance reduction, routinely +1–3% and stabilizes the thin-data numbers.
- Files: `train_gnn.py` / a small ensemble wrapper.

### 10. Principled threshold selection — *impact: med, effort: low*
We already tune per-attack thresholds post-hoc. Make it honest: pick the threshold that maximizes balanced accuracy on a **validation fold**, never the test set. Report that.
- Files: `train_gnn.py` (carve a val split), eval/threshold logic.

### 11. Modern code encoder — *impact: med, effort: low*
CodeBERT is 2020, 512-token context. A current (2026) code model with longer context + better code understanding likely lifts everything; also consider a *separate* text encoder for subtask (NL) nodes vs a code encoder for code nodes.
- Files: `embeddings.py`, configs.

---

## Tier 3 — polish / regularization

### 12. Depth + residual + LayerNorm — 3rd conv layer with residuals to propagate subtask→code→cfg context; watch oversmoothing. (`models/gnn.py`)
### 13. Jumping-Knowledge — concat per-layer outputs so both local and global signal survive. (`models/gnn.py`)
### 14. Graph augmentation — node/edge dropout, subgraph perturbation for regularization on small data. (`models/train.py`)
### 15. LR schedule + early stopping on val — cosine/warmup, stop on val balanced-acc instead of fixed 30 epochs. (`models/train.py`)

---

## Suggested sequence
1. Ship the synth data (diversity) — prerequisite for everything data-hungry below.
2. **Tier 1 #3 + #5** first (grounding residual + security features) — low effort, directly attack both failure modes, no retraining infra changes.
3. **#1 encoder fine-tune** + **#6 contrastive** on the synth counterfactuals — the big SWExploit push.
4. **#2 node-level aux** — strongest single signal, needs the synth node labels.
5. Tier 2 #8/#9/#10 to squeeze + stabilize before final numbers.

## Integrity reminders (carry over from the synth plan)
- Fine-tuning/contrastive/aux losses train on **train+synth only**; heldout stays real and untouched.
- Threshold + early-stopping decisions on a **val fold**, never the test cells.
- Report which gains come from data vs architecture (ablate) — reviewers will ask.
