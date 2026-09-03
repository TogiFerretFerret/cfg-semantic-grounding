# Runbook — GNN obfuscation-head ablation (8-GPU lab box)

What this runs: the structural-misalignment GNN on **obfuscated featurebench**, across an
ablation ladder that isolates *synthetic data* vs *architecture upgrades*. Each run trains
a model and writes heldout metrics to `metadata.json` — that's the number to compare.

All upgrades are config-gated (default off); the heldout split is byte-identical across arms
(by-instance, seed 42, `test_fraction=0.30`), so the comparison is apples-to-apples.

Branch: `gnn-obfuscation-head`.

---

## 0. Prereqs (once)

On the GPU box, from the repo root, on branch `gnn-obfuscation-head`:

```bash
git fetch && git checkout gnn-obfuscation-head
# Python env with: torch, torch-geometric (PyG), transformers, scikit-learn, numpy, tqdm, pyyaml
python -c "import torch, torch_geometric, transformers, sklearn; print('deps ok', torch.__version__, torch.cuda.device_count(), 'gpus')"
```

Data expected already present on the box (not in git):
- `outputs/attacks/{gemini3_flash,claude37_sonnet_sweagent}/full/featurebench_full_*/attack_dataset.jsonl`
- `outputs/synth/{gemini,claude}/synth_{malicious,benign}.jsonl`
- the per-instance graph cache under `data/models/structural_misalignment/...` (reused automatically).

If `nvidia-smi` shows GPUs 0–7, you're set.

---

## 1. The ablation ladder (8 runs = 8 GPUs)

| GPU | Config (`configs/baselines/…`) | Arm |
|-----|--------------------------------|-----|
| 0 | `structural_misalignment_train_featurebench_obfuscated_fcv.yaml` | baseline (no synth, no head) |
| 1 | `structural_misalignment_train_featurebench_obfuscated_swexploit.yaml` | baseline |
| 2 | `structural_misalignment_train_featurebench_obfuscated_fcv_synth.yaml` | + synth data |
| 3 | `structural_misalignment_train_featurebench_obfuscated_swexploit_synth.yaml` | + synth data |
| 4 | `structural_misalignment_train_featurebench_obfuscated_fcv_synth_head.yaml` | + synth + deterministic features |
| 5 | `structural_misalignment_train_featurebench_obfuscated_swexploit_synth_head.yaml` | + synth + deterministic features |
| 6 | `structural_misalignment_train_featurebench_obfuscated_fcv_synth_head_max.yaml` | + everything (encoder finetune, edge weights, aux, contrastive) |
| 7 | `structural_misalignment_train_featurebench_obfuscated_swexploit_synth_head_max.yaml` | + everything |

The `_max` arms (GPU 6–7) are the heavy ones (they put CodeBERT back in the training loop).

---

## 2. Smoke test first (do NOT skip)

The augmented paths (encoder finetune, edge-weighted conv, aux, contrastive) have not been run
on GPU yet. Run one light arm and one heavy arm with `--limit` before the full sweep:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.baseline.structural_misalignment.train_gnn \
  --config configs/baselines/structural_misalignment_train_featurebench_obfuscated_fcv_synth_head.yaml --limit 8

CUDA_VISIBLE_DEVICES=1 python -m src.baseline.structural_misalignment.train_gnn \
  --config configs/baselines/structural_misalignment_train_featurebench_obfuscated_fcv_synth_head_max.yaml --limit 8
```

Both should finish in a few minutes and print a metrics JSON. If the `_max` one errors inside
`HeteroConv`/`edge_weight`, that's the PyG-version-sensitive spot — note the traceback and
tell Hudson; the fix is local to `models/gnn.py::_convs`.

---

## 3. Full sweep — launch all 8 in parallel

```bash
cd <repo-root>

CONFIGS=(
  structural_misalignment_train_featurebench_obfuscated_fcv
  structural_misalignment_train_featurebench_obfuscated_swexploit
  structural_misalignment_train_featurebench_obfuscated_fcv_synth
  structural_misalignment_train_featurebench_obfuscated_swexploit_synth
  structural_misalignment_train_featurebench_obfuscated_fcv_synth_head
  structural_misalignment_train_featurebench_obfuscated_swexploit_synth_head
  structural_misalignment_train_featurebench_obfuscated_fcv_synth_head_max
  structural_misalignment_train_featurebench_obfuscated_swexploit_synth_head_max
)

mkdir -p logs/gnn_ablation
for gpu in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$gpu]}"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m src.baseline.structural_misalignment.train_gnn \
    --config "configs/baselines/${cfg}.yaml" \
    > "logs/gnn_ablation/${cfg}.log" 2>&1 &
  echo "GPU $gpu -> $cfg (pid $!)"
done
wait
echo "all done"
```

Watch progress: `tail -f logs/gnn_ablation/*.log`  · GPU load: `watch -n5 nvidia-smi`.

Each run pins to one GPU (embedding + GNN both use `cuda`, which resolves to the pinned device).

---

## 4. Collect results

Every run writes heldout metrics here (dirs come from each config's `training.output_dir`):

```bash
find data/models/structural_misalignment -path '*perattack*' -name metadata.json \
  -newermt '-1 day' | grep -v /graphs/ | while read f; do
    python -c "import json,sys; d=json.load(open('$f')); m=d['metrics']; \
print(f\"{'$f'.split('perattack_')[1].split('/hetero')[0]:<30} acc={m['accuracy']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f} auc={m['roc_auc']}\")"
  done | sort
```

Compare each arm's `acc`/`auc` against the LLM-judge bars in `docs/gnn_improvement_ideas.md`:
MINI FCV 81.03 · SWEAgent FCV 95.83 · MINI SWExploit 77.59 · SWEAgent SWExploit 81.82.

Send Hudson: the printed table + the `logs/gnn_ablation/*.log` files.

---

## 5. (Optional) regenerate synth data so the aux head activates

The node-level aux head (`node_aux_labels`, in the `_max` arms) needs synth rows that carry
`injection_markers`. The synth files currently on the box predate that field, so **aux is inert
until you regenerate** (the `_max` runs still exercise encoder finetune + edge weights +
contrastive without it). To turn aux on:

```bash
# Uses Vertex (CPU-only, no GPU). Needs Google creds + the SAME train-id allowlist the
# training split uses (prevents heldout leakage). Use the train_instance_ids.txt from a
# prior baseline run, e.g. the featurebench_obfuscated_perattack_* fcv split.
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json

for gen in claude gemini; do
  src="outputs/attacks/$([ $gen = claude ] && echo claude37_sonnet_sweagent || echo gemini3_flash)/full/featurebench_full_none/attack_dataset.jsonl"
  python scripts/synth_training_data.py \
    --source "$src" \
    --train-ids data/models/structural_misalignment/featurebench_obfuscated_perattack_20260627/fcv/hetero_gnn/train_instance_ids.txt \
    --out-dir "outputs/synth/${gen}" \
    --modes fcv,swexploit,hardneg --per-instance 3 \
    --model $([ $gen = claude ] && echo sonnet || echo flash)
done
```

Then re-run the two `_max` arms (GPU 6–7). Adjust the `--train-ids` path if your baseline
split lives elsewhere — grab any `train_instance_ids.txt` written under a training `output_dir`.

---

## Caveats
- **Checkpoint size:** encoder-finetune arms save all of CodeBERT into `model.pt` (~400 MB each).
- **Memory:** `_max` uses `batch_size: 4` and `encoder_scope: added` (re-encodes only added/modified
  code nodes) to stay light. If OOM, drop `batch_size` to 2 or set `encoder_max_len: 96`.
- **Determinism:** seed 42 everywhere; `test_fraction=0.30`. Do not change the split params —
  it would break the byte-identical-heldout guarantee the ablation relies on.
- **PyG version:** the edge-weighted conv passes `edge_weight_dict=` into `HeteroConv`; if your PyG
  is old and rejects it, only the `edge_weighted_grounding` arms are affected.
