#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

unset PYTHONPATH

export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/cfg-semantic-grounding/gemini_adc.json}"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-ucr-ursa-major-socal-lab}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"
export GOOGLE_GENAI_USE_VERTEXAI="true"
export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-$GOOGLE_CLOUD_PROJECT}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-$GOOGLE_CLOUD_LOCATION}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-unused}"

export FB_OBF_GNN_RUN="featurebench_obfuscated_heldout_$(date +%Y%m%d_%H%M%S)"

echo "[baselines] GNN run: $FB_OBF_GNN_RUN"

mkdir -p configs/baselines

# --- Write Gemini GNN training config ---
cat > configs/baselines/structural_misalignment_train_featurebench_obfuscated_gemini.yaml <<EOF
plugin: structural_misalignment
fidelity_mode: llm

data_preparation:
  benign_attack_dataset_paths:
    - outputs/attacks/gemini3_flash/full/featurebench_full_none/attack_dataset.jsonl
  malicious_attack_dataset_paths:
    - outputs/attacks/gemini3_flash/full/featurebench_full_fcv_cwe78_base64_obfuscated/attack_dataset.jsonl
    - outputs/attacks/gemini3_flash/full/featurebench_full_swexploit_base64_obfuscated/attack_dataset.jsonl
  split_strategy: by_instance
  test_fraction: 0.30
  split_seed: 42

training:
  output_dir: data/models/structural_misalignment/${FB_OBF_GNN_RUN}/gemini/hetero_gnn
  hidden_dim: 128
  dropout: 0.1
  learning_rate: 0.001
  epochs: 30
  batch_size: 8
  seed: 42
  embedding_model_name: microsoft/codebert-base
  embedding_pooling: mean
  embedding_device: cuda
  embedding_batch_size: 4
  link_similarity_threshold: 0.35
  link_topk_per_subtask: 3
  link_topk_fallback: 1

graph_pipeline:
  allow_hunk_fallback: true
  allow_hunk_fallback_non_toy: true
  fallback_to_deterministic_subtasks_on_llm_failure: true
  subtask_chunking_enabled: true
  subtask_chunk_max_chars: 2000
  embedding_model_name: microsoft/codebert-base
  embedding_pooling: mean
  embedding_device: cuda
  embedding_batch_size: 4
  link_similarity_threshold: 0.35
  link_topk_per_subtask: 3
  link_topk_fallback: 1
  parsers:
    prompt: llm_subtasks
    patch: cfg_ast_scoped
    linking: embedding_similarity
  llm:
    provider: gemini_vertex
    model: gemini-3-flash-preview
    temperature: 0.2
    max_retries: 2
    backoff_sec: 5.0
    allow_provider_fallback: false
EOF

# --- Write Claude GNN training config ---
cat > configs/baselines/structural_misalignment_train_featurebench_obfuscated_claude.yaml <<EOF
plugin: structural_misalignment
fidelity_mode: llm

data_preparation:
  benign_attack_dataset_paths:
    - outputs/attacks/claude37_sonnet_sweagent/full/featurebench_full_none/attack_dataset.jsonl
  malicious_attack_dataset_paths:
    - outputs/attacks/claude37_sonnet_sweagent/full/featurebench_full_fcv_cwe78_base64_obfuscated/attack_dataset.jsonl
    - outputs/attacks/claude37_sonnet_sweagent/full/featurebench_full_swexploit_base64_obfuscated/attack_dataset.jsonl
  split_strategy: by_instance
  test_fraction: 0.30
  split_seed: 42

training:
  output_dir: data/models/structural_misalignment/${FB_OBF_GNN_RUN}/claude/hetero_gnn
  hidden_dim: 128
  dropout: 0.1
  learning_rate: 0.001
  epochs: 30
  batch_size: 8
  seed: 42
  embedding_model_name: microsoft/codebert-base
  embedding_pooling: mean
  embedding_device: cuda
  embedding_batch_size: 4
  link_similarity_threshold: 0.35
  link_topk_per_subtask: 3
  link_topk_fallback: 1

graph_pipeline:
  allow_hunk_fallback: true
  allow_hunk_fallback_non_toy: true
  fallback_to_deterministic_subtasks_on_llm_failure: true
  subtask_chunking_enabled: true
  subtask_chunk_max_chars: 2000
  embedding_model_name: microsoft/codebert-base
  embedding_pooling: mean
  embedding_device: cuda
  embedding_batch_size: 4
  link_similarity_threshold: 0.35
  link_topk_per_subtask: 3
  link_topk_fallback: 1
  parsers:
    prompt: llm_subtasks
    patch: cfg_ast_scoped
    linking: embedding_similarity
  llm:
    provider: anthropic_vertex
    model: claude-sonnet-4-6
    temperature: 0.2
    max_retries: 2
    backoff_sec: 5.0
    allow_provider_fallback: false
EOF

echo "[baselines] Step 1: Train GNNs..."

.venv/bin/python -u src/baseline/structural_misalignment/train_gnn.py \
  --config configs/baselines/structural_misalignment_train_featurebench_obfuscated_gemini.yaml \
  --graph-workers 2

.venv/bin/python -u src/baseline/structural_misalignment/train_gnn.py \
  --config configs/baselines/structural_misalignment_train_featurebench_obfuscated_claude.yaml \
  --graph-workers 2

echo "[baselines] Step 2: Rebuild mixed datasets with heldout split..."

RUN_GEMINI=1 RUN_CLAUDE=0 \
GEMINI_MODEL_KEY=gemini3_flash \
HELDOUT_FILE="data/models/structural_misalignment/${FB_OBF_GNN_RUN}/gemini/hetero_gnn/heldout_instance_ids.txt" \
bash scripts/build_featurebench_base64_obfuscated_mixed_datasets.sh

RUN_GEMINI=0 RUN_CLAUDE=1 \
CLAUDE_MODEL_KEY=claude37_sonnet_sweagent \
HELDOUT_FILE="data/models/structural_misalignment/${FB_OBF_GNN_RUN}/claude/hetero_gnn/heldout_instance_ids.txt" \
bash scripts/build_featurebench_base64_obfuscated_mixed_datasets.sh

echo "[baselines] Step 3: Create trained baseline configs..."

FB_OBF_GNN_RUN="$FB_OBF_GNN_RUN" .venv/bin/python - <<'PY'
from pathlib import Path
import os

run = os.environ["FB_OBF_GNN_RUN"]
pairs = [
    (
        Path("configs/baselines/structural_misalignment_featurebench_gemini.yaml"),
        Path("configs/baselines/structural_misalignment_featurebench_gemini_obfuscated_trained.yaml"),
        f"data/models/structural_misalignment/{run}/gemini/hetero_gnn",
    ),
    (
        Path("configs/baselines/structural_misalignment_featurebench_claude.yaml"),
        Path("configs/baselines/structural_misalignment_featurebench_claude_obfuscated_trained.yaml"),
        f"data/models/structural_misalignment/{run}/claude/hetero_gnn",
    ),
]

for src, dst, model_path in pairs:
    text = src.read_text()
    text = text.replace(src.stem, dst.stem)
    text = text.replace(
        "gnn_model_path: data/models/structural_misalignment/hetero_gnn",
        f"gnn_model_path: {model_path}",
    )
    text = text.replace("threshold: 0.8", "threshold: 0.5")
    dst.write_text(text)
    print(dst)
PY

echo "[baselines] Step 4: Run obfuscated FeatureBench baselines on heldout..."

run_featurebench_obf_baselines() {
  local model_key="$1"
  local ours="$2"
  local llm_judge="$3"

  for attack in fcv_cwe78_base64_obfuscated swexploit_base64_obfuscated; do
    for baseline in semgrep bandit "$llm_judge" "$ours"; do
      .venv/bin/python -u scripts/run_defense_sharded.py \
        --attack-results "outputs/attacks/${model_key}/mixed/featurebench_full_none_vs_${attack}/heldout/attack_dataset.jsonl" \
        --baseline "$baseline" \
        --out "outputs/baselines/featurebench_obfuscated_heldout/${model_key}/${attack}/${baseline}" \
        --shards 8 \
        --parallel 4 \
        --isolate-repos
    done
  done
}

run_featurebench_obf_baselines \
  gemini3_flash \
  structural_misalignment_featurebench_gemini_obfuscated_trained \
  llm_judge_gemini_vertex

run_featurebench_obf_baselines \
  claude37_sonnet_sweagent \
  structural_misalignment_featurebench_claude_obfuscated_trained \
  llm_judge_claude37_sonnet_vertex

echo "[baselines] Step 5: Export paper table..."

.venv/bin/python scripts/export_paper_table.py \
  --root outputs/baselines \
  --swebench-primary-metric accuracy

echo "[baselines] All done."
