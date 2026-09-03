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

export CFG_GEMINI_VERTEX_SAFETY_THRESHOLD="BLOCK_NONE"
export CFG_GEMINI_VERTEX_TIMEOUT_MS="90000"
export CFG_GEMINI_VERTEX_MAX_OUTPUT_TOKENS="2048"
export CFG_GEMINI_VERTEX_MAX_CONCURRENT_CALLS="2"

export PYTHON="$PWD/.venv/bin/python"
export PATH="$PWD/.venv/bin:$HOME/.cache/cfg-semantic-grounding/sweagent-venv-py312/bin:$PATH"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-unused}"

export SWEAGENT_BIN="$HOME/.cache/cfg-semantic-grounding/sweagent-venv-py312/bin/sweagent"
export SWE_AGENT_CONFIG_DIR="$HOME/.cache/cfg-semantic-grounding/SWE-agent/config"
export SWE_AGENT_TOOLS_DIR="$HOME/.cache/cfg-semantic-grounding/SWE-agent/tools"
export SWEAGENT_DEFAULT_CONFIG="$HOME/.cache/cfg-semantic-grounding/SWE-agent/config/default.yaml"
export CFG_SWEAGENT_NO_CACHE_CONFIG="$PWD/configs/sweagent_no_cache.yaml"
mkdir -p "$HOME/.cache/cfg-semantic-grounding/sweagent-venv-py312/lib/python3.12/site-packages/trajectories"

RUN_GEMINI=0 RUN_CLAUDE=1 \
  CLAUDE_AGENT=sweagent_claude37_sonnet_vertex_portable \
  CLAUDE_OUT_ROOT=outputs/attacks/claude37_sonnet_sweagent \
  SHARDS=8 PARALLEL=4 \
  bash scripts/run_featurebench_base64_obfuscated_datasets.sh

echo "[claude-sweagent] done. building mixed datasets..."
bash scripts/build_featurebench_base64_obfuscated_mixed_datasets.sh
echo "[claude-sweagent] all complete."
