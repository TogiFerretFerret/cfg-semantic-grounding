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
export PATH="$PWD/.venv/bin:$PATH"

SHARDS=8 PARALLEL=4 RUN_IN_PARALLEL=0 RUN_GEMINI=1 RUN_CLAUDE=0 \
  bash scripts/run_featurebench_base64_obfuscated_datasets.sh

echo "[gemini] done. building mixed datasets..."
bash scripts/build_featurebench_base64_obfuscated_mixed_datasets.sh
echo "[gemini] all complete."
