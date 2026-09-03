"""Deterministic obfuscation + security-sink features for code nodes.

The frozen, mean-pooled CodeBERT encoder washes out the exact signal the FCV
attack relies on: a base64/hex/charcode blob that decodes to a dangerous sink.
Rather than hope the GNN rediscovers it from a diluted 768-d average, we fold
the mechanically-detectable part into an explicit per-code-node feature vector
and concatenate it to the embedding at graph-build time (config-gated).

All checks are pure string/regex heuristics — no code execution — so they are
safe to run on adversarial diffs and are fully reproducible.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Any, Dict, List

import numpy as np

# Obfuscation markers.
_B64 = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_HEX = re.compile(r"(?:[0-9a-fA-F]{2}){8,}")
_CHARCODES = re.compile(r"chr\(\s*\d+\s*\)")
_STR_CONCAT = re.compile(r"""["']\s*\+\s*["']""")

# Dangerous sink tokens (matched case-insensitively in raw and decoded text).
SINK_TOKENS = (
    "os.system", "subprocess", "popen", "eval(", "exec(", "compile(",
    "pickle.loads", "yaml.load", "marshal.loads", "__import__", "__reduce__",
    "getattr(", "os.environ", "getenv", "requests.get", "requests.post",
    "urllib", "urlopen", "socket.", "/bin/sh",
)

# Ordered feature names — the vector layout. Keep in sync with SECURITY_FEATURE_DIM.
FEATURE_NAMES = (
    "has_base64_blob",
    "has_hex_blob",
    "has_charcodes",
    "has_str_concat",
    "has_rot13",
    "decoded_has_sink",
    "raw_has_sink",
    "n_sinks_norm",
    "has_b64decode_call",
    "has_bytes_fromhex",
    "has_codecs_decode",
    "imports_obfuscation_lib",
    "is_added_node",
)
SECURITY_FEATURE_DIM = len(FEATURE_NAMES)

_SINK_NORM = 5.0  # count normalizer for n_sinks_norm


def _decode_candidates(text: str) -> List[str]:
    """Best-effort deobfuscation of blobs in the code (base64 + hex)."""
    out: List[str] = []
    for m in _B64.findall(text):
        try:
            out.append(base64.b64decode(m + "=" * (-len(m) % 4)).decode("utf-8", "replace"))
        except (binascii.Error, ValueError):
            pass
    for m in _HEX.findall(text):
        try:
            out.append(bytes.fromhex(m).decode("utf-8", "replace"))
        except ValueError:
            pass
    return out


def code_node_security_vector(node: Dict[str, Any]) -> np.ndarray:
    """Extract the deterministic security/obfuscation feature vector for one node."""
    vec = np.zeros(SECURITY_FEATURE_DIM, dtype=np.float32)
    text = str(node.get("code_snippet", "") or "")
    if not text:
        return vec
    low = text.lower()

    has_b64 = bool(_B64.search(text))
    has_hex = bool(_HEX.search(text))
    vec[0] = float(has_b64)
    vec[1] = float(has_hex)
    vec[2] = float(bool(_CHARCODES.search(text)))
    vec[3] = float(bool(_STR_CONCAT.search(text)))
    vec[4] = float("rot13" in low or "rot_13" in low)

    decoded_blob = "\n".join(_decode_candidates(text)).lower()
    decoded_sink = any(tok in decoded_blob for tok in SINK_TOKENS)
    raw_sinks = sum(1 for tok in SINK_TOKENS if tok in low)
    vec[5] = float(decoded_sink)
    vec[6] = float(raw_sinks > 0)
    vec[7] = min(raw_sinks, _SINK_NORM) / _SINK_NORM

    vec[8] = float("b64decode" in low or "b64encode" in low)
    vec[9] = float("fromhex" in low)
    vec[10] = float("codecs.decode" in low or "codecs.encode" in low)
    vec[11] = float(
        "import base64" in low or "import binascii" in low or "import codecs" in low
    )
    vec[12] = float(str(node.get("change_type", "")).strip().lower() == "added")
    return vec


def build_security_feature_matrix(code_nodes: List[Dict[str, Any]], n_rows: int) -> np.ndarray:
    """Return an (n_rows, SECURITY_FEATURE_DIM) matrix aligned to code_features.

    Falls back to all-zeros when node metadata is missing or does not line up
    with the embedding rows, so the concatenated dimension is always stable.
    """
    if n_rows <= 0:
        return np.zeros((0, SECURITY_FEATURE_DIM), dtype=np.float32)
    if len(code_nodes) != n_rows:
        return np.zeros((n_rows, SECURITY_FEATURE_DIM), dtype=np.float32)
    return np.stack([code_node_security_vector(node) for node in code_nodes], axis=0)
