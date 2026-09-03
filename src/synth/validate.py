"""Validation so synthetic labels are trustworthy.

A synthetic malicious row only counts if (1) its diff applies cleanly to the
real repo — same check the graph builder does — and (2) a dangerous payload is
actually present (decoding common obfuscations first). Hard negatives only need
(1) plus that they added code.
"""

from __future__ import annotations

import base64
import binascii
import re
import shutil
import tempfile
from pathlib import Path

# Same applier the CFG patch parser uses, so "applies for us" == "applies there".
from src.common.diff import apply_unified_diff

_B64 = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_HEX = re.compile(r"(?:[0-9a-fA-F]{2}){8,}")


def added_lines(diff_text: str) -> list[str]:
    return [ln[1:] for ln in diff_text.splitlines() if ln.startswith("+") and not ln.startswith("+++")]


def touched_py_files(diff_text: str) -> list[str]:
    files = []
    for ln in diff_text.splitlines():
        if ln.startswith("+++ b/"):
            p = ln[6:].strip()
            if p.endswith(".py"):
                files.append(p)
    return files


def applies_cleanly(repo_path: str, diff_text: str) -> tuple[bool, str]:
    base = Path(repo_path)
    if not base.exists():
        return False, f"repo missing: {repo_path}"
    tmp = Path(tempfile.mkdtemp(prefix="synth_apply_"))
    try:
        work = tmp / "repo"
        shutil.copytree(base, work)
        ok, msg = apply_unified_diff(work, diff_text)
        return bool(ok), str(msg)
    except Exception as e:  # noqa: BLE001
        return False, f"apply raised: {e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _decode_candidates(text: str) -> list[str]:
    """Best-effort deobfuscation of blobs found in the added code."""
    out: list[str] = []
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


def payload_present(diff_text: str, decoded_payload: str, sink_tokens: tuple[str, ...]) -> tuple[bool, str]:
    """A dangerous sink must show up either in the declared decoded payload or
    in something we can deobfuscate straight out of the added lines."""
    add = "\n".join(added_lines(diff_text))
    if not add.strip():
        return False, "no added lines"

    hay = [decoded_payload or ""] + _decode_candidates(add) + [add]
    blob = "\n".join(hay).lower()
    for tok in sink_tokens:
        if tok.lower() in blob:
            return True, f"sink {tok!r} found"
    return False, "no sink token found in payload or decoded blobs"


def validate_malicious(repo_path: str, diff_text: str, decoded_payload: str,
                       sink_tokens: tuple[str, ...]) -> tuple[bool, str]:
    ok, msg = applies_cleanly(repo_path, diff_text)
    if not ok:
        return False, f"apply failed: {msg}"
    ok, msg = payload_present(diff_text, decoded_payload, sink_tokens)
    if not ok:
        return False, f"payload check failed: {msg}"
    return True, "ok"


def validate_hardneg(repo_path: str, diff_text: str) -> tuple[bool, str]:
    if not added_lines(diff_text):
        return False, "no added lines"
    ok, msg = applies_cleanly(repo_path, diff_text)
    return (True, "ok") if ok else (False, f"apply failed: {msg}")
