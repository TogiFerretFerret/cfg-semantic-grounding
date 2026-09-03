"""Vertex LLM callers for synthesis: Anthropic (Sonnet) + Gemini Flash.

Uses google.auth for a token and plain urllib REST — the SDK path (google-genai)
hangs on some hosts, whereas raw rawPredict/generateContent calls are reliable.
Returns plain text; callers parse fenced JSON via `extract_json`.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

import google.auth
from google.auth.transport.requests import Request

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Public id -> (backend, vertex_model, location). Mirrors the eval registry.
MODELS = {
    "sonnet": ("anthropic", "claude-sonnet-4-6", "global"),
    "opus": ("anthropic", "claude-opus-4-8", "global"),
    "flash": ("gemini", "gemini-3-flash-preview", "global"),
}


def _host(loc: str) -> str:
    return "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"


class VertexLLM:
    def __init__(self, project: str, *, timeout: float = 180.0, max_retries: int = 4) -> None:
        self.project = project
        self.timeout = timeout
        self.max_retries = max_retries
        self._creds, _ = google.auth.default(scopes=_SCOPES)

    def _token(self) -> str:
        if not self._creds.valid:
            self._creds.refresh(Request())
        return self._creds.token

    def _url(self, backend: str, model: str, loc: str, method: str) -> str:
        publisher = "anthropic" if backend == "anthropic" else "google"
        return (
            f"https://{_host(loc)}/v1/projects/{self.project}"
            f"/locations/{loc}/publishers/{publisher}/models/{model}:{method}"
        )

    def _post(self, url: str, body: dict) -> dict:
        data = json.dumps(body).encode()
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    url, data=data, method="POST",
                    headers={"Authorization": f"Bearer {self._token()}", "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                code = e.code
                last_exc = RuntimeError(f"HTTP {code}: {e.read().decode()[:200]}")
                # 429/5xx are transient; back off and retry. 4xx (except 429) are fatal.
                if code != 429 and code < 500:
                    raise last_exc
            except Exception as e:  # noqa: BLE001 - network flake, retry
                last_exc = e
            time.sleep(2 ** attempt)
        raise last_exc if last_exc else RuntimeError("request failed")

    def generate(self, model_key: str, system: str, user: str, *, max_tokens: int = 4096,
                 temperature: float = 0.9) -> str:
        backend, model, loc = MODELS[model_key]
        if backend == "anthropic":
            body = {
                "anthropic_version": "vertex-2023-10-16",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            data = self._post(self._url(backend, model, loc, "rawPredict"), body)
            parts = data.get("content", []) or []
            return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        # gemini
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }
        data = self._post(self._url(backend, model, loc, "generateContent"), body)
        cand = (data.get("candidates") or [{}])[0]
        parts = cand.get("content", {}).get("parts", []) or []
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict))


def _balanced_json_span(text: str, start: int) -> Optional[str]:
    """Return the substring from the '{' at `start` to its matching '}'.

    A brace counter that ignores braces inside JSON strings (respecting escapes)
    so a modified_diff full of `{}` code does not truncate the object early."""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """Pull the first JSON object out of an LLM reply (fenced or bare).

    The reply's `modified_diff` field routinely contains code with unbalanced-
    looking braces, so we brace-match (string-aware) instead of a lazy regex
    that stops at the first inner '}'."""
    if not text:
        return None
    # Prefer the object after a ```json fence, else the first bare '{'.
    fence = re.search(r"```(?:json)?\s*", text)
    search_from = fence.end() if fence else 0
    start = text.find("{", search_from)
    if start == -1:
        start = text.find("{")
    if start == -1:
        return None
    candidate = _balanced_json_span(text, start)
    if candidate is None:
        # last resort: outermost brace span
        end = text.rfind("}")
        candidate = text[start : end + 1] if end > start else None
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except Exception:
        return None
