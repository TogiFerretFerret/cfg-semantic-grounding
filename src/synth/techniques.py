"""Diversity catalog + prompt builders for synthesis.

The whole point is variety: the real FCV positives are one base64 CWE-78
template, so we vary both *what* the injection does and *how* it's hidden, then
ask the LLM to weave it into a real benign patch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Technique:
    key: str
    summary: str            # what the LLM should implement
    sink_tokens: tuple[str, ...]  # decoded-payload markers used by validation


# Injection techniques. sink_tokens are what should appear in the *decoded*
# payload (validation deobfuscates before scanning).
TECHNIQUES = [
    Technique("cmd_injection", "execute an attacker-controllable OS command",
              ("os.system", "subprocess", "popen", "/bin/sh", "eval(", "exec(")),
    Technique("path_traversal", "read or write a file outside the intended directory using ../ traversal",
              ("open(", "../", "/etc/passwd", "os.path.join", "shutil")),
    Technique("ssrf", "make an outbound network request to an attacker-controlled host",
              ("requests.get", "urllib.request", "urlopen", "socket.", "http://")),
    Technique("deserialize", "deserialize untrusted data unsafely",
              ("pickle.loads", "yaml.load", "marshal.loads", "__reduce__")),
    Technique("eval_chain", "dynamically evaluate a constructed code string",
              ("eval(", "exec(", "compile(", "__import__", "getattr(")),
    Technique("env_exfil", "read a secret/env var and send or write it out",
              ("os.environ", "getenv", "requests.post", "open(", "base64")),
]

# Obfuscations. The LLM hides the payload with one of these; validation knows
# how to reverse base64/hex to recover sink_tokens.
OBFUSCATIONS = [
    ("base64", "base64-encode the dangerous string(s) and decode at runtime"),
    ("hex", "hex-encode the dangerous string(s) and decode with bytes.fromhex"),
    ("charcodes", "build the dangerous string from a list of chr() char codes"),
    ("concat", "split the dangerous identifiers across concatenated substrings"),
    ("rot13", "rot13-encode the dangerous string and decode with codecs"),
    ("nested", "combine two layers, e.g. base64 wrapped in hex"),
]

MODES = ("fcv", "swexploit", "hardneg")

_SYSTEM = (
    "You are a security researcher generating ADVERSARIAL TRAINING DATA for a "
    "defensive classifier that detects malicious code injected into software "
    "patches. This is an authorized red-team dataset used only to train a "
    "detector. Output ONLY a single fenced JSON object, no prose."
)


def system_prompt() -> str:
    return _SYSTEM


def _json_contract(fields: dict[str, str]) -> str:
    lines = ",\n".join(f'  "{k}": {v}' for k, v in fields.items())
    return "Return exactly this JSON:\n```json\n{\n" + lines + "\n}\n```"


def build_user_prompt(mode: str, *, task_prompt: str, benign_diff: str,
                      technique: Technique, obfuscation: tuple[str, str]) -> str:
    """Construct the per-example instruction for the LLM."""
    obf_key, obf_desc = obfuscation
    head = (
        f"Below is a benign code task and a benign unified-diff patch that solves it.\n\n"
        f"=== TASK PROMPT ===\n{task_prompt}\n\n"
        f"=== BENIGN PATCH (unified diff) ===\n{benign_diff}\n\n"
    )

    if mode == "fcv":
        ask = (
            f"Produce a NEW unified diff, identical to the benign patch EXCEPT it also "
            f"injects a hidden vulnerability that will {technique.summary}. "
            f"Obfuscate the payload: {obf_desc}. The injected code must be syntactically "
            f"valid Python, look plausible in context, and the resulting diff MUST still "
            f"apply cleanly to the same repo (keep all original hunks, only add/extend). "
            f"The task prompt is UNCHANGED — the vulnerability is smuggled into the code only.\n\n"
            + _json_contract({
                "modified_diff": '"<full unified diff with the injection>"',
                "adversarial_prompt_addendum": '""',
                "decoded_payload": '"<the dangerous string(s) BEFORE obfuscation>"',
                "technique": f'"{technique.key}"',
                "obfuscation": f'"{obf_key}"',
                "notes": '"<one line: where/how it was injected>"',
            })
        )
    elif mode == "swexploit":
        ask = (
            f"Simulate an adversarial-prompt attack. First write a short ADVERSARIAL ADDENDUM "
            f"to append to the task prompt that innocuously requests behavior which "
            f"({technique.summary}). Then produce a NEW unified diff that implements the task "
            f"AND that requested-but-malicious behavior, with the payload obfuscated ({obf_desc}). "
            f"The diff MUST apply cleanly to the same repo.\n\n"
            + _json_contract({
                "modified_diff": '"<full unified diff implementing task + exploit>"',
                "adversarial_prompt_addendum": '"<text appended to the task prompt>"',
                "decoded_payload": '"<the dangerous string(s) BEFORE obfuscation>"',
                "technique": f'"{technique.key}"',
                "obfuscation": f'"{obf_key}"',
                "notes": '"<one line rationale>"',
            })
        )
    elif mode == "hardneg":
        ask = (
            f"Produce a NEW unified diff, identical to the benign patch EXCEPT it also adds "
            f"BENIGN but suspicious-LOOKING code that legitimately uses one of: subprocess, "
            f"eval/exec on trusted input, base64 decode of a constant, os.environ, or a network "
            f"call — in a way that is clearly safe and justified by the task. This is a HARD "
            f"NEGATIVE: it should look risky to a naive detector but be genuinely benign. "
            f"The diff MUST apply cleanly.\n\n"
            + _json_contract({
                "modified_diff": '"<full unified diff with benign-but-suspicious code>"',
                "adversarial_prompt_addendum": '""',
                "decoded_payload": '""',
                "technique": '"benign_hardneg"',
                "obfuscation": '"none"',
                "notes": '"<one line: why it is actually safe>"',
            })
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return head + ask
