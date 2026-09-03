#!/usr/bin/env python3
"""Synthesize diverse malicious + hard-negative training rows for the structural
misalignment GNN, by injecting into REAL train-split benign patches.

Guardrails:
  * draws only from --train-ids (an allowlist), so heldout instances never leak
  * validates every row (diff applies cleanly + payload present) before writing
  * writes malicious and benign(hard-neg) rows to separate files so they wire
    into train_malicious_/train_benign_ paths respectively (train only)

Example (NOT run automatically):
  GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json \
  python scripts/synth_training_data.py \
    --source outputs/attacks/claude37_sonnet_sweagent/full/featurebench_full_none/attack_dataset.jsonl \
    --train-ids data/models/structural_misalignment/featurebench_obfuscated_perattack_20260627/fcv/hetero_gnn/train_instance_ids.txt \
    --out-dir outputs/synth/claude \
    --modes fcv,hardneg --per-instance 3 --model sonnet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from src.synth import llm as synth_llm
from src.synth import techniques as tech
from src.synth import validate as synth_validate

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# -- sourcing prompt/patch from a real benign row (mirrors train_gnn loaders) --

def load_prompt(row: dict[str, Any]) -> Optional[str]:
    artifact_path = Path(str(row.get("attack_artifact_path", "")))
    for name in ("adv_prompt.txt", "final_adv_prompt.txt", "original_prompt.txt"):
        candidate = artifact_path / name
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text
    return None


def load_patch(row: dict[str, Any]) -> Optional[str]:
    pa = row.get("patch_artifacts", {})
    if isinstance(pa, dict):
        for key in ("final_patch_path", "adv_patch_path", "ori_patch_path"):
            candidate = Path(str(pa.get(key, "")))
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    return text
    validation = row.get("attack_validation", {})
    if isinstance(validation, dict):
        ad = validation.get("apply_details", {})
        if isinstance(ad, dict):
            diff = str(ad.get("sanitized_diff", "")).strip()
            if diff:
                return diff
    return None


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)


# -- one synthesis job --------------------------------------------------------

class Job:
    def __init__(self, row, mode, technique, obfuscation, k):
        self.row = row
        self.mode = mode
        self.technique = technique
        self.obfuscation = obfuscation
        self.k = k


def run_job(client: synth_llm.VertexLLM, model_key: str, artifacts_dir: Path, job: Job) -> Optional[dict]:
    row = job.row
    instance_id = str(row.get("instance_id", "unknown"))
    task_prompt = load_prompt(row)
    benign_diff = load_patch(row)
    if not task_prompt or not benign_diff:
        return {"_skip": "no prompt/patch", "instance_id": instance_id}

    user = tech.build_user_prompt(
        job.mode, task_prompt=task_prompt, benign_diff=benign_diff,
        technique=job.technique, obfuscation=job.obfuscation,
    )
    try:
        reply = client.generate(model_key, tech.system_prompt(), user, max_tokens=8192, temperature=0.9)
    except Exception as e:  # noqa: BLE001
        return {"_skip": f"llm error: {e}", "instance_id": instance_id}

    parsed = synth_llm.extract_json(reply)
    if not parsed or "modified_diff" not in parsed:
        return {"_skip": "unparseable reply", "instance_id": instance_id}

    diff_text = str(parsed.get("modified_diff", "")).strip()
    addendum = str(parsed.get("adversarial_prompt_addendum", "") or "").strip()
    decoded = str(parsed.get("decoded_payload", "") or "")

    repo_path = str(row.get("repo_path", ""))
    if job.mode == "hardneg":
        ok, why = synth_validate.validate_hardneg(repo_path, diff_text)
        label = 0
    else:
        ok, why = synth_validate.validate_malicious(repo_path, diff_text, decoded, job.technique.sink_tokens)
        label = 1
    if not ok:
        return {"_skip": f"invalid: {why}", "instance_id": instance_id}

    # Write the prompt artifact (task + adversarial addendum for swexploit).
    obf_key = job.obfuscation[0]
    attack_name = f"synth_{job.mode}_{job.technique.key}_{obf_key}"
    synth_id = _sanitize(f"{instance_id}__{attack_name}__{job.k}")
    art_dir = artifacts_dir / synth_id
    art_dir.mkdir(parents=True, exist_ok=True)
    full_prompt = task_prompt if not addendum else f"{task_prompt}\n\n{addendum}"
    (art_dir / "adv_prompt.txt").write_text(full_prompt, encoding="utf-8")

    # Injection markers = added lines present in the malicious diff but not in the
    # benign source diff -> node-level supervision for the GNN aux head. Malicious
    # modes only; hard negatives add no true injection.
    injection_markers: list[str] = []
    if job.mode != "hardneg":
        benign_added = {ln.strip() for ln in synth_validate.added_lines(benign_diff)}
        seen = set()
        for ln in synth_validate.added_lines(diff_text):
            stripped = ln.strip()
            if len(stripped) >= 6 and stripped not in benign_added and stripped not in seen:
                seen.add(stripped)
                injection_markers.append(stripped)
        injection_markers = injection_markers[:50]

    patch_hash = hashlib.sha256(diff_text.encode()).hexdigest()
    return {
        "instance_id": instance_id,
        "attack_name": attack_name,
        "attack_dataset_finalized": True,
        "graph_label": label,
        "repo_path": repo_path,
        "dataset": row.get("dataset", ""),
        # Match the real rows' vestigial split field ("test") so the loader's
        # by-instance seed-split runs and places these in train via their
        # (train-allowlisted) instance_id. Setting "train" here would wrongly
        # make the loader treat synth as the *entire* train set. See train_gnn
        # _resolve_split_rows / _split_rows.
        "split": str(row.get("split", "test")),
        "base_commit": row.get("base_commit", ""),
        "attack_artifact_path": str(art_dir),
        "patch_artifacts": {},
        "patch_hash": patch_hash,
        "attack_validation": {"apply_details": {"sanitized_diff": diff_text}},
        "synth": {
            "mode": job.mode,
            "technique": job.technique.key,
            "obfuscation": obf_key,
            "decoded_payload": decoded,
            "injection_markers": injection_markers,
            "notes": str(parsed.get("notes", "")),
            "source_instance_id": instance_id,
            "source_dataset_path": str(row.get("source_attack_dataset_path", "")),
            "gen_model": model_key,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="benign attack_dataset.jsonl to draw from")
    ap.add_argument("--train-ids", required=True, help="train_instance_ids.txt allowlist (no heldout leakage)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--modes", default="fcv,hardneg", help="comma list of fcv,swexploit,hardneg")
    ap.add_argument("--per-instance", type=int, default=3, help="examples per instance per mode")
    ap.add_argument("--model", default="sonnet", choices=list(synth_llm.MODELS.keys()))
    ap.add_argument("--project", default=os.environ.get("VERTEXAI_PROJECT")
                    or os.environ.get("GOOGLE_CLOUD_PROJECT") or "ucr-ursa-major-socal-lab")
    ap.add_argument("--max-benign-diff-chars", type=int, default=6000,
                    help="skip source patches larger than this (LLM must reproduce them faithfully)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of source instances")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in tech.MODES:
            raise SystemExit(f"unknown mode {m!r}; valid: {tech.MODES}")

    train_ids = {ln.strip() for ln in Path(args.train_ids).read_text().splitlines() if ln.strip()}
    rows = [json.loads(l) for l in Path(args.source).read_text().splitlines() if l.strip()]

    # benign, train-split, small-enough source patches only
    src_rows = []
    for r in rows:
        if str(r.get("attack_name", "")) not in ("", "none"):
            continue
        if str(r.get("instance_id", "")) not in train_ids:
            continue
        patch = load_patch(r)
        if not patch or len(patch) > args.max_benign_diff_chars:
            continue
        src_rows.append(r)

    rng = random.Random(args.seed)
    rng.shuffle(src_rows)
    if args.limit:
        src_rows = src_rows[: args.limit]

    # Build jobs with round-robin technique/obfuscation for maximum diversity.
    jobs: list[Job] = []
    counter = 0
    for r in src_rows:
        for mode in modes:
            for k in range(args.per_instance):
                t = tech.TECHNIQUES[counter % len(tech.TECHNIQUES)]
                o = tech.OBFUSCATIONS[(counter // len(tech.TECHNIQUES)) % len(tech.OBFUSCATIONS)]
                jobs.append(Job(r, mode, t, o, k))
                counter += 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"

    client = synth_llm.VertexLLM(args.project)
    print(f"[synth] {len(src_rows)} source instances -> {len(jobs)} jobs "
          f"({modes}, {args.per_instance}/mode, model={args.model})", flush=True)

    malicious, benign, skips = [], [], []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_job, client, args.model, artifacts_dir, j) for j in jobs]
        it = as_completed(futs)
        if tqdm is not None:
            it = tqdm(it, total=len(futs), desc="synth", unit="job")
        for fut in it:
            res = fut.result()
            if res is None:
                continue
            if "_skip" in res:
                skips.append(res)
            elif res["graph_label"] == 1:
                malicious.append(res)
            else:
                benign.append(res)

    mal_path = out_dir / "synth_malicious.jsonl"
    ben_path = out_dir / "synth_benign.jsonl"
    mal_path.write_text("".join(json.dumps(r) + "\n" for r in malicious), encoding="utf-8")
    ben_path.write_text("".join(json.dumps(r) + "\n" for r in benign), encoding="utf-8")

    from collections import Counter
    summary = {
        "source": args.source,
        "jobs": len(jobs),
        "malicious_written": len(malicious),
        "benign_written": len(benign),
        "skipped": len(skips),
        "by_technique": dict(Counter(r["synth"]["technique"] for r in malicious)),
        "by_obfuscation": dict(Counter(r["synth"]["obfuscation"] for r in malicious)),
        "skip_reasons": dict(Counter(s["_skip"].split(":")[0] for s in skips)),
        "outputs": {"malicious": str(mal_path), "benign": str(ben_path)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
