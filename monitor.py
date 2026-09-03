#!/usr/bin/env python3
"""Simple web UI to monitor featurebench attack/baseline run progress."""

import re
import json
import subprocess
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

ROOT = Path(__file__).parent


def get_running_procs():
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", "run_attack_sharded|run_defense_sharded|train_gnn|run_retry|run_featurebench|run_gemini|run_claude"],
            text=True
        )
        return [l.strip() for l in out.strip().split("\n") if l.strip()]
    except Exception:
        return []


def parse_latest_tqdm(log_path):
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
        for line in reversed(lines):
            m = re.search(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]<]+)<([^\]]+)\]", line)
            if m:
                return {"pct": int(m.group(1)), "done": int(m.group(2)), "total": int(m.group(3)),
                        "elapsed": m.group(4).strip(), "eta": m.group(5).strip()}
    except Exception:
        pass
    return None


def scan_attacks():
    results = []
    for model in ["gemini3_flash", "claude37_sonnet_sweagent"]:
        for attack in ["none", "fcv_cwe78_base64_obfuscated", "swexploit_base64_obfuscated"]:
            name = f"featurebench_full_{attack}"
            final = ROOT / f"outputs/attacks/{model}/full/{name}/attack_results.jsonl"
            shards_dir = ROOT / f"outputs/attacks/{model}/full/shards"

            # Find newest shard log
            logs = sorted(shards_dir.glob(f"shard_*/{name}/logs/runner/stdout.log"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
            progress = None
            if logs:
                progress = parse_latest_tqdm(logs[-1])

            # Determine if actively running
            running = any(
                f"--attack {attack}" in p and model in p
                for p in get_running_procs()
            )

            finalized = 0
            discarded = 0
            if final.exists():
                rows = [json.loads(l) for l in final.read_text().splitlines() if l.strip()]
                empty_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                finalized = sum(1 for r in rows if r.get("patch_hash") and r["patch_hash"] != empty_hash)
                discarded = len(rows) - finalized

            results.append({
                "model": model, "attack": attack,
                "running": running,
                "finalized": finalized, "discarded": discarded,
                "progress": progress,
            })
    return results


def scan_baselines():
    results = []
    base = ROOT / "outputs/baselines/featurebench_obfuscated_heldout"
    if not base.exists():
        return results
    for results_file in sorted(base.glob("*/*/*/results.jsonl")):
        parts = results_file.parts[-5:]
        model, attack, baseline = parts[0], parts[1], parts[2]
        count = sum(1 for _ in open(results_file))
        results.append({"model": model, "attack": attack, "baseline": baseline, "rows": count})
    return results


def build_html():
    procs = get_running_procs()
    attacks = scan_attacks()
    baselines = scan_baselines()

    def status_badge(running, finalized, discarded, progress):
        if running and progress:
            pct = progress["pct"]
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            return f'<span style="color:#f39c12">⟳ running</span> <code style="font-size:11px">{bar} {progress["done"]}/{progress["total"]} eta:{progress["eta"]}</code>'
        elif running:
            return '<span style="color:#f39c12">⟳ running</span>'
        elif finalized > 0 or discarded > 0:
            return f'<span style="color:#2ecc71">✓ done</span> <span style="color:#8b949e;font-size:12px">{finalized} finalized / {discarded} discarded</span>'
        else:
            return '<span style="color:#95a5a6">— pending</span>'

    attack_rows = ""
    for r in attacks:
        a = r["attack"].replace("_base64_obfuscated", "").replace("fcv_cwe78", "fcv")
        m = r["model"].replace("claude37_sonnet_sweagent", "claude-sweagent").replace("gemini3_flash", "gemini-mini")
        sb = status_badge(r["running"], r["finalized"], r["discarded"], r["progress"])
        attack_rows += f"<tr><td>{m}</td><td>{a}</td><td>{sb}</td></tr>"

    baseline_rows = ""
    for r in baselines:
        a = r["attack"].replace("_base64_obfuscated", "").replace("fcv_cwe78", "fcv")
        m = r["model"].replace("claude37_sonnet_sweagent", "claude").replace("gemini3_flash", "gemini")
        b = r["baseline"].replace("structural_misalignment_featurebench_", "sm-").replace("_obfuscated_trained", "").replace("llm_judge_", "llm-judge-").replace("_vertex", "")
        baseline_rows += f"<tr><td>{m}</td><td>{a}</td><td>{b}</td><td>{r['rows']}</td></tr>"

    proc_html = "".join(
        f'<div style="font-family:monospace;font-size:11px;color:#ccc;padding:3px 0;border-bottom:1px solid #21262d">{p[:150]}</div>'
        for p in procs
    ) or "<em style='color:#8b949e'>no matching processes</em>"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>featurebench monitor</title>
<meta http-equiv="refresh" content="15">
<style>
* {{box-sizing:border-box;}}
body {{font-family:-apple-system,sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:20px;}}
h1 {{color:#e6edf3;font-size:20px;margin-bottom:20px;}}
h2 {{color:#58a6ff;font-size:14px;text-transform:uppercase;letter-spacing:1px;margin:0 0 12px;}}
.section {{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:16px;}}
table {{border-collapse:collapse;width:100%;}}
th {{color:#8b949e;font-size:11px;text-transform:uppercase;padding:6px 10px;text-align:left;border-bottom:1px solid #30363d;}}
td {{padding:8px 10px;font-size:13px;border-bottom:1px solid #21262d;}}
code {{background:#0d1117;padding:2px 6px;border-radius:4px;}}
</style>
</head>
<body>
<h1>🔬 featurebench monitor <span style="font-size:12px;color:#8b949e">auto-refresh 15s</span></h1>

<div class="section">
<h2>Attack datasets</h2>
<table>
<tr><th>Model</th><th>Attack</th><th>Status</th></tr>
{attack_rows}
</table>
</div>

<div class="section">
<h2>Baseline results ({len(baselines)} complete)</h2>
<table>
<tr><th>Model</th><th>Attack</th><th>Baseline</th><th>Rows</th></tr>
{baseline_rows or '<tr><td colspan=4 style="color:#8b949e"><em>none yet</em></td></tr>'}
</table>
</div>

<div class="section">
<h2>Active processes ({len(procs)})</h2>
{proc_html}
</div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        html = build_html().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html)
    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = 8765
    print(f"Monitor at http://localhost:{port}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
