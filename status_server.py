#!/usr/bin/env python3
"""Tiny status server — shows GNN training progress from Windows log."""
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer

LOGS = {
    "Per-Attack (current)": ("riverxia-pc", "C:\\gnn_perattack.log"),
    "Gemini Eval 2": ("riverxia-pc", "C:\\gnn_eval2.log"),
    "Gemini (FCV retrain)": ("riverxia-pc", "C:\\gnn_gemini2.log"),
}

def fetch_log(host, path, lines=20):
    try:
        result = subprocess.run(
            ["ssh", f"river@{host}", f"powershell -Command \"Get-Content '{path}' -ErrorAction SilentlyContinue | Select-Object -Last {lines}\""],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() or "(empty)"
    except Exception as e:
        return f"(error: {e})"

def build_html():
    sections = ""
    for name, (host, path) in LOGS.items():
        content = fetch_log(host, path)
        color = "#00ff88" if "DONE" in content else "#ffcc00" if content else "#888"
        sections += f"""
        <div class="box">
            <h2 style="color:{color}">{name}</h2>
            <pre>{content}</pre>
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
<title>GNN Status</title>
<meta http-equiv="refresh" content="30">
<style>
  body {{ background:#111; color:#ccc; font-family:monospace; padding:20px; }}
  h1 {{ color:#fff; }}
  .box {{ background:#1a1a1a; border:1px solid #333; padding:16px; margin:16px 0; border-radius:6px; }}
  h2 {{ margin:0 0 10px 0; font-size:14px; }}
  pre {{ margin:0; font-size:13px; white-space:pre-wrap; word-break:break-all; }}
  .refresh {{ color:#555; font-size:12px; }}
</style>
</head>
<body>
<h1>GNN Training Status</h1>
<p class="refresh">Auto-refreshes every 30s</p>
{sections}
</body>
</html>"""

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        html = build_html().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, *args):
        pass

if __name__ == "__main__":
    port = 8787
    print(f"Status page: http://localhost:{port}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
