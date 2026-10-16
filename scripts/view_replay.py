"""Serve the web viewer pointed at a run's replay.json — loads straight from the results folder.

Usage (repo root, venv active):
    python scripts/view_replay.py                                   # latest results/**/replay.json
    python scripts/view_replay.py results/run_single/2026-06-09/14-30-22
    python scripts/view_replay.py path/to/replay.json --port 8001 --system

Serves the REPO ROOT over HTTP so the viewer assets (viz/) and the run's replay.json (results/) are
both reachable, then opens ``viz/index.html?replay=<abs-path-from-root>``. Nothing is copied — the
viewer fetches the replay straight from its run folder. Ctrl-C to stop.

Port handling: if the requested port is busy it rolls to the next free one, so a stale
``python -m http.server 8000`` won't collide. SO_REUSEADDR is left OFF on Windows, where it would
otherwise let two servers bind the same port and answer requests unpredictably (mystery 404s).

VS Code: from the integrated terminal it prints the URL for the built-in **Simple Browser**
(Ctrl/Cmd+Shift+P -> "Simple Browser: Show" -> paste) instead of popping the external browser.
Use --system to force the external browser, or --no-open to just print the URL.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import socketserver
import sys
import webbrowser
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]

# SO_REUSEADDR is correct on POSIX (avoids TIME_WAIT re-bind failures) but a footgun on Windows,
# where it lets a SECOND server bind an already-used port and then steal requests unpredictably.
# Off on Windows; the port-fallback in _serve() covers the restart case there.
socketserver.TCPServer.allow_reuse_address = os.name != "nt"


class _QuietHTTP(http.server.SimpleHTTPRequestHandler):
    """Static handler that stays silent per request unless --verbose, so the console isn't spammed
    with one access-log line ('GET ... 200') per asset every time the page (re)loads."""

    quiet = True

    def end_headers(self):
        # No-store so an edited app.js / replay.json is always re-fetched: ES modules are cached
        # hard by filename (src="./app.js" has no version query), so a plain reload otherwise
        # serves stale JS.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        if not _QuietHTTP.quiet:
            super().log_message(fmt, *args)


def _resolve_replay(arg):
    """Resolve a run dir / replay path, or fall back to the newest results/**/replay.json."""
    if arg:
        p = Path(arg).resolve()
        if p.is_dir():
            p = p / "replay.json"
        if not p.is_file():
            sys.exit(f"[view] no replay.json at {p}")
        return p
    results = ROOT / "results"
    cands = list(results.rglob("replay.json")) if results.is_dir() else []
    if not cands:
        sys.exit(
            f"[view] no replay.json under {results} — run `python scripts/run_single.py` first, "
            f"or pass an explicit run dir / replay.json path."
        )
    return max(cands, key=lambda q: q.stat().st_mtime)


def _serve(handler, port, tries=20):
    """Bind the first free port at/above `port` (loud failure; never a silent dual-bind)."""
    last = None
    for p in range(port, port + tries):
        try:
            return socketserver.TCPServer(("", p), handler), p
        except OSError as e:
            last = e
    sys.exit(f"[view] no free port in {port}..{port + tries - 1}: {last}")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "run", nargs="?", help="run dir or replay.json path (default: latest under results/)"
    )
    ap.add_argument(
        "--port", type=int, default=8000, help="preferred port (rolls to next free one)"
    )
    # Default host 127.0.0.1, not localhost: a browser that HTTPS-upgrades (HSTS / HTTPS-First mode)
    # speaks TLS to our plain-HTTP port and every request 400s. HSTS is keyed by hostname, so the
    # numeric host dodges a poisoned `localhost` entry. Pass --host localhost to override.
    ap.add_argument(
        "--host", default="127.0.0.1", help="host in the printed URL (default 127.0.0.1)"
    )
    ap.add_argument(
        "--system", action="store_true", help="open the external browser even inside VS Code"
    )
    ap.add_argument(
        "--no-open", action="store_true", help="don't auto-open anything, just print the URL"
    )
    ap.add_argument(
        "--verbose", action="store_true", help="log every HTTP request (off by default)"
    )
    args = ap.parse_args(argv)

    replay = _resolve_replay(args.run)
    try:
        rel = replay.relative_to(ROOT)
    except ValueError:
        sys.exit(
            f"[view] {replay} is outside the repo root {ROOT}; move it under results/ to serve it."
        )

    _QuietHTTP.quiet = not args.verbose
    handler = functools.partial(_QuietHTTP, directory=str(ROOT))
    httpd, port = _serve(handler, args.port)
    url = f"http://{args.host}:{port}/viz/index.html?replay=/{quote(rel.as_posix())}"
    in_vscode = os.environ.get("TERM_PROGRAM") == "vscode"

    with httpd:
        print(f"[view] serving {ROOT} at http://{args.host}:{port}/")
        if port != args.port:
            print(f"[view] (port {args.port} was busy — rolled to {port})")
        print(f"[view] replay: {rel.as_posix()}")
        print(f"[view] open:   {url}")
        # In VS Code, paste the URL above into Simple Browser (we don't pop the external browser).
        # --system forces the external browser; --no-open suppresses opening entirely.
        if not (args.no_open or (in_vscode and not args.system)):
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[view] stopped.")


if __name__ == "__main__":
    main(sys.argv[1:])
