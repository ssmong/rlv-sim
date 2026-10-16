"""Serve the standalone coordinate-frame explorer.

Usage (repo root):
    python scripts/view_frames.py
    python scripts/view_frames.py --system
    python scripts/view_frames.py --port 8001 --no-open

The server exposes only static repository files and opens ``viz/frame_studio.html``. No simulation
run or replay.json is required. Press Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import socketserver
import sys
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

socketserver.TCPServer.allow_reuse_address = os.name != "nt"


class _QuietHTTP(http.server.SimpleHTTPRequestHandler):
    quiet = True

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        if not self.quiet:
            super().log_message(fmt, *args)


def _serve(
    handler: Callable[..., http.server.SimpleHTTPRequestHandler], port: int, tries: int = 20
) -> tuple[socketserver.TCPServer, int]:
    last = None
    for candidate in range(port, port + tries):
        try:
            return socketserver.TCPServer(("", candidate), handler), candidate
        except OSError as error:
            last = error
    sys.exit(f"[frames] no free port in {port}..{port + tries - 1}: {last}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--port", type=int, default=8000, help="preferred port (rolls to next free one)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="host in the printed URL")
    parser.add_argument(
        "--system",
        action="store_true",
        help="open the external browser even inside VS Code",
    )
    parser.add_argument(
        "--no-open", action="store_true", help="print the URL without opening a browser"
    )
    parser.add_argument("--verbose", action="store_true", help="log every HTTP request")
    args = parser.parse_args(argv)

    _QuietHTTP.quiet = not args.verbose
    handler = functools.partial(_QuietHTTP, directory=str(ROOT))
    httpd, port = _serve(handler, args.port)
    url = f"http://{args.host}:{port}/viz/frame_studio.html"
    in_vscode = os.environ.get("TERM_PROGRAM") == "vscode"

    with httpd:
        print(f"[frames] serving {ROOT} at http://{args.host}:{port}/")
        if port != args.port:
            print(f"[frames] (port {args.port} was busy; rolled to {port})")
        print(f"[frames] open: {url}")
        if not (args.no_open or (in_vscode and not args.system)):
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[frames] stopped.")


if __name__ == "__main__":
    main()
