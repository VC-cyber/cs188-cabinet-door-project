#!/usr/bin/env python
"""
AI-driven experiment manager for the cabinet-door project using the Cursor CLI.

Each cycle runs the Cursor CLI agent once. The agent is responsible for
updating PLANS.md and for deciding and running the next experiment command
from the repo root. This script does not parse PLANS.md.

Requirements:
  - Cursor CLI `agent` on PATH.
  - Run from repo root with venv active.

Usage examples:
  python scripts/ai_cursor_daemon.py --once
  python scripts/ai_cursor_daemon.py --interval 1800   # every 30 minutes
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


AI_PROMPT = """
You are the autonomous experiment manager for this repository's
OpenCabinet cabinet-door project.

You are running in Cursor CLI Agent mode with full tool access.

Your job in EACH invocation:

1) Read `AGENTS.md` and `PLANS.md` in this repo.

2) Update `PLANS.md` as needed: keep structure and headings; add or reorder TODOs
   (action-horizon sweeps, more epochs, Diffusion Policy, DAgger, etc.); move
   tasks between TODO / In Progress / Done with short notes and metrics.
   Make minimal edits.

3) Run the next experiment yourself: decide which single experiment should run
   next from PLANS.md (usually the first unchecked TODO that has a Command).
   Run that command from the repo root (e.g. `scripts/cabinet_experiment.sh train`
   or eval/viz). Use the exact command from PLANS.md for that task. Run it in the
   foreground and wait for it to complete. If there is no runnable experiment,
   say so and do not run a shell command.

4) In your final message, give a SHORT summary: what you changed in PLANS.md
   and which command you ran (if any).
"""


def run_cursor_agent() -> int:
    """Invoke Cursor CLI `agent` once with the AI_PROMPT."""
    cmd = [
        "agent",
        "--trust",
        "-p",
        AI_PROMPT,
        "--model",
        "opus-4.6",
        "--print",
        "--output-format",
        "text",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )

    # Mirror output to this script's stdout/stderr for logging/debugging.
    if proc.stdout:
        print("=== Cursor agent stdout ===", file=sys.stdout)
        print(proc.stdout, file=sys.stdout)
    if proc.stderr:
        print("=== Cursor agent stderr ===", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)

    if proc.returncode != 0:
        print(
            f"[ai_cursor_daemon] Cursor agent exited with code {proc.returncode}",
            file=sys.stderr,
        )
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AI-driven experiment manager using Cursor CLI Agent."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1800,
        help="Seconds to sleep between cycles (ignored with --once).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single AI+daemon cycle and exit.",
    )
    args = parser.parse_args(argv)

    def cycle() -> None:
        print("[ai_cursor_daemon] Starting cycle (agent updates PLANS and runs next experiment)...", file=sys.stderr)
        run_cursor_agent()
        print("[ai_cursor_daemon] Cycle complete.", file=sys.stderr)

    if args.once:
        cycle()
        return 0

    print(
        f"[ai_cursor_daemon] Running in loop mode (interval={args.interval}s). "
        "Press Ctrl+C to stop.",
        file=sys.stderr,
    )
    try:
        while True:
            cycle()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("[ai_cursor_daemon] Stopped by user.", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

