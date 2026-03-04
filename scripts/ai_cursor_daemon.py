#!/usr/bin/env python
"""
AI-driven experiment manager for the cabinet-door project using the Cursor CLI.

This script:
  1) Invokes the Cursor CLI `agent` command with a tight prompt so it:
       - Reads `AGENTS.md` and `PLANS.md`
       - Updates `PLANS.md` (plan, reorder TODOs, promote items to Done, etc.)
       - Chooses the *next* experiment by ensuring the first unchecked TODO with
         a `Command: scripts/cabinet_experiment.sh ...` line is what it wants next
       - Does NOT run shell commands itself
  2) Runs `scripts/cabinet_daemon.sh --once` to actually execute the chosen
     experiment (if any), respecting the lock file.
  3) Sleeps for a configurable interval and repeats.

Requirements:
  - Cursor CLI installed and `agent` available on PATH.
    See: https://cursor.com/docs/cli/overview
  - Run this from your virtualenv / environment that can run your experiments.

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
  2) Treat `PLANS.md` as the single source of truth for planned, running,
     and completed experiments.
  3) Update ONLY `PLANS.md` to:
       - Keep its existing structure and headings.
       - Add or tweak TODO items based on current results and the README's
         suggestions (action-horizon sweeps, more epochs, Diffusion Policy,
         DAgger ideas, etc.).
       - Move tasks between TODO / In Progress / Done when appropriate,
         including short notes + metrics for completed runs.
       - ENSURE that the FIRST unchecked TODO item which contains a
         `Command: scripts/cabinet_experiment.sh ...`
         line is exactly the NEXT experiment you want the shell daemon
         (`scripts/cabinet_daemon.sh`) to run.
  4) DO NOT run any shell commands or start jobs yourself. Only edit files.
  5) Make the MINIMAL necessary edits to `PLANS.md` — avoid rewriting
     large sections or changing unrelated text.

Very important:
  - Do not edit any files other than `PLANS.md` unless there is a compelling,
    experiment-related reason.
  - Do not assume external GPUs / remote clusters; stay consistent with
    the local cabinet-door project context.
  - If the bottom of `PLANS.md` still contains legacy Whisper-related
    content from a different project, you may remove that section so the
    file only describes the cabinet-door task.

At the end of your work, write a SHORT natural-language summary (a few lines)
in your final message describing:
  - Which TODO (command) you queued first.
  - Any key changes you made to `PLANS.md`.
"""


def run_cursor_agent() -> int:
    """Invoke Cursor CLI `agent` once with the AI_PROMPT."""
    cmd = [
        "agent",
        "-p",
        AI_PROMPT,
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


def run_shell_daemon_once() -> int:
    """
    Ask the shell-level daemon to execute at most one experiment based on PLANS.md.
    This respects the lock file and will no-op if something is already running
    or no Command is available.
    """
    cmd = ["scripts/cabinet_daemon.sh", "--once"]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
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
        print(
            "[ai_cursor_daemon] Starting cycle: calling Cursor agent to update PLANS.md...",
            file=sys.stderr,
        )
        run_cursor_agent()

        print(
            "[ai_cursor_daemon] Calling shell daemon: scripts/cabinet_daemon.sh --once",
            file=sys.stderr,
        )
        run_shell_daemon_once()

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

