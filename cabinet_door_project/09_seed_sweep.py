"""
Step 9: Overnight seed sweep for OpenCabinet policy
===================================================
Runs multiple train+eval jobs across random seeds and writes all output to a text log.

Usage:
    python 09_seed_sweep.py
    python 09_seed_sweep.py --seeds 0,1,2,3,4,5 --epochs 30 --num_rollouts 50
    python 09_seed_sweep.py --split target --num_rollouts 100
"""

import argparse
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path


def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_and_log(cmd, log_fh, cwd):
    """Run command, stream stdout/stderr to console and log file."""
    log_fh.write(f"\n$ {' '.join(cmd)}\n")
    log_fh.flush()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        log_fh.write(line)
        lines.append(line)
    proc.wait()
    log_fh.flush()
    return proc.returncode, "".join(lines)


def _python_unbuffered_cmd(python_exe, script_name, extra_args):
    """Build a python command that forces unbuffered stdout/stderr."""
    return [python_exe, "-u", script_name] + extra_args


def _parse_eval_metrics(output_text):
    """Extract success metrics from 07_evaluate_policy.py output."""
    success_rate = None
    successes = None
    episodes = None

    m_rate = re.search(r"Success rate:\s+([0-9.]+)%", output_text)
    if m_rate:
        success_rate = float(m_rate.group(1))

    m_count = re.search(r"Successes:\s+(\d+)/(\d+)", output_text)
    if m_count:
        successes = int(m_count.group(1))
        episodes = int(m_count.group(2))

    return success_rate, successes, episodes


def main():
    parser = argparse.ArgumentParser(description="Overnight seed sweep for OpenCabinet training/eval")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seeds, e.g. 0,1,2,3,4")
    parser.add_argument("--policy_type", type=str, default="bc_unet", choices=["bc_unet", "diffusion", "mlp"])
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per seed")
    parser.add_argument("--num_rollouts", type=int, default=50, help="Eval episodes per seed")
    parser.add_argument("--max_steps", type=int, default=600, help="Max timesteps per eval episode")
    parser.add_argument("--split", type=str, default="pretrain", choices=["pretrain", "target"])
    parser.add_argument("--checkpoint_root", type=str, default="/tmp/cabinet_policy_seed_sweep")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output text log (optional)")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to run scripts")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    checkpoint_root = Path(args.checkpoint_root).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    sweep_dir = script_dir / "sweep_results"
    sweep_dir.mkdir(exist_ok=True)
    output_file = Path(args.output_file) if args.output_file else sweep_dir / f"seed_sweep_{_timestamp()}.txt"

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("No seeds provided")

    results = []
    with output_file.open("w", encoding="utf-8") as log_fh:
        log_fh.write("=" * 70 + "\n")
        log_fh.write("OpenCabinet Overnight Seed Sweep\n")
        log_fh.write("=" * 70 + "\n")
        log_fh.write(f"Started:         {datetime.datetime.now().isoformat()}\n")
        log_fh.write(f"Script dir:      {script_dir}\n")
        log_fh.write(f"Policy type:     {args.policy_type}\n")
        log_fh.write(f"Seeds:           {seeds}\n")
        log_fh.write(f"Epochs:          {args.epochs}\n")
        log_fh.write(f"Eval rollouts:   {args.num_rollouts}\n")
        log_fh.write(f"Eval max_steps:  {args.max_steps}\n")
        log_fh.write(f"Eval split:      {args.split}\n")
        log_fh.write(f"Checkpoint root: {checkpoint_root}\n")
        log_fh.write("=" * 70 + "\n")
        log_fh.flush()

        for i, seed in enumerate(seeds, start=1):
            log_fh.write(f"\n\n{'#' * 70}\n")
            log_fh.write(f"Seed {seed} ({i}/{len(seeds)})\n")
            log_fh.write(f"{'#' * 70}\n")
            log_fh.flush()

            seed_ckpt_dir = checkpoint_root / f"seed_{seed}"
            seed_ckpt_dir.mkdir(parents=True, exist_ok=True)
            best_ckpt = seed_ckpt_dir / "best_policy.pt"

            train_cmd = _python_unbuffered_cmd(
                args.python,
                "06_train_policy.py",
                [
                    "--policy_type",
                    args.policy_type,
                    "--epochs",
                    str(args.epochs),
                    "--checkpoint_dir",
                    str(seed_ckpt_dir),
                ],
            )
            eval_cmd = _python_unbuffered_cmd(
                args.python,
                "07_evaluate_policy.py",
                [
                    "--checkpoint",
                    str(best_ckpt),
                    "--num_rollouts",
                    str(args.num_rollouts),
                    "--max_steps",
                    str(args.max_steps),
                    "--split",
                    args.split,
                    "--seed",
                    str(seed),
                ],
            )

            train_code, train_out = _run_and_log(train_cmd, log_fh, cwd=str(script_dir))
            if train_code != 0 or not best_ckpt.exists():
                results.append(
                    {
                        "seed": seed,
                        "status": "train_failed",
                        "success_rate": None,
                        "successes": None,
                        "episodes": None,
                    }
                )
                log_fh.write(f"\n[seed {seed}] TRAIN FAILED (exit_code={train_code})\n")
                log_fh.flush()
                continue

            eval_code, eval_out = _run_and_log(eval_cmd, log_fh, cwd=str(script_dir))
            if eval_code != 0:
                results.append(
                    {
                        "seed": seed,
                        "status": "eval_failed",
                        "success_rate": None,
                        "successes": None,
                        "episodes": None,
                    }
                )
                log_fh.write(f"\n[seed {seed}] EVAL FAILED (exit_code={eval_code})\n")
                log_fh.flush()
                continue

            success_rate, successes, episodes = _parse_eval_metrics(eval_out)
            results.append(
                {
                    "seed": seed,
                    "status": "ok",
                    "success_rate": success_rate,
                    "successes": successes,
                    "episodes": episodes,
                }
            )

            log_fh.write(
                f"\n[seed {seed}] success_rate={success_rate}%, successes={successes}/{episodes}\n"
            )
            log_fh.flush()

        ok_results = [r for r in results if r["status"] == "ok" and r["success_rate"] is not None]
        ok_results_sorted = sorted(ok_results, key=lambda r: r["success_rate"], reverse=True)

        log_fh.write("\n\n" + "=" * 70 + "\n")
        log_fh.write("Sweep Summary\n")
        log_fh.write("=" * 70 + "\n")
        for r in results:
            if r["status"] == "ok":
                log_fh.write(
                    f"seed={r['seed']:>3}  success={r['successes']}/{r['episodes']}  "
                    f"rate={r['success_rate']:.1f}%\n"
                )
            else:
                log_fh.write(f"seed={r['seed']:>3}  status={r['status']}\n")

        if ok_results_sorted:
            best = ok_results_sorted[0]
            log_fh.write(
                f"\nBest seed: {best['seed']}  "
                f"({best['successes']}/{best['episodes']} = {best['success_rate']:.1f}%)\n"
            )
        else:
            log_fh.write("\nNo successful train+eval runs completed.\n")

        log_fh.write(f"\nFinished: {datetime.datetime.now().isoformat()}\n")

    print(f"\nDone. Seed sweep log written to:\n  {output_file}")


if __name__ == "__main__":
    main()
