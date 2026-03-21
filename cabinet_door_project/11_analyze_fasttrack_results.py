"""
Analyze fast-track ablation outputs and generate report-ready artifacts.

Usage:
  python 11_analyze_fasttrack_results.py \
    --run_dir sweep_results/ablations_fasttrack_20260320_155808
"""

import argparse
import csv
import os
import re
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EP_RE = re.compile(
    r"Episode\s+\d+/\d+:\s+(SUCCESS|FAIL)\s+"
    r"\(steps=\s*\d+,\s*reward=[^)]+\)\s+layout=(\d+),\s+style=(\d+),"
)
EVAL_HEADER_RE = re.compile(
    r"---- Eval: config=([^,]+), seed=(\d+), max_steps=(\d+) ----"
)


def read_results_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            try:
                row["seed"] = int(row["seed"])
                row["max_steps"] = int(row["max_steps"])
                row["epochs"] = int(row["epochs"])
                row["chunk_size"] = int(row["chunk_size"])
                row["n_action_steps"] = int(row["n_action_steps"])
                row["successes"] = int(row["successes"])
                row["episodes"] = int(row["episodes"])
                row["success_rate"] = float(row["success_rate"])
            except Exception:
                continue
            rows.append(row)
    return rows


def parse_episode_combos(log_path):
    """
    Parse per-episode layout/style outcomes from the ablation log and
    attach them to (config, seed, max_steps) context.
    """
    combos = []
    current = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m_eval = EVAL_HEADER_RE.search(line)
            if m_eval:
                current = {
                    "config": m_eval.group(1).strip(),
                    "seed": int(m_eval.group(2)),
                    "max_steps": int(m_eval.group(3)),
                }
                continue

            m_ep = EP_RE.search(line)
            if m_ep and current is not None:
                combos.append(
                    {
                        "config": current["config"],
                        "seed": current["seed"],
                        "max_steps": current["max_steps"],
                        "success": m_ep.group(1) == "SUCCESS",
                        "layout": int(m_ep.group(2)),
                        "style": int(m_ep.group(3)),
                    }
                )
    return combos


def write_rankings_md(rows, combo_rows, out_md):
    lines = []
    lines.append("# Fast-Track Analysis\n")

    lines.append("## Best Runs")
    lines.append("")
    lines.append("| rank | config | seed | max_steps | success | rate |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    ranked = sorted(
        rows,
        key=lambda r: (r["success_rate"], r["successes"], -r["max_steps"]),
        reverse=True,
    )
    for i, r in enumerate(ranked[:10], start=1):
        lines.append(
            f"| {i} | {r['config']} | {r['seed']} | {r['max_steps']} | "
            f"{r['successes']}/{r['episodes']} | {r['success_rate']:.1f}% |"
        )

    lines.append("")
    lines.append("## Top Layout+Style Combinations (from episode logs)")
    lines.append("")
    lines.append("| rank | layout | style | successes | attempts | success_rate |")
    lines.append("|---:|---:|---:|---:|---:|---:|")

    combo_counts = defaultdict(lambda: {"succ": 0, "total": 0})
    for c in combo_rows:
        key = (c["layout"], c["style"])
        combo_counts[key]["total"] += 1
        if c["success"]:
            combo_counts[key]["succ"] += 1

    top = sorted(
        combo_counts.items(),
        key=lambda kv: (kv[1]["succ"], kv[1]["succ"] / max(1, kv[1]["total"])),
        reverse=True,
    )
    for i, ((layout, style), v) in enumerate(top[:15], start=1):
        rate = 100.0 * v["succ"] / max(1, v["total"])
        lines.append(
            f"| {i} | {layout} | {style} | {v['succ']} | {v['total']} | {rate:.1f}% |"
        )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_success_by_config(rows, out_png):
    if not rows:
        return
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["config"], r["max_steps"])].append(r["success_rate"])

    labels = []
    vals = []
    for key in sorted(by_key.keys()):
        cfg, ms = key
        labels.append(f"{cfg}\n{ms}")
        vals.append(sum(by_key[key]) / len(by_key[key]))

    plt.figure(figsize=(8.2, 4.0))
    plt.bar(range(len(vals)), vals, color="#4C78A8")
    plt.xticks(range(len(vals)), labels, fontsize=8)
    plt.ylabel("Mean success rate (%)")
    plt.title("Fast-track ablation: mean success by config and max_steps")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_success_by_seed(rows, out_png):
    if not rows:
        return
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r["success_rate"])

    seeds = sorted(by_seed.keys())
    means = [sum(by_seed[s]) / len(by_seed[s]) for s in seeds]
    plt.figure(figsize=(5.0, 3.6))
    plt.bar([str(s) for s in seeds], means, color="#72B7B2")
    plt.ylabel("Mean success rate (%)")
    plt.xlabel("Seed")
    plt.title("Fast-track ablation: mean success by seed")
    plt.ylim(0, 100)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def plot_top_combos(combo_rows, out_png):
    if not combo_rows:
        return
    stats = defaultdict(lambda: {"succ": 0, "total": 0})
    for c in combo_rows:
        k = (c["layout"], c["style"])
        stats[k]["total"] += 1
        if c["success"]:
            stats[k]["succ"] += 1

    top = sorted(
        stats.items(),
        key=lambda kv: (kv[1]["succ"], kv[1]["succ"] / max(1, kv[1]["total"])),
        reverse=True,
    )[:10]
    labels = [f"L{l}-S{s}" for (l, s), _ in top]
    succ = [v["succ"] for _, v in top]

    plt.figure(figsize=(8.5, 4.0))
    plt.bar(range(len(succ)), succ, color="#F58518")
    plt.xticks(range(len(succ)), labels, rotation=25, ha="right", fontsize=8)
    plt.ylabel("Success count")
    plt.title("Top successful layout/style combinations (current log)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze fast-track ablation outputs")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to fast-track run directory")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    csv_path = os.path.join(run_dir, "results.csv")
    log_path = os.path.join(run_dir, "ablations.log")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    rows = read_results_csv(csv_path)
    combo_rows = parse_episode_combos(log_path)

    summary_md = os.path.join(run_dir, "analysis_summary.md")
    write_rankings_md(rows, combo_rows, summary_md)

    plot_success_by_config(rows, os.path.join(fig_dir, "success_by_config_steps.png"))
    plot_success_by_seed(rows, os.path.join(fig_dir, "success_by_seed.png"))
    plot_top_combos(combo_rows, os.path.join(fig_dir, "top_layout_style_combos.png"))

    print(f"Run dir: {run_dir}")
    print(f"Completed rows: {len(rows)}")
    print(f"Episode entries parsed: {len(combo_rows)}")
    print(f"Summary: {summary_md}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()

