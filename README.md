# CS188 Cabinet Door Project - based on starter repo
Venkat Chitturi and Sophie Zhu


Project repository for low-dimensional imitation learning on RoboCasa OpenCabinet.

This README is written to document the actual setup, commands, logs, and artifacts
used in our final pipeline (BC U-Net + handle augmentation + ablation + 50-rollout
comparisons).

## 1) Environment Setup

From repo root:

```bash
./install.sh
source .venv/bin/activate
cd cabinet_door_project
python 00_verify_installation.py
```

Notes:
- On macOS, on-screen viewer scripts may require `mjpython` instead of `python`.
- Off-screen evaluation/video generation in this project is run with standard Python.

## 2) Dataset Setup

Download dataset:

```bash
cd cabinet_door_project
python 04_download_dataset.py
```

Add handle-relevant augmented features:

```bash
python 05b_augment_handle_data.py
```

This creates `augmented/` parquet features that are automatically consumed by the
training/eval pipeline.

## 3) Core Training and Evaluation Commands

### A) Baseline report reproduction (BC U-Net)

Train:

```bash
python 06_train_policy.py --policy_type bc_unet --epochs 30 --checkpoint_dir repro_logs/checkpoints
```

Evaluate with 50 rollouts:

```bash
python 07_evaluate_policy.py --checkpoint repro_logs/checkpoints/best_policy.pt --num_rollouts 50 --max_steps 500
python 07_evaluate_policy.py --checkpoint repro_logs/checkpoints/best_policy.pt --num_rollouts 50 --max_steps 600
```

### B) Fast-track ablation run (completed)

Completed run directory:

`cabinet_door_project/sweep_results/ablations_fasttrack`

Key outputs:
- `results.csv`
- `ablations.log`
- `analysis_summary.md`
- `figures/`

### C) Post-ablation 50-rollout confirmation for best config

Best candidate checkpoint from fast-track:

`/tmp/cabinet_ablations_fasttrack_20260320_155808/chunk12_6/seed_0/best_policy.pt`

Evaluations run:

```bash
python 07_evaluate_policy.py --checkpoint /tmp/cabinet_ablations_fasttrack_20260320_155808/chunk12_6/seed_0/best_policy.pt --num_rollouts 50 --max_steps 500 --seed 0
python 07_evaluate_policy.py --checkpoint /tmp/cabinet_ablations_fasttrack_20260320_155808/chunk12_6/seed_0/best_policy.pt --num_rollouts 50 --max_steps 600 --seed 0
```

Saved logs:
- `cabinet_door_project/repro_logs/eval_chunk12_6_seed0_50x500.log`
- `cabinet_door_project/repro_logs/eval_chunk12_6_seed0_50x600.log`

## 4) What "chunk12_6" Means

Compared to original `chunk16_8`:
- `chunk_size=12`: predict 12 future actions each re-plan
- `n_action_steps=6`: execute first 6 actions, then re-plan

This is shorter horizon / more frequent re-planning than `16/8`.

## 5) Metrics Snapshot (Current)

### Original 50-rollout reproduction
- BC U-Net `max_steps=500`: `21/50 = 42.0%`
- BC U-Net `max_steps=600`: `20/50 = 40.0%`

### Post-ablation 50-rollout best-config confirmation
- chunk12_6 seed0 `max_steps=500`: `25/50 = 50.0%`
- chunk12_6 seed0 `max_steps=600`: `24/50 = 48.0%`

Improvement vs original reproduction: `+8` points at both horizons.

## 6) Evaluation Metric Note

Reported success metrics in this project use a **slightly relaxed success criterion**
(cabinet opening threshold) implemented consistently in the evaluation helpers.

This is intentional and aligns with our report protocol for stable state-based comparisons.

## 7) Figures and Logs Used in Final Report

### Main report figures
- `cabinet_door_project/repro_logs/figures/training_curve_bc_unet.png`
- `cabinet_door_project/repro_logs/figures/eval_success_comparison.png`
- `cabinet_door_project/sweep_results/ablations_fasttrack/figures/success_by_config_steps.png`

### Main logs
- `cabinet_door_project/repro_logs/train_bc_unet_e30.log`
- `cabinet_door_project/repro_logs/eval_bc_unet_50x500.log`
- `cabinet_door_project/repro_logs/eval_bc_unet_50x600.log`
- `cabinet_door_project/repro_logs/eval_chunk12_6_seed0_50x500.log`
- `cabinet_door_project/repro_logs/eval_chunk12_6_seed0_50x600.log`
- `cabinet_door_project/sweep_results/ablations_fasttrack/results.csv`
- `cabinet_door_project/sweep_results/ablations_fasttrack/analysis_summary.md`

## 8) Video Artifacts

### BC U-Net eval clips
- `cabinet_door_project/repro_logs/eval_bc_unet_seed0_ep1to2_600.mp4`
- `cabinet_door_project/repro_logs/eval_bc_unet_seed0_ep1to2_600_2x.mp4`
- `cabinet_door_project/repro_logs/eval_seed0_ep1to2_600_from10s.mp4`
- `cabinet_door_project/repro_logs/eval_seed0_ep1to2_600_from10s_2x.mp4`

### Best/failure visualization clips (chunk12_6 checkpoint)
- `cabinet_door_project/repro_logs/best_chunk12_seed1_vis.mp4` (successful example)
- `cabinet_door_project/repro_logs/best_chunk12_seed0_fail_vis.mp4` (explicit failure example)
- `cabinet_door_project/repro_logs/best_chunk12_seed0_fail_vis_2x.mp4`

## 9) Two-Column Report File

Final two-column LaTeX report:

- `FINAL_PROJECT_REPORT.tex`

## 10) Troubleshooting Quick Notes

- If checkpoint path is under `/tmp` and disappears, retrain with `--checkpoint_dir` inside repo.
- If rendering/window issues occur on macOS, use off-screen mode for videos.
- If `ffmpeg` is unavailable, clips in this repo were processed with Python `imageio`.

