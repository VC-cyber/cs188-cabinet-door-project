# Cabinet-Door Policy Test Plan

This file is the **living plan** for the OpenCabinet cabinet-door project.
Use it to coordinate experiments, record results, and capture pitfalls so future cycles don’t repeat the same mistakes.

## Document Status (Living)

- This file is **project-specific for the cabinet-door task**.
- Keep all planned tasks under `TODO`.
- Move tasks to `In Progress` when work starts.
- Move tasks to `Done` when finished, with:
  - Date (UTC)
  - Exact command (ideally via `scripts/cabinet_experiment.sh ...`)
  - Short result note (metrics + any key observations).
- Do **not** delete completed items; keep history so future agents can see what was tried and what worked.
- Any high-impact pitfall/learning MUST be added to `Critical Pitfalls / Learnings` immediately.

## Task Tracker

### TODO

- [ ] YYYY-MM-DD HH:MM UTC: Train a baseline action-chunking BC policy with default settings  
      Command: `scripts/cabinet_experiment.sh train`
- [ ] YYYY-MM-DD HH:MM UTC: Evaluate the best baseline checkpoint on multiple layouts/styles (pretrain split)  
      Command: `scripts/cabinet_experiment.sh eval --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt --num_rollouts 20`
- [ ] YYYY-MM-DD HH:MM UTC: Visualize a few successful and failed rollouts to understand failure modes  
      Command: `scripts/cabinet_experiment.sh viz --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt --num_episodes 3`
- [ ] YYYY-MM-DD HH:MM UTC: Train with shorter action horizon K=4 and compare success rate  
      Command: `scripts/cabinet_experiment.sh train --action_horizon 4`
- [ ] YYYY-MM-DD HH:MM UTC: Train with medium action horizon K=8 and compare success rate  
      Command: `scripts/cabinet_experiment.sh train --action_horizon 8`
- [ ] YYYY-MM-DD HH:MM UTC: Train with longer action horizon K=16 and compare smoothness/success  
      Command: `scripts/cabinet_experiment.sh train --action_horizon 16`
- [ ] YYYY-MM-DD HH:MM UTC: High-epochs baseline run to test whether more training improves success  
      Command: `scripts/cabinet_experiment.sh train --epochs 400`
- [ ] YYYY-MM-DD HH:MM UTC: Print official Diffusion Policy / pi-0 / GR00T instructions from the script  
      Command: `scripts/cabinet_experiment.sh train --use_diffusion_policy`
- [ ] YYYY-MM-DD HH:MM UTC: Launch a teleop/DAgger data-collection session to gather correction demos (requires human keyboard control)  
      Command: `cd cabinet_door_project && python 03_teleop_collect_demos.py`
- [ ] YYYY-MM-DD HH:MM UTC: After collecting new DAgger demos, retrain the baseline policy on the augmented dataset and compare eval vs the original baseline  
      Command: `scripts/cabinet_experiment.sh train --epochs 200`

> When you add new experiments, follow the same pattern: one line summary + the **exact command** you plan to run.

### In Progress

- [ ] (move an item from TODO here when you start working on it; keep the command line visible)

### Done

- [ ] (once a task finishes, copy it here, mark `[x]`, fill in date + brief metrics summary)

Example (keep this as a formatting template, then replace with real runs):

- [x] 2026-03-04: Baseline training run completed.  
      Command: `scripts/cabinet_experiment.sh train --epochs 200`  
      Result: `best_val_loss=...`, checkpoint at `/tmp/cabinet_policy_checkpoints/best_policy.pt`.

## Critical Pitfalls / Learnings

Add entries as: `YYYY-MM-DD - impact - what happened - mitigation`.
Only keep items that affect experiment correctness, reproducibility, or major runtime/cost.

Examples (replace with real cabinet-door issues as you discover them):

- YYYY-MM-DD - high impact - sim would not render because MuJoCo/GL env vars were misconfigured on Linux - mitigation: ensure `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa` for headless runs (see environment handling in `07_evaluate_policy.py` / `08_visualize_policy_rollout.py`).
- YYYY-MM-DD - medium impact - evaluation success rate was unstable due to too few episodes (`num_rollouts` too small) - mitigation: standardize on at least 20 episodes for comparing checkpoints.

## Goal

Train and select a cabinet-door policy that:
- Achieves a **high success rate** at opening the cabinet across diverse layouts/styles.
- Generalizes from pretrain to target scenes.
- Is easy to retrain and re-evaluate via the standard CLI wrapper.

## Core Tests

1. **Baseline BC with action chunking**
   - Use `06_train_policy.py` via `scripts/cabinet_experiment.sh train`.
   - Monitor training and validation losses.

2. **Policy evaluation**
   - Use `07_evaluate_policy.py` via `scripts/cabinet_experiment.sh eval ...`.
   - Report:
     - Success rate (%)
     - Average episode length
     - Average reward

3. **Qualitative rollout inspection**
   - Use `08_visualize_policy_rollout.py` via `scripts/cabinet_experiment.sh viz ...`.
   - Watch where the policy fails (handle approach, grasp, pull, etc.).

4. **Ablations / sweeps (optional follow-ups)**
   - Vary `action_horizon`, `obs_horizon`, or network size.
   - Compare success rates and qualitative behavior.

## Important Reminders

1. **Use the canonical CLI wrapper**
   - Prefer `scripts/cabinet_experiment.sh` over raw `python ...` commands in prompts and logs.

2. **Keep runs reproducible**
   - Always record:
     - Script + arguments
     - Random seed(s) if you override defaults

3. **Avoid committing artifacts**
   - Checkpoints, videos, and logs should stay outside of version control unless explicitly requested.

## Run Log Template (per experiment)

For each meaningful run, add a short block under the relevant `Done` item that looks like this:

- Run name/tag:
- Date (UTC):
- Exact command (using `scripts/cabinet_experiment.sh ...`):
- Key hyperparameters (epochs, horizons, batch size, etc.):
- Key metrics (success rate, avg reward, losses):
- Notes (failures, bottlenecks, qualitative behavior, next tweak):

*** End of File
