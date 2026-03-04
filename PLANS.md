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

- [ ] 2026-03-04: Evaluate the best baseline checkpoint on multiple layouts/styles (pretrain split)  
      Command: `scripts/cabinet_experiment.sh eval --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt --num_rollouts 20`
- [ ] 2026-03-04: Visualize a few successful and failed rollouts to understand failure modes  
      Command: `scripts/cabinet_experiment.sh viz --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt --num_episodes 3`
- [ ] 2026-03-04: Action-horizon sweep K=4 — compare success rate vs baseline  
      Command: `scripts/cabinet_experiment.sh train --action_horizon 4`
- [ ] 2026-03-04: Action-horizon sweep K=8 — compare success rate vs baseline  
      Command: `scripts/cabinet_experiment.sh train --action_horizon 8`
- [ ] 2026-03-04: Action-horizon sweep K=16 — compare smoothness and success vs baseline  
      Command: `scripts/cabinet_experiment.sh train --action_horizon 16`
- [ ] 2026-03-04: High-epochs baseline (400) to test whether longer training improves success  
      Command: `scripts/cabinet_experiment.sh train --epochs 400`
- [ ] 2026-03-04: Print official Diffusion Policy / pi-0 / GR00T setup instructions  
      Command: `scripts/cabinet_experiment.sh train --use_diffusion_policy`
- [ ] 2026-03-04: (Manual) Launch a teleop/DAgger data-collection session — requires human keyboard control  
      Command: `cd cabinet_door_project && python 03_teleop_collect_demos.py`
- [ ] 2026-03-04: After collecting DAgger demos, retrain on augmented dataset and compare eval vs original baseline  
      Command: `scripts/cabinet_experiment.sh train --epochs 200`

> When you add new experiments, follow the same pattern: one line summary + the **exact command** you plan to run.

### In Progress

- [ ] 2026-03-04: Train a baseline action-chunking BC policy with default settings (200 epochs, K=10, obs_horizon=5)  
      Command: `scripts/cabinet_experiment.sh train`  
      Started: 2026-03-04

### Done

(none yet)

> Template — copy this when recording a completed run:
>
> - [x] YYYY-MM-DD: Short description.  
>       Command: `scripts/cabinet_experiment.sh ...`  
>       Result: key metrics, checkpoint path, notes.

## Critical Pitfalls / Learnings

Add entries as: `YYYY-MM-DD - impact - what happened - mitigation`.
Only keep items that affect experiment correctness, reproducibility, or major runtime/cost.

Examples (replace with real cabinet-door issues as you discover them):

- 2026-03-04 - high impact - pip failed with `Invalid version: '2.2.5 2'` when installing or listing packages - mitigation: venv metadata was corrupted; recreate with `rm -rf .venv` then `./install.sh` (see README Troubleshooting).
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
