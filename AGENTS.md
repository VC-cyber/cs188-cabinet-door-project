## Agent Execution Protocol (MANDATORY – Cabinet Door Project)

### 0) One-time setup and permissions

Before running any automation (CLI Agent, daemons, etc.) in this repo:

- **Install project dependencies** (once per machine):
  - From the repo root: `./install.sh`
- **Trust this directory for the Cursor CLI** (required so `agent` can run non-interactively):
  - From the repo root:
    - `agent --trust .`
  - Alternatively, run `agent` interactively once in this directory and accept the prompt.
- **Do NOT pass `--yolo` / `--force` flags from automated scripts.**
  - Trust decisions should be made by a human; automation assumes the repo has already been trusted.

This file defines how agents (and any daemon) should interact with this **cabinet-door OpenCabinet project**.

The main goals are:
- Keep experiment commands short and standardized (low-token prompts).
- Avoid accidental artifact commits.
- Use `PLANS.md` as the single source of truth for what to run next.

### 1) Local edits and validation

- **Edit locally in this repo.**
- **Minimal validation for touched Python files** before you consider the work “ready to run”:
  - For each changed `.py` file, run:
    - `python3 -m py_compile <file.py>`
- **Git hygiene (for humans and tools that are allowed to commit):**
  - Use `git status` and stage only intended files.
  - Do **not** commit `runs/`, large videos, or generated artifacts unless explicitly requested.
  - Keep commits focused and descriptive if/when they are created.

> Tooling note: automated agents in this repo **must not assume** they can always commit/push; defer to higher-level policies or explicit user instructions.

### 2) How to run cabinet-door experiments (canonical CLI)

All training/eval/visualization for this project should be launched through the small wrapper:

- **Canonical entrypoint**
  - `scripts/cabinet_experiment.sh`

This wrapper knows which file to run, so agents do **not** need to remember individual script paths.

- **Train policy (Step 6)**
  - `scripts/cabinet_experiment.sh train [args...]`
  - Examples:
    - `scripts/cabinet_experiment.sh train`
    - `scripts/cabinet_experiment.sh train --epochs 300`
    - `scripts/cabinet_experiment.sh train --config cabinet_door_project/configs/diffusion_policy.yaml`
  - Internally calls: `python cabinet_door_project/06_train_policy.py ...`

- **Evaluate policy (Step 7)**
  - `scripts/cabinet_experiment.sh eval --checkpoint <path> [more args...]`
  - Example:
    - `scripts/cabinet_experiment.sh eval --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt --num_rollouts 20`
  - Internally calls: `python cabinet_door_project/07_evaluate_policy.py ...`

- **Visualize rollouts (Step 8)**
  - `scripts/cabinet_experiment.sh viz --checkpoint <path> [--offscreen ...]`
  - Examples:
    - `scripts/cabinet_experiment.sh viz --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt`
    - `scripts/cabinet_experiment.sh viz --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt --offscreen --video_path /tmp/policy_rollout.mp4`
  - Internally calls: `python cabinet_door_project/08_visualize_policy_rollout.py ...`

When designing prompts for a daemon or automated agent, **prefer these short commands** over hand-writing long `python ...` invocations.

### 3) Coordination with `PLANS.md`

- Treat `PLANS.md` as a **living tracker for this cabinet-door project**:
  - All planned experiments go under `TODO`.
  - When work starts, move an item to `In Progress`.
  - When finished, move it to `Done` with:
    - Date
    - Exact command used (ideally via `scripts/cabinet_experiment.sh ...`)
    - Key metrics (e.g., success rate, average reward)
- Any **important pitfalls or learnings** (simulation issues, environment quirks, training failure modes) must be written to the `Critical Pitfalls / Learnings` section in `PLANS.md`.

Automated agents should:
- **Read `PLANS.md` first** to choose the next task.
- Prefer running a **single well-scoped experiment command** per cycle.
- Append or update the corresponding entry in `PLANS.md` when the run completes.

### 4) Runtime environment assumptions

- Experiments are expected to run **locally** unless otherwise specified.
- If you later introduce a remote/VM workflow for this project, document:
  - Which helper scripts to use.
  - Required conda/env setup.
  - How logs and artifacts are retrieved.
  - Any tmux or scheduling conventions.
- Update this file accordingly so daemons and agents can follow the same protocol.

### 5) Repo references

- Cabinet-door training/eval/visualization scripts:
  - `cabinet_door_project/06_train_policy.py`
  - `cabinet_door_project/07_evaluate_policy.py`
  - `cabinet_door_project/08_visualize_policy_rollout.py`
- Environment / convenience helper:
  - `scripts/env_cabinet.sh` (must be sourced: `source scripts/env_cabinet.sh`)
- Experiment wrapper CLI:
  - `scripts/cabinet_experiment.sh`
- Project plan and experiment log:
  - `PLANS.md`

### 6) AI-driven daemon with Cursor CLI

This repo supports an **AI-driven planning loop** on top of the shell daemon:

- **Low-level executor (shell daemon)**
  - `scripts/cabinet_daemon.sh`
  - Behavior:
    - Reads `PLANS.md` and finds the first unchecked TODO with a
      `Command: scripts/cabinet_experiment.sh ...` line.
    - Uses `.cabinet_daemon.lock` to avoid overlapping runs; if a PID in the
      lock file is alive, it sleeps and does not start a new job.
    - Executes at most one experiment per `--once` call:
      - `scripts/cabinet_daemon.sh --once`

- **AI planner (Cursor CLI)**
  - `scripts/ai_cursor_daemon.py`
  - This script assumes:
    - The Cursor CLI `agent` command is installed and on `PATH`.
    - It runs from the repo root (or any directory inside the repo).
  - High-level loop:
    1. Call `agent` with a tight prompt that:
       - Reads `AGENTS.md` and `PLANS.md`.
       - Edits **only** `PLANS.md` to:
         - Maintain the cabinet-door plan structure.
         - Add/reorder TODOs based on previous results and README suggestions.
         - Move items between TODO / In Progress / Done with short notes.
         - Ensure the first unchecked TODO with a `Command:` line is the
           next experiment to run.
       - Does **not** run shell commands itself.
    2. After Cursor finishes editing, call:
       - `scripts/cabinet_daemon.sh --once`
       which actually executes the newly chosen experiment (if any).
    3. Sleep for a configurable interval and repeat.

- **Running the AI daemon**
  - Single cycle (plan + run once):
    - `python scripts/ai_cursor_daemon.py --once`
  - Continuous loop (e.g., every 30 minutes) under `tmux`:
    - `tmux new -s ai-cabinet-daemon`
    - Inside tmux:
      - `python scripts/ai_cursor_daemon.py --interval 1800`
    - Detach with `Ctrl-b d`, reattach with `tmux attach -t ai-cabinet-daemon`.

- **Where results live**
  - All experiment planning and results should be captured in `PLANS.md`:
    - New experiments: added under `TODO` with a precise `Command:` line.
    - In-progress work: moved to `In Progress`.
    - Completed runs: moved to `Done` with:
      - Date (UTC)
      - Exact command
      - Key metrics (e.g., success rate, avg reward, losses)
      - Brief notes on behavior/failures/next tweaks.
  - Git history (e.g., on a branch like `venkat`) provides an additional
    audit trail for how the automation and plans evolved over time.

### 7) Git update helper (venkat branch)

When it is appropriate to commit and push changes (for example after a
meaningful batch of experiments or code improvements), use the helper:

- `scripts/git_update_venkat.sh "short commit message"`

Behavior:
- Runs `python3 -m py_compile` on modified `.py` files.
- Stages tracked changes plus new whitelisted files:
  - `AGENTS.md`, `PLANS.md`
  - `scripts/*.sh`, `scripts/*.py`
  - `cabinet_door_project/*.py`
  - `cabinet_door_project/configs/diffusion_policy.yaml`
- Creates a commit with the given message.
- Pushes the current branch to `origin` (intended for the `venkat` branch).

Automated agents should:
- Only call this helper when explicitly authorized by the user or higher-level
  workflow policy.
- Prefer descriptive commit messages indicating what changed
  (e.g., `"Log baseline eval results"` or `"Add K=8 action-horizon sweep"`).

