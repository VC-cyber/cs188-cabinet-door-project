# OpenCabinet Implementation Status

This document summarizes all major changes implemented so far to improve OpenCabinet performance toward the 30% success target.

## Goal and references

- Goal: improve policy performance to >= 30% success.
- Guidance sources used:
  - `WORKING_SETUP.md`
  - Professor notes (handle augmentation, 1D U-Net, relaxed success).

## What was implemented

### 1) Dataset augmentation with handle features

- Added: `cabinet_door_project/05b_augment_handle_data.py`
- Purpose:
  - Replays episode states from `extras/episode_*/states.npz`
  - Extracts per-step handle features and writes augmented parquet files.
- Added columns:
  - `observation.handle_pos`
  - `observation.handle_to_eef_pos`
  - `observation.door_openness`
  - `observation.handle_xaxis`
  - `observation.hinge_direction`
- Output location:
  - `<dataset_path>/augmented/*.parquet`

### 2) Unified dataset path helper

- Added helper in `cabinet_door_project/policy_utils.py`:
  - `get_dataset_path()`
- Used by training and augmentation scripts to avoid path drift.

### 3) Data loader support for real parquet schema

- Updated `load_dataset_arrays()` in `cabinet_door_project/policy_utils.py`:
  - Prefers `<dataset>/augmented` if present.
  - Supports both schema variants:
    - split columns (`state.*`, `action.*`)
    - packed columns (`observation.state`, `action`)
- This fixed runtime failures where no state-action pairs were detected.

### 4) Action mapping fixes (critical)

- Implemented/updated:
  - `dataset_action_to_env_action()` in `policy_utils.py`
- Correct mapping now matches `WORKING_SETUP.md`:
  - Dataset: `[base(3), torso(1), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]`
  - Env: `[eef_pos(3), eef_rot(3), gripper(1), base(3), torso(1), base_mode(1)]`
- Gripper and base mode are binarized at threshold `0.0`.

### 5) Eval-time handle observation wrapper

- Added in `policy_utils.py`:
  - `HandleObservationWrapper`
  - internal handle extraction helper
- Used so eval observations include:
  - `robot0_handle_pos`
  - `robot0_handle_to_eef_pos`
- This ensures eval has the same feature family as augmented training data.

### 6) Eval/train feature alignment improvements

- Updated `ROBOSUITE_STATE_KEYS` in `policy_utils.py` to use full 16D proprio:
  - `robot0_base_pos` (3)
  - `robot0_base_quat` (4)
  - `robot0_base_to_eef_pos` (3)
  - `robot0_base_to_eef_quat` (4)
  - `robot0_gripper_qpos` (2)
- Combined with handle features, eval state becomes 22D to match current training inputs.

### 7) Handle vector sign consistency

- Updated sign convention to match `WORKING_SETUP.md`:
  - `handle_to_eef = eef_pos - handle_pos`
- Applied in:
  - `05b_augment_handle_data.py`
  - eval wrapper helper in `policy_utils.py`

### 8) Relaxed success criterion (one door open)

- Added `check_success_relaxed(env)` in `policy_utils.py`.
- Replaced strict success checks in:
  - `07_evaluate_policy.py`
  - `08_visualize_policy_rollout.py`
- Criteria: any relevant door hinge sufficiently open (approx >= 0.3 rad equivalent).

### 9) BC 1D U-Net policy path

- Added model in `policy_utils.py`:
  - `build_bc_unet_policy(...)`
- Added checkpoint loading support for `policy_type == "bc_unet"`.
- Training script updated (`06_train_policy.py`) with:
  - `bc_unet` policy option
  - episode-level train/val split
  - early stopping
  - chunking defaults aligned with notes (`chunk_size=16`, `n_action_steps=8`)
  - boundary contamination mitigation (skipping initial chunk window)

### 10) Evaluation and rollout wiring updates

- `07_evaluate_policy.py`:
  - supports `bc_unet`
  - uses `dataset_action_to_env_action()`
  - uses `HandleObservationWrapper` when needed
  - uses `check_success_relaxed()`
  - default rollouts set to 50
- `08_visualize_policy_rollout.py`:
  - same core wiring as evaluation (action mapping + relaxed success + wrapper + bc_unet support)

### 11) Seed sweep runner

- Added: `cabinet_door_project/09_seed_sweep.py`
- Function:
  - sweeps seeds overnight
  - trains + evaluates each seed
  - writes full output and summary to timestamped text file
  - identifies best seed by success rate.
- Update requested by user:
  - default eval max timesteps now set to `600` in seed sweep
  - passed via `--max_steps` into `07_evaluate_policy.py`.

## Why reward appears as 0.0 while success is non-zero

- Current eval success is based on relaxed success criterion.
- Env reward signal in this setup is sparse/strict and can remain 0.0 even when relaxed success counts an episode as successful.
- Therefore `reward=0.0` with non-zero success can be expected.

## Most recent observed result

- Example run reported:
  - `11/50` successes
  - `22.0%` success rate
  - policy: `bc_unet`
  - state dim: `22`
  - action dim: `12`

## Remaining gap to 30%

- Current state is close to target and now stable enough for seed sweeping.
- Primary next step:
  - run `09_seed_sweep.py` overnight
  - pick best seed/checkpoint by `Success rate`.

