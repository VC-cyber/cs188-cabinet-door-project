# CS188 Final Project Report

## Title
OpenCabinet with Low-Dim Behavior Cloning + Handle-Augmented State

## Team
- Team members: [Fill in names]
- Course: COM SCI 188 (Winter 2026)

---

## 1. Problem Definition

The goal is to train an imitation learning policy for RoboCasa's OpenCabinet task using provided demonstrations. The robot must open cabinet doors across diverse kitchen layouts and styles.

Key challenge: with only ~107 demonstrations, naive behavior cloning and baseline diffusion setups often fail due to (1) action mapping mismatches, (2) missing task-critical state (cabinet handle location), and (3) high evaluation variance.

Success criterion used for this project:
- Primary reported metric: relaxed task success (any target cabinet door opened beyond threshold), measured over 50 episodes.
- This aligns with instructor guidance for state-based policies.

---

## 2. Method

### 2.1 Data augmentation (task-relevant state)

We added a preprocessing step that replays saved MuJoCo states for each demonstration and extracts:
- `observation.handle_pos`
- `observation.handle_to_eef_pos`
- `observation.door_openness`
- (plus auxiliary `handle_xaxis`, `hinge_direction`)

This is implemented in:
- `cabinet_door_project/05b_augment_handle_data.py`

Output:
- augmented parquet files under `<dataset_path>/augmented/`.

### 2.2 Policy architecture

We implemented a low-dimensional BC policy with a 1D convolutional U-Net backbone:
- Predicts action chunks (horizon-based control), no denoising loop.
- Chunked action setup: `chunk_size=16`, `n_action_steps=8`.

Implemented in:
- `cabinet_door_project/policy_utils.py` (`build_bc_unet_policy`)
- `cabinet_door_project/06_train_policy.py` (`--policy_type bc_unet`)

### 2.3 Training setup

- Episode-level train/val split (no leakage): 85/15.
- Early stopping via validation loss.
- Boundary contamination mitigation by skipping initial sequence window.
- Default BC training settings used:
  - batch size 128
  - learning rate 1e-3
  - max epochs 30-50 (with early stopping logic available)

### 2.4 Action and evaluation correctness fixes

To match demonstration format and avoid silent failure modes:
- Correct dataset-to-env action remap was implemented.
- Gripper/base mode binarization uses threshold 0.0 (not 0.5).
- Eval-time handle feature extraction was aligned with augmentation behavior.
- Relaxed success check replaced strict all-doors-open checks in eval scripts.

Updated files:
- `cabinet_door_project/policy_utils.py`
- `cabinet_door_project/07_evaluate_policy.py`
- `cabinet_door_project/08_visualize_policy_rollout.py`

---

## 3. Evaluation Protocol

- Environment split: `pretrain`
- Rollouts: 50 episodes
- Max timesteps per episode tested: 500 and 600
- Policy checkpoint selection: best validation checkpoint

Commands used:
- Train:
  - `python 06_train_policy.py --policy_type bc_unet --epochs 30`
- Evaluate:
  - `python 07_evaluate_policy.py --checkpoint <best_policy.pt> --num_rollouts 50 --max_steps 500`
  - `python 07_evaluate_policy.py --checkpoint <best_policy.pt> --num_rollouts 50 --max_steps 600`

---

## 4. Results

From overnight ablation logs (`cabinet_door_project/sweep_results/ablations_20260315_014646`):

Baseline config:
- `epochs=30`, `chunk=16`, `exec=8`, `handle_mode=both`
- Seed 0 results:
  - `max_steps=500`: **18/50 = 36.0%**
  - `max_steps=600`: **17/50 = 34.0%**

Interpretation:
- The project met the target threshold of 30% in this run (36%).
- Increasing max steps from 500 to 600 did not improve this checkpoint.

---

## 5. What happened in the ablation sweep

The overnight ablation run started correctly and produced valid baseline results for seed 0. The sweep then stopped early at seed 1 with:
- `ERROR: Dataset not found. Run 04_download_dataset.py first.`

As a result, only the first config/seed pair was fully recorded in `results.csv`.

Impact:
- We have a strong positive result (36%) but incomplete sweep coverage.
- Next step is to re-run the ablation script after fixing sweep robustness to continue on per-seed failures.

---

## 6. Discussion and Reflection

### What worked
- Handle-augmented state substantially improved policy usefulness.
- Correcting action remap and gripper threshold removed major hidden failure modes.
- 1D U-Net with chunked BC provided temporally smoother behavior and higher success than earlier baselines.

### Limitations
- Full multi-seed ablation was interrupted by a sweep runtime issue.
- Current results are from pretrain split only in this report snapshot.
- Reward remained near zero because success was tracked by relaxed hinge criterion rather than sparse environment reward.

### Potential improvements
- Fix sweep script error handling and complete seed ablations.
- Add target split evaluation with the same protocol.
- Compare `handle_mode=both` vs `relative_only` once sweep is fully stable.
- Perform small model-size and chunking sensitivity studies across multiple seeds.

---

## 7. Reproducibility

Code and scripts:
- Training: `cabinet_door_project/06_train_policy.py`
- Evaluation: `cabinet_door_project/07_evaluate_policy.py`
- Augmentation: `cabinet_door_project/05b_augment_handle_data.py`
- Seed sweep: `cabinet_door_project/09_seed_sweep.py`, `09_seed_sweep.sh`
- Ablations: `cabinet_door_project/10_overnight_ablations.sh`

Artifacts:
- Example ablation run:
  - `cabinet_door_project/sweep_results/ablations_20260315_014646/ablations.log`
  - `cabinet_door_project/sweep_results/ablations_20260315_014646/results.csv`

---

## 8. Team Contributions

- [Fill in teammate name]: [data pipeline / augmentation / evaluation]
- [Fill in teammate name]: [modeling / training / ablations]
- [Fill in teammate name]: [analysis / report / website / video]

---

## 9. GenAI and external resources disclosure

This project used AI-assisted coding and analysis tools to accelerate implementation and debugging. All resulting code was tested and integrated by the team.

External codebases and references:
- Starter project: `https://github.com/HoldenGs/cs188-cabinet-door-project`
- RoboCasa / Robosuite dependencies
- Instructor guidance and project notes (`WORKING_SETUP.md` and professor recommendations)

