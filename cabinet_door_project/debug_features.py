"""
Quick diagnostic: verify that training features match inference features.

Run this after training to check if the state vectors the model sees at
inference time match what it saw during training.

Usage:
    python debug_features.py
"""

import os
import sys

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np
import torch


def main():
    ckpt_path = "/tmp/cabinet_policy_checkpoints/best_policy.pt"
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint at {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print("=" * 60)
    print("  Feature Alignment Diagnostic")
    print("=" * 60)

    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Policy type: {ckpt.get('policy_type', 'simple')}")
    print(f"State dim: {ckpt['state_dim']}, Action dim: {ckpt['action_dim']}")
    print(f"State keys in ckpt: {ckpt.get('state_keys', 'NOT SAVED')}")

    state_mean = ckpt["state_mean"]
    state_std = ckpt["state_std"]
    action_mean = ckpt["action_mean"]
    action_std = ckpt["action_std"]

    print(f"\nTraining state_mean: {state_mean}")
    print(f"Training state_std:  {state_std}")
    print(f"Training action_mean: {action_mean}")
    print(f"Training action_std:  {action_std}")

    # ---- Inspect the parquet data ----
    print("\n" + "=" * 60)
    print("  Parquet Data Inspection")
    print("=" * 60)

    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path
    import pyarrow.parquet as pq

    ds_path = get_ds_path("OpenCabinet", source="human")
    data_dir = os.path.join(ds_path, "data")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(ds_path, "lerobot", "data")

    chunk_dir = os.path.join(data_dir, "chunk-000")
    pf = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".parquet"))[0]
    table = pq.read_table(os.path.join(chunk_dir, pf))
    df = table.to_pandas()

    print(f"\nParquet file: {pf}")
    print(f"Columns ({len(df.columns)}):")
    for c in df.columns:
        sample = df[c].iloc[0]
        if isinstance(sample, np.ndarray):
            print(f"  {c:50s} ndarray shape={sample.shape}")
        else:
            print(f"  {c:50s} {type(sample).__name__} = {sample}")

    state_cols = [c for c in df.columns if c.startswith("observation.state")]
    print(f"\nState columns: {state_cols}")

    row = df.iloc[0]
    if state_cols:
        print("\nFirst row state values:")
        for c in state_cols:
            val = row[c]
            if isinstance(val, np.ndarray):
                print(f"  {c}: {val}")
            else:
                print(f"  {c}: {val}")

    # ---- Inspect env observations ----
    print("\n" + "=" * 60)
    print("  Environment Observation Inspection")
    print("=" * 60)

    from robocasa.utils.env_utils import create_env
    from policy import extract_state, STATE_KEYS

    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=42,
        camera_widths=64,
        camera_heights=64,
    )
    obs = env.reset()

    print(f"\nSTATE_KEYS: {STATE_KEYS}")
    print(f"\nAll non-image observation keys:")
    for k in sorted(obs.keys()):
        v = obs[k]
        if isinstance(v, np.ndarray) and not k.endswith("_image"):
            print(f"  {k:40s} shape={str(v.shape):10s} val={v}")

    state_vec = extract_state(obs, 16, state_keys=STATE_KEYS)
    print(f"\nExtracted state (STATE_KEYS order): {state_vec}")

    # ---- Compare ----
    print("\n" + "=" * 60)
    print("  Feature-by-Feature Comparison")
    print("=" * 60)
    print(f"\n{'Dim':>4s}  {'Env Value':>12s}  {'Train Mean':>12s}  {'Train Std':>12s}  {'Z-score':>8s}  Status")
    print("-" * 72)

    mismatches = 0
    for i in range(min(16, len(state_vec))):
        z = abs(state_vec[i] - state_mean[i]) / max(state_std[i], 1e-6)
        status = "OK" if z < 5 else "MISMATCH"
        if status == "MISMATCH":
            mismatches += 1
        print(f"  {i:2d}   {state_vec[i]:+12.4f}  {state_mean[i]:+12.4f}  {state_std[i]:12.4f}  {z:8.2f}  {status}")

    if mismatches > 0:
        print(f"\n*** {mismatches} dimensions have z-score > 5 ***")
        print("This means the env observations DON'T match the training data ordering.")
        print("The parquet's observation.state column stores features in a different")
        print("order than STATE_KEYS. We need to fix the ordering.")
    else:
        print(f"\nAll dimensions within expected range. Feature alignment looks correct.")
        print("If the policy still fails, it may be a model capacity / training issue.")

    # ---- Test model prediction ----
    print("\n" + "=" * 60)
    print("  Model Prediction Test")
    print("=" * 60)

    from policy import load_policy_from_checkpoint, ActionChunkingInference, reorder_action_for_env

    model, ckpt2 = load_policy_from_checkpoint(ckpt_path, torch.device("cpu"))
    policy_type = ckpt2.get("policy_type", "simple")

    if policy_type == "action_chunking":
        inference = ActionChunkingInference(model, ckpt2, torch.device("cpu"))

        actions = []
        for step in range(5):
            state_vec = extract_state(obs, 16, state_keys=STATE_KEYS)
            action = inference.predict(state_vec)
            actions.append(action.copy())
            env_action = reorder_action_for_env(action)
            obs, _, _, _ = env.step(
                np.pad(env_action, (0, max(0, env.action_dim - len(env_action))))[:env.action_dim]
            )

        print("\nFirst 5 predicted actions:")
        for i, a in enumerate(actions):
            print(f"  step {i}: mag={np.abs(a).mean():.4f}  val={a}")

        unique_mags = len(set(f"{np.abs(a).mean():.4f}" for a in actions))
        if unique_mags == 1:
            print("\n*** All actions have identical magnitude -- model is outputting a constant! ***")
            print("This confirms feature mismatch or model collapse.")
        else:
            print(f"\n{unique_mags} distinct action magnitudes -- model IS responding to input.")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
