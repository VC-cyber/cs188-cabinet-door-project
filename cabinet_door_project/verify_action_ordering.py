"""
Verify action ordering by testing known actions in the env.

Test 1: Apply a pure arm-forward action and check if eef moves.
Test 2: Replay first demo episode's actions through env with reordering.
"""
import os, sys
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np


def test_known_actions():
    """Apply a known action in env order and verify the arm responds."""
    import robocasa  # noqa
    from robocasa.utils.env_utils import create_env

    env = create_env(env_name="OpenCabinet", render_onscreen=False, seed=42,
                     camera_widths=64, camera_heights=64)
    obs = env.reset()

    robot = env.robots[0]
    cc = robot.composite_controller
    print("=== ENV ACTION ORDERING ===")
    for part in cc.part_controllers:
        s, e = cc._action_split_indexes[part]
        print(f"  {part:25s} [{s}:{e}]")
    print(f"  Total: {env.action_dim}")

    eef_before = obs["robot0_base_to_eef_pos"].copy()
    print(f"\nEEF before: {eef_before}")

    # Test: arm_pos_x = +1 (forward), mode = -1 (arm control)
    action = np.zeros(env.action_dim)
    action[0] = 1.0   # arm pos x
    action[11] = -1.0  # mode = arm
    print(f"\nApplying env-order action: arm_pos_x=1.0, mode=-1")

    for _ in range(10):
        obs, _, _, _ = env.step(action)

    eef_after = obs["robot0_base_to_eef_pos"].copy()
    delta = eef_after - eef_before
    print(f"EEF after:  {eef_after}")
    print(f"EEF delta:  {delta}")
    moved = np.abs(delta).max() > 0.01
    print(f"Arm moved:  {moved}")

    if moved:
        print("ENV ACTION ORDERING CONFIRMED: index 0 = arm_pos_x")
    else:
        print("WARNING: Arm did NOT move with action[0]=1.0!")

    env.close()
    return moved


def test_reorder_with_demo():
    """Replay demo actions with our reordering and check trajectory."""
    import robocasa  # noqa
    from robocasa.utils.env_utils import create_env
    from robocasa.utils.dataset_registry_utils import get_ds_path
    import pyarrow.parquet as pq
    from policy import reorder_action_for_env, PARQUET_TO_ENV_ACTION

    ds_path = get_ds_path("OpenCabinet", source="human")
    chunk_dir = os.path.join(ds_path, "data", "chunk-000")
    pf = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".parquet"))[0]
    table = pq.read_table(os.path.join(chunk_dir, pf))
    df = table.to_pandas()

    actions = np.stack(df["action"].values)
    print(f"\n=== DEMO REPLAY TEST ===")
    print(f"Demo file: {pf}, {len(actions)} steps")
    print(f"PARQUET_TO_ENV_ACTION: {PARQUET_TO_ENV_ACTION}")

    env = create_env(env_name="OpenCabinet", render_onscreen=False, seed=42,
                     camera_widths=64, camera_heights=64)
    obs = env.reset()

    print(f"\n--- WITH reordering (parquet→env) ---")
    eef_start = obs["robot0_base_to_eef_pos"].copy()
    print(f"EEF start: {eef_start}")

    for i in range(min(50, len(actions))):
        env_action = reorder_action_for_env(actions[i])
        obs, reward, done, info = env.step(env_action)
        if i % 10 == 0:
            eef = obs["robot0_base_to_eef_pos"]
            grip = obs["robot0_gripper_qpos"]
            print(f"  step {i:3d}: eef={eef}, grip={grip}, reward={reward}")

    eef_end = obs["robot0_base_to_eef_pos"].copy()
    delta_reorder = eef_end - eef_start
    print(f"EEF total delta (with reorder): {delta_reorder}, mag={np.linalg.norm(delta_reorder):.4f}")

    env.close()

    # Now try WITHOUT reordering (raw parquet order → env)
    env2 = create_env(env_name="OpenCabinet", render_onscreen=False, seed=42,
                      camera_widths=64, camera_heights=64)
    obs2 = env2.reset()

    print(f"\n--- WITHOUT reordering (raw parquet→env) ---")
    eef_start2 = obs2["robot0_base_to_eef_pos"].copy()
    print(f"EEF start: {eef_start2}")

    for i in range(min(50, len(actions))):
        obs2, reward, done, info = env2.step(actions[i])
        if i % 10 == 0:
            eef = obs2["robot0_base_to_eef_pos"]
            grip = obs2["robot0_gripper_qpos"]
            print(f"  step {i:3d}: eef={eef}, grip={grip}, reward={reward}")

    eef_end2 = obs2["robot0_base_to_eef_pos"].copy()
    delta_no_reorder = eef_end2 - eef_start2
    print(f"EEF total delta (no reorder):   {delta_no_reorder}, mag={np.linalg.norm(delta_no_reorder):.4f}")

    env2.close()

    print(f"\n=== VERDICT ===")
    mag_reorder = np.linalg.norm(delta_reorder)
    mag_no_reorder = np.linalg.norm(delta_no_reorder)
    print(f"Arm movement WITH reorder:    {mag_reorder:.4f}")
    print(f"Arm movement WITHOUT reorder: {mag_no_reorder:.4f}")
    if mag_reorder > mag_no_reorder:
        print("REORDER IS BETTER → parquet order differs from env order")
    elif mag_no_reorder > mag_reorder:
        print("NO REORDER IS BETTER → parquet IS already in env order!")
        print("*** THE REORDER IS WRONG -- REMOVE IT ***")
    else:
        print("Both similar - inconclusive")


if __name__ == "__main__":
    ok = test_known_actions()
    if ok:
        test_reorder_with_demo()
