"""
Step 8: Visualize a Policy Rollout
=====================================
Loads a trained policy checkpoint from 06_train_policy.py and runs it
live in the OpenCabinet environment so you can watch the robot.

Supports both the simple MLP baseline and the diffusion policy
with action chunking.

Two rendering modes:
  On-screen  (default)  — interactive MuJoCo viewer window, real-time
  Off-screen (--offscreen) — renders to a video file, works without a display

Usage:
    # Watch live in a window (WSL/Linux) + save video
    python 08_visualize_policy_rollout.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt

    # Save to video only (no display needed)
    python 08_visualize_policy_rollout.py --checkpoint ... --offscreen

    # Mac users must use mjpython for the on-screen window
    mjpython 08_visualize_policy_rollout.py --checkpoint ...
"""

import os
import sys

# ── Rendering mode detection ────────────────────────────────────────────
_OFFSCREEN = "--offscreen" in sys.argv

if _OFFSCREEN:
    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
else:
    if sys.platform == "linux" and "__TELEOP_DISPLAY_OK" not in os.environ:
        _env = dict(os.environ)
        _changed = False
        if _env.get("WAYLAND_DISPLAY"):
            if not _env.get("DISPLAY", "").startswith(":"):
                _env["DISPLAY"] = ":0"
                _changed = True
            if _env.get("GALLIUM_DRIVER") != "llvmpipe":
                _env["GALLIUM_DRIVER"] = "llvmpipe"
                _changed = True
            if _env.get("MESA_GL_VERSION_OVERRIDE") != "4.5":
                _env["MESA_GL_VERSION_OVERRIDE"] = "4.5"
                _changed = True
        if _changed:
            _env["__TELEOP_DISPLAY_OK"] = "1"
            os.execve(sys.executable, [sys.executable] + sys.argv, _env)
        else:
            os.environ["__TELEOP_DISPLAY_OK"] = "1"
# ────────────────────────────────────────────────────────────────────────

import argparse
import time

import numpy as np
import robocasa  # noqa: F401
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

from policy_utils import (
    extract_state_from_obs,
    load_policy_checkpoint,
    normalize,
    denormalize,
)


def _predict_action(model, obs, ckpt, state_norm, action_norm, action_buffer, device):
    """Get the next action from the policy, handling both MLP and diffusion types."""
    import torch

    policy_type = ckpt.get("policy_type", "mlp")

    if policy_type == "diffusion" and len(action_buffer) > 0:
        return action_buffer.pop(0), action_buffer

    state = extract_state_from_obs(obs)

    if policy_type == "diffusion":
        state_n = normalize(state.reshape(1, -1), state_norm)
        with torch.no_grad():
            state_t = torch.from_numpy(state_n).to(device)
            chunk = model.sample(state_t)  # (1, chunk_size, action_dim)
        chunk_np = denormalize(chunk[0], action_norm)
        n_action_steps = ckpt.get("n_action_steps", 4)
        new_buffer = list(chunk_np[:n_action_steps])
        action = new_buffer.pop(0)
        return action, new_buffer
    else:
        state_dim = ckpt["state_dim"]
        if len(state) < state_dim:
            state = np.pad(state, (0, state_dim - len(state)))
        elif len(state) > state_dim:
            state = state[:state_dim]
        with torch.no_grad():
            s_t = torch.from_numpy(state).unsqueeze(0).to(device)
            action = model(s_t).cpu().numpy().squeeze(0)
        return action, action_buffer


def _pad_action(action, env_dim):
    if len(action) < env_dim:
        return np.pad(action, (0, env_dim - len(action)))
    elif len(action) > env_dim:
        return action[:env_dim]
    return action


# ── On-screen rollout ────────────────────────────────────────────────────

def run_onscreen(model, ckpt, state_norm, action_norm, args):
    import torch
    device = next(model.parameters()).device

    env = robosuite.make(
        env_name="OpenCabinet",
        robots="PandaOmron",
        controller_configs=load_composite_controller_config(robot="PandaOmron"),
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="robot0_frontview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
    )
    env = VisualizationWrapper(env)

    successes = 0
    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep+1}/{args.num_episodes} ---")
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")
        print(f"  Task:   {lang}")
        print(f"  Layout: {env.layout_id}   Style: {env.style_id}")

        success = False
        hold_count = 0
        action_buffer = []

        for step in range(args.max_steps):
            action, action_buffer = _predict_action(
                model, obs, ckpt, state_norm, action_norm, action_buffer, device
            )
            action = _pad_action(action, env.action_dim)
            obs, reward, done, info = env.step(action)

            if step % 20 == 0:
                is_open = env._check_success()
                status = "cabinet OPEN" if is_open else "in progress"
                print(f"  step {step:4d}  reward={reward:+.3f}  [{status}]")

            if env._check_success():
                hold_count += 1
                if hold_count >= 15:
                    success = True
                    break
            else:
                hold_count = 0

            time.sleep(1.0 / args.max_fr)

        result = "SUCCESS" if success else "did not open cabinet"
        print(f"\n  Result: {result}")
        if success:
            successes += 1

    env.close()
    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


# ── Off-screen rollout ───────────────────────────────────────────────────

def run_offscreen(model, ckpt, state_norm, action_norm, args):
    import torch
    import imageio
    from robocasa.utils.env_utils import create_env

    device = next(model.parameters()).device
    video_dir = os.path.dirname(args.video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    cam_h, cam_w = 512, 768
    successes = 0
    all_frames = []

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep+1}/{args.num_episodes} ---")
        env = create_env(
            env_name="OpenCabinet",
            render_onscreen=False,
            seed=args.seed + ep,
            camera_widths=cam_w,
            camera_heights=cam_h,
        )
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")
        print(f"  Task:   {lang}")
        print(f"  Layout: {env.layout_id}   Style: {env.style_id}")

        success = False
        hold_count = 0
        action_buffer = []
        ep_frames = []

        for step in range(args.max_steps):
            action, action_buffer = _predict_action(
                model, obs, ckpt, state_norm, action_norm, action_buffer, device
            )
            action = _pad_action(action, env.action_dim)
            obs, reward, done, info = env.step(action)

            frame = env.sim.render(
                height=cam_h, width=cam_w, camera_name="robot0_agentview_center"
            )[::-1]
            ep_frames.append(frame)

            if step % 20 == 0:
                is_open = env._check_success()
                status = "cabinet OPEN" if is_open else "in progress"
                print(f"  step {step:4d}  reward={reward:+.3f}  [{status}]")

            if env._check_success():
                hold_count += 1
                if hold_count >= 15:
                    success = True
                    break
            else:
                hold_count = 0

        result = "SUCCESS" if success else "did not open cabinet"
        print(f"  Result: {result}  ({len(ep_frames)} frames)")
        if success:
            successes += 1
        all_frames.extend(ep_frames)
        env.close()

    print(f"\nWriting {len(all_frames)} frames to {args.video_path} ...")
    with imageio.get_writer(args.video_path, fps=args.fps) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    print(f"Video saved: {args.video_path}")
    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained policy rollout in OpenCabinet"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="/tmp/cabinet_policy_checkpoints/best_policy.pt",
        help="Path to policy checkpoint (.pt) saved by 06_train_policy.py",
    )
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--offscreen", action="store_true", help="Render to video file")
    parser.add_argument("--video_path", type=str, default="/tmp/policy_rollout.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max_fr", type=int, default=20, help="On-screen playback rate cap")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Policy Rollout Visualizer")
    print("=" * 60)
    print()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required.  Run: pip install torch")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Train a policy first with:  python 06_train_policy.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, state_norm, action_norm = load_policy_checkpoint(args.checkpoint, device)

    policy_type = ckpt.get("policy_type", "mlp")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Policy type: {policy_type}")
    print(f"  Epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f}")
    print(f"  State dim: {ckpt['state_dim']},  Action dim: {ckpt['action_dim']}")
    if policy_type == "diffusion":
        print(f"  Chunk size: {ckpt['chunk_size']}, Action steps: {ckpt['n_action_steps']}")
    print(f"  Device: {device}")
    print()

    mode = "off-screen (video)" if args.offscreen else "on-screen (viewer window)"
    print(f"Mode:     {mode}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps/ep: {args.max_steps}")
    if args.offscreen:
        print(f"Output:   {args.video_path}")
    print()

    if args.offscreen:
        run_offscreen(model, ckpt, state_norm, action_norm, args)
    else:
        print("Opening viewer window...")
        print("  Tip: orbit the camera with the mouse to see the gripper.\n")
        run_onscreen(model, ckpt, state_norm, action_norm, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
