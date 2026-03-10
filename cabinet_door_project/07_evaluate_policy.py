"""
Step 7: Evaluate a Trained Policy
===================================
Runs a trained policy in the OpenCabinet environment and reports
success rate across multiple episodes and kitchen scenes.

Supports both the simple MLP baseline and the diffusion policy
with action chunking.

Usage:
    # Evaluate diffusion policy
    python 07_evaluate_policy.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt

    # Evaluate with more episodes
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --num_rollouts 50

    # Evaluate on target (held-out) kitchen scenes
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --split target

    # Save evaluation video
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --video_path /tmp/eval.mp4
"""

import argparse
import os
import sys

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

import robocasa  # noqa: F401
from robocasa.utils.env_utils import create_env

from policy_utils import (
    extract_state_from_obs,
    load_policy_checkpoint,
    normalize,
    denormalize,
)


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_evaluation(model, ckpt, state_norm, action_norm,
                   num_rollouts, max_steps, split, video_path, seed):
    """Run evaluation rollouts and collect statistics."""
    import torch
    import imageio

    device = next(model.parameters()).device
    policy_type = ckpt.get("policy_type", "mlp")
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    chunk_size = ckpt.get("chunk_size", 1)
    n_action_steps = ckpt.get("n_action_steps", 1)

    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=seed,
        split=split,
        camera_widths=256,
        camera_heights=256,
    )

    video_writer = None
    if video_path:
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)

    results = {"successes": [], "episode_lengths": [], "rewards": []}

    for ep in range(num_rollouts):
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")

        ep_reward = 0.0
        success = False
        action_buffer = []

        for step in range(max_steps):
            if policy_type == "diffusion" and len(action_buffer) > 0:
                action = action_buffer.pop(0)
            else:
                state = extract_state_from_obs(obs)

                if policy_type == "diffusion":
                    state_n = normalize(state.reshape(1, -1), state_norm)
                    with torch.no_grad():
                        state_t = torch.from_numpy(state_n).to(device)
                        chunk = model.sample(state_t)  # (1, chunk_size, action_dim)
                    chunk_np = denormalize(chunk[0], action_norm)
                    action_buffer = list(chunk_np[:n_action_steps])
                    action = action_buffer.pop(0)
                else:
                    with torch.no_grad():
                        s_t = torch.from_numpy(state).unsqueeze(0).to(device)
                        action = model(s_t).cpu().numpy().squeeze(0)

            env_action_dim = env.action_dim
            if len(action) < env_action_dim:
                action = np.pad(action, (0, env_action_dim - len(action)))
            elif len(action) > env_action_dim:
                action = action[:env_action_dim]

            obs, reward, done, info = env.step(action)
            ep_reward += reward

            if video_writer is not None:
                frame = env.sim.render(
                    height=512, width=768, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(frame)

            if env._check_success():
                success = True
                break

        results["successes"].append(success)
        results["episode_lengths"].append(step + 1)
        results["rewards"].append(ep_reward)

        status = "SUCCESS" if success else "FAIL"
        print(
            f"  Episode {ep+1:3d}/{num_rollouts}: {status:7s} "
            f"(steps={step+1:4d}, reward={ep_reward:.1f}) "
            f'layout={env.layout_id}, style={env.style_id}, task="{lang}"'
        )

    if video_writer:
        video_writer.close()
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained OpenCabinet policy")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to policy checkpoint (.pt file)",
    )
    parser.add_argument("--num_rollouts", type=int, default=20, help="Number of episodes")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument(
        "--split", type=str, default="pretrain", choices=["pretrain", "target"],
        help="Kitchen scene split to evaluate on",
    )
    parser.add_argument("--video_path", type=str, default=None, help="Save evaluation video")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required.  pip install torch")
        sys.exit(1)

    print("=" * 60)
    print("  OpenCabinet - Policy Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, ckpt, state_norm, action_norm = load_policy_checkpoint(args.checkpoint, device)

    policy_type = ckpt.get("policy_type", "mlp")
    print(f"Policy type:   {policy_type}")
    print(f"Trained epoch: {ckpt['epoch']}, loss={ckpt['loss']:.6f}")
    print(f"State dim:     {ckpt['state_dim']}, Action dim: {ckpt['action_dim']}")
    if policy_type == "diffusion":
        print(f"Chunk size:    {ckpt['chunk_size']}, Action steps: {ckpt['n_action_steps']}")
        print(f"Diff steps:    {ckpt['n_diffusion_steps']}")

    print_section(f"Evaluating on {args.split} split ({args.num_rollouts} episodes)")

    results = run_evaluation(
        model=model,
        ckpt=ckpt,
        state_norm=state_norm,
        action_norm=action_norm,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        split=args.split,
        video_path=args.video_path,
        seed=args.seed,
    )

    print_section("Evaluation Results")

    num_success = sum(results["successes"])
    success_rate = num_success / args.num_rollouts * 100
    avg_length = np.mean(results["episode_lengths"])
    avg_reward = np.mean(results["rewards"])

    print(f"  Split:          {args.split}")
    print(f"  Episodes:       {args.num_rollouts}")
    print(f"  Successes:      {num_success}/{args.num_rollouts}")
    print(f"  Success rate:   {success_rate:.1f}%")
    print(f"  Avg ep length:  {avg_length:.1f} steps")
    print(f"  Avg reward:     {avg_reward:.3f}")

    if args.video_path:
        print(f"\n  Video saved to: {args.video_path}")

    print_section("Performance Context")
    print(
        "Expected success rates from the RoboCasa benchmark:\n"
        "\n"
        "  Method                 | Pretrain | Target\n"
        "  -----------------------|----------|-------\n"
        "  Random actions         |    ~0%   |   ~0%\n"
        "  Simple MLP (baseline)  |    ~0%   |   ~0%\n"
        "  Diffusion (low-dim)    |   ~5-15% |  ~3-10%\n"
        "  Diffusion Policy (vis) |  ~30-60% | ~20-50%\n"
        "  pi-0                   |  ~40-70% | ~30-60%\n"
    )


if __name__ == "__main__":
    main()
