"""
Step 6: Train a Policy for OpenCabinet
=======================================
Supports two policy types:

  mlp        – Simple MLP baseline (fast, but will almost certainly get 0%).
  diffusion  – Minimal Diffusion Policy with action chunking (recommended).

The diffusion policy properly handles multi-modal demonstrations by
modelling the action distribution with a denoising diffusion process,
and uses action chunking for temporally coherent behaviour.

Usage:
    # Train diffusion policy (recommended — non-zero success rate)
    python 06_train_policy.py --policy_type diffusion --epochs 200

    # Train simple MLP baseline (educational only)
    python 06_train_policy.py --policy_type mlp --epochs 50

    # Print setup instructions for official Diffusion Policy / pi-0 / GR00T
    python 06_train_policy.py --use_diffusion_policy
"""

import argparse
import os
import sys
import yaml

import numpy as np

from policy_utils import (
    build_diffusion_policy,
    build_simple_policy,
    compute_norm_params,
    load_dataset_arrays,
    normalize,
    _norm_to_serializable,
)


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_path():
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path

    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        print("ERROR: Dataset not found. Run 04_download_dataset.py first.")
        sys.exit(1)
    return path


# ── Diffusion Policy training ────────────────────────────────────────────

def train_diffusion_policy(config):
    """Train a Minimal Diffusion Policy with action chunking."""
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("ERROR: PyTorch is required.  pip install torch")
        sys.exit(1)

    print_section("Minimal Diffusion Policy + Action Chunking")

    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}")

    print("\nLoading dataset...")
    states, actions, episode_ids = load_dataset_arrays(dataset_path)
    print(f"  Total timesteps: {len(states)}")
    print(f"  Episodes:        {len(np.unique(episode_ids))}")
    print(f"  State dim:       {states.shape[-1]}")
    print(f"  Action dim:      {actions.shape[-1]}")

    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]
    chunk_size = config.get("chunk_size", 8)
    n_action_steps = config.get("n_action_steps", 4)
    n_diffusion_steps = config.get("n_diffusion_steps", 50)
    hidden_dim = config.get("hidden_dim", 256)
    epochs = config.get("epochs", 200)
    batch_size = config.get("batch_size", 256)
    lr = config.get("learning_rate", 1e-4)

    # ── Normalization ─────────────────────────────────────────────────
    state_norm = compute_norm_params(states)
    action_norm = compute_norm_params(actions)

    states_n = normalize(states, state_norm)
    actions_n = normalize(actions, action_norm)

    # ── Build action-chunked training set ─────────────────────────────
    print("\nBuilding action-chunked training set...")
    state_chunks = []
    action_chunks = []
    for ep_id in np.unique(episode_ids):
        mask = episode_ids == ep_id
        ep_s = states_n[mask]
        ep_a = actions_n[mask]
        for t in range(len(ep_s)):
            chunk = ep_a[t : t + chunk_size]
            if len(chunk) < chunk_size:
                pad = np.tile(chunk[-1:], (chunk_size - len(chunk), 1))
                chunk = np.concatenate([chunk, pad], axis=0)
            state_chunks.append(ep_s[t])
            action_chunks.append(chunk)

    state_chunks = np.array(state_chunks, dtype=np.float32)
    action_chunks = np.array(action_chunks, dtype=np.float32)
    print(f"  Training samples: {len(state_chunks)}")
    print(f"  Chunk size:       {chunk_size}")
    print(f"  Action steps:     {n_action_steps}")

    dataset = TensorDataset(
        torch.from_numpy(state_chunks),
        torch.from_numpy(action_chunks),
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
    )

    # ── Model ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"\nDevice: {device}")

    model = build_diffusion_policy(
        state_dim, action_dim,
        chunk_size=chunk_size,
        hidden_dim=hidden_dim,
        n_diffusion_steps=n_diffusion_steps,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ─────────────────────────────────────────────────
    print_section("Training")
    print(f"  Epochs:          {epochs}")
    print(f"  Batch size:      {batch_size}")
    print(f"  LR:              {lr}")
    print(f"  Diffusion steps: {n_diffusion_steps}")
    print(f"  Hidden dim:      {hidden_dim}")

    checkpoint_dir = config.get("checkpoint_dir", "/tmp/cabinet_policy_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for s_batch, a_batch in dataloader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)

            loss = model.compute_loss(s_batch, a_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:4d}/{epochs}  Loss: {avg_loss:.6f}  LR: {cur_lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_diffusion_ckpt(
                os.path.join(checkpoint_dir, "best_policy.pt"),
                model, optimizer, epoch + 1, best_loss,
                state_dim, action_dim, chunk_size, n_action_steps,
                n_diffusion_steps, hidden_dim, state_norm, action_norm,
            )

    final_path = os.path.join(checkpoint_dir, "final_policy.pt")
    _save_diffusion_ckpt(
        final_path, model, optimizer, epochs, avg_loss,
        state_dim, action_dim, chunk_size, n_action_steps,
        n_diffusion_steps, hidden_dim, state_norm, action_norm,
    )

    print(f"\nTraining complete!")
    print(f"  Best loss:        {best_loss:.6f}")
    print(f"  Best checkpoint:  {os.path.join(checkpoint_dir, 'best_policy.pt')}")
    print(f"  Final checkpoint: {final_path}")

    print_section("Next Steps")
    print(
        "Evaluate your trained policy:\n"
        f"  python 07_evaluate_policy.py --checkpoint {os.path.join(checkpoint_dir, 'best_policy.pt')}\n"
        "\n"
        "Visualize a rollout:\n"
        f"  python 08_visualize_policy_rollout.py --checkpoint {os.path.join(checkpoint_dir, 'best_policy.pt')}\n"
    )


def _save_diffusion_ckpt(path, model, optimizer, epoch, loss,
                         state_dim, action_dim, chunk_size, n_action_steps,
                         n_diffusion_steps, hidden_dim, state_norm, action_norm):
    import torch
    torch.save({
        "policy_type": "diffusion",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "chunk_size": chunk_size,
        "n_action_steps": n_action_steps,
        "n_diffusion_steps": n_diffusion_steps,
        "hidden_dim": hidden_dim,
        "state_norm": _norm_to_serializable(state_norm),
        "action_norm": _norm_to_serializable(action_norm),
    }, path)


# ── Simple MLP training (original baseline) ──────────────────────────────

def train_simple_policy(config):
    """Train a simple MLP behaviour-cloning policy (original baseline)."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("ERROR: PyTorch is required.  pip install torch")
        sys.exit(1)

    print_section("Simple MLP Behavior Cloning (baseline)")

    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}")

    print("\nLoading dataset...")
    states, actions, _ = load_dataset_arrays(dataset_path)
    print(f"  Loaded {len(states)} state-action pairs")
    print(f"  State dim:  {states.shape[-1]}")
    print(f"  Action dim: {actions.shape[-1]}")

    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]

    dataset = TensorDataset(
        torch.from_numpy(states),
        torch.from_numpy(actions),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 64),
        shuffle=True,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = build_simple_policy(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 1e-4))

    epochs = config.get("epochs", 50)
    checkpoint_dir = config.get("checkpoint_dir", "/tmp/cabinet_policy_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print_section("Training")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {config.get('batch_size', 64)}")
    print(f"  LR:         {config.get('learning_rate', 1e-4)}")

    best_loss = float("inf")
    avg_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for s_batch, a_batch in dataloader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            pred = model(s_batch)
            loss = nn.functional.mse_loss(pred, a_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}  Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "policy_type": "mlp",
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "state_dim": state_dim,
                "action_dim": action_dim,
            }, os.path.join(checkpoint_dir, "best_policy.pt"))

    final_path = os.path.join(checkpoint_dir, "final_policy.pt")
    torch.save({
        "policy_type": "mlp",
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }, final_path)

    print(f"\nTraining complete!")
    print(f"  Best loss:        {best_loss:.6f}")
    print(f"  Best checkpoint:  {os.path.join(checkpoint_dir, 'best_policy.pt')}")
    print(f"  Final checkpoint: {final_path}")

    print_section("Note")
    print(
        "The simple MLP baseline is for educational purposes.\n"
        "For a non-zero success rate, train with:\n"
        "  python 06_train_policy.py --policy_type diffusion --epochs 200\n"
    )


# ── Instructions for official repos ──────────────────────────────────────

def print_diffusion_policy_instructions():
    print_section("Official Diffusion Policy Training")
    print(
        "For production-quality policy training, use the official repos:\n"
        "\n"
        "Option A: Diffusion Policy (recommended for single-task)\n"
        "  git clone https://github.com/robocasa-benchmark/diffusion_policy\n"
        "  cd diffusion_policy && pip install -e .\n"
        "\n"
        "  # Train\n"
        "  python train.py \\\n"
        "    --config-name=train_diffusion_transformer_bs192 \\\n"
        "    task=robocasa/OpenCabinet\n"
        "\n"
        "  # Evaluate\n"
        "  python eval_robocasa.py \\\n"
        "    --checkpoint <path-to-checkpoint> \\\n"
        "    --task_set atomic \\\n"
        "    --split target\n"
        "\n"
        "Option B: pi-0 via OpenPi (for foundation model fine-tuning)\n"
        "  git clone https://github.com/robocasa-benchmark/openpi\n"
        "  cd openpi && pip install -e . && pip install -e packages/openpi-client/\n"
        "\n"
        "  XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 python scripts/train.py \\\n"
        "    robocasa_OpenCabinet --exp-name=cabinet_door\n"
        "\n"
        "Option C: GR00T N1.5 (NVIDIA foundation model)\n"
        "  git clone https://github.com/robocasa-benchmark/Isaac-GR00T\n"
        "  cd groot && pip install -e .\n"
        "\n"
        "  python scripts/gr00t_finetune.py \\\n"
        "    --output-dir experiments/cabinet_door \\\n"
        "    --dataset_soup robocasa_OpenCabinet \\\n"
        "    --max_steps 50000\n"
    )


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a policy for OpenCabinet")
    parser.add_argument(
        "--policy_type",
        type=str,
        default="diffusion",
        choices=["mlp", "diffusion"],
        help="Policy architecture (default: diffusion)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--chunk_size", type=int, default=8, help="Action chunk size (diffusion only)")
    parser.add_argument("--n_action_steps", type=int, default=4, help="Steps to execute per chunk")
    parser.add_argument("--n_diffusion_steps", type=int, default=50, help="DDPM diffusion steps")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument(
        "--checkpoint_dir", type=str,
        default="/tmp/cabinet_policy_checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides CLI args)",
    )
    parser.add_argument(
        "--use_diffusion_policy", action="store_true",
        help="Print instructions for the official Diffusion Policy repo",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Policy Training")
    print("=" * 60)

    if args.use_diffusion_policy:
        print_diffusion_policy_instructions()
        return

    if args.config:
        config = load_config(args.config)
    else:
        if args.policy_type == "diffusion":
            config = {
                "epochs": args.epochs or 200,
                "batch_size": args.batch_size or 256,
                "learning_rate": args.lr or 1e-4,
                "checkpoint_dir": args.checkpoint_dir,
                "chunk_size": args.chunk_size,
                "n_action_steps": args.n_action_steps,
                "n_diffusion_steps": args.n_diffusion_steps,
                "hidden_dim": args.hidden_dim,
            }
        else:
            config = {
                "epochs": args.epochs or 50,
                "batch_size": args.batch_size or 64,
                "learning_rate": args.lr or 1e-4,
                "checkpoint_dir": args.checkpoint_dir,
            }

    if args.policy_type == "diffusion":
        train_diffusion_policy(config)
    else:
        train_simple_policy(config)


if __name__ == "__main__":
    main()
