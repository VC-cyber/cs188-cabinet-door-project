"""
Step 6: Train an Action-Chunking Behavior Cloning Policy
=========================================================
Trains a behavior-cloning policy with action chunking for the OpenCabinet
task.  Instead of predicting a single action per timestep, the policy takes
a window of recent observations and predicts the next K actions as a
coherent chunk.

Key improvements over the baseline single-step MLP:
  - Temporal context:  observes the last obs_horizon states (not just one)
  - Action chunking:   predicts action_horizon future actions at once
  - Normalization:     zero-mean unit-variance for states and actions
  - Residual MLP:      LayerNorm + GELU + skip connections
  - Cosine LR:         learning rate annealing for better convergence
  - Gradient clipping: prevents training instabilities
  - Train/val split:   monitors overfitting

Usage:
    python 06_train_policy.py
    python 06_train_policy.py --epochs 300 --action_horizon 16
    python 06_train_policy.py --config configs/diffusion_policy.yaml
    python 06_train_policy.py --use_diffusion_policy   # Print official repo instructions
"""

import argparse
import os
import sys
import yaml

import numpy as np


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_path():
    """Get the path to the OpenCabinet dataset."""
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path

    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        print("ERROR: Dataset not found. Run 04_download_dataset.py first.")
        sys.exit(1)
    return path


def train_policy(config):
    """
    Train an action-chunking behavior cloning policy.

    The policy observes a window of recent states and predicts
    multiple future actions as a single coherent chunk.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset, random_split
    except ImportError:
        print("ERROR: PyTorch is required for training.")
        print("Install with: pip install torch torchvision")
        sys.exit(1)

    from policy import ActionChunkingPolicy, STATE_KEYS

    print_section("Action-Chunking Behavior Cloning")

    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}")

    obs_horizon = config.get("obs_horizon", 5)
    action_horizon = config.get("action_horizon", 10)
    action_exec_horizon = config.get("action_exec_horizon", 5)
    hidden_dim = config.get("hidden_dim", 512)
    n_blocks = config.get("n_blocks", 4)
    dropout = config.get("dropout", 0.1)

    # ----------------------------------------------------------------
    # 1. Episode-aware dataset with sliding windows
    # ----------------------------------------------------------------
    print("\nLoading dataset...")

    class CabinetDemoDataset(Dataset):
        """
        Loads demonstration episodes from LeRobot-format parquet files
        and creates sliding windows of (obs_history, future_actions).
        """

        def __init__(self, dataset_path, obs_horizon, action_horizon, max_episodes=None):
            self.obs_horizon = obs_horizon
            self.action_horizon = action_horizon
            self.windows = []
            self.state_dim = None
            self.action_dim = None

            episodes = self._load_episodes(dataset_path, max_episodes)

            for states, actions in episodes:
                if self.state_dim is None:
                    self.state_dim = states.shape[1]
                    self.action_dim = actions.shape[1]

                T = len(states)
                for t in range(T):
                    obs_seq = self._pad_window(states, t, obs_horizon, side="left")
                    act_seq = self._pad_window(actions, t, action_horizon, side="right")
                    self.windows.append((obs_seq, act_seq))

            if not self.windows:
                print("WARNING: No data found in parquet files.")
                print("Generating synthetic data for pipeline testing...")
                self._generate_synthetic()

            all_s = np.stack([w[0] for w in self.windows]).reshape(-1, self.state_dim)
            all_a = np.stack([w[1] for w in self.windows]).reshape(-1, self.action_dim)
            self.state_mean = all_s.mean(axis=0).astype(np.float32)
            self.state_std = np.clip(all_s.std(axis=0), 1e-6, None).astype(np.float32)
            self.action_mean = all_a.mean(axis=0).astype(np.float32)
            self.action_std = np.clip(all_a.std(axis=0), 1e-6, None).astype(np.float32)

            print(f"Loaded {len(self.windows)} windows from {len(episodes)} episodes")
            print(f"State dim:       {self.state_dim}")
            print(f"Action dim:      {self.action_dim}")
            print(f"Obs horizon:     {obs_horizon}")
            print(f"Action horizon:  {action_horizon}")

        # -- data loading helpers -----------------------------------------

        def _load_episodes(self, dataset_path, max_episodes):
            import pyarrow.parquet as pq

            data_dir = os.path.join(dataset_path, "data")
            if not os.path.exists(data_dir):
                data_dir = os.path.join(dataset_path, "lerobot", "data")
            if not os.path.exists(data_dir):
                raise FileNotFoundError(
                    f"Data directory not found under: {dataset_path}\n"
                    "Run 04_download_dataset.py first."
                )

            chunk_dir = os.path.join(data_dir, "chunk-000")
            if not os.path.exists(chunk_dir):
                raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

            parquet_files = sorted(
                f for f in os.listdir(chunk_dir) if f.endswith(".parquet")
            )
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files in {chunk_dir}")

            episodes = []
            for pf in parquet_files:
                table = pq.read_table(os.path.join(chunk_dir, pf))
                df = table.to_pandas()

                # Determine state columns, preferring STATE_KEYS order
                state_cols = self._resolve_state_cols(df)
                action_cols = [
                    c for c in df.columns
                    if c == "action" or c.startswith("action.")
                ]
                if not state_cols or not action_cols:
                    continue

                ep_col = None
                for candidate in ["episode_index", "episode_id", "episode"]:
                    if candidate in df.columns:
                        ep_col = candidate
                        break

                groups = df.groupby(ep_col) if ep_col else [(0, df)]
                for _, ep_df in groups:
                    ep_s, ep_a = [], []
                    for _, row in ep_df.iterrows():
                        s = self._row_to_vec(row, state_cols)
                        a = self._row_to_vec(row, action_cols)
                        if s is not None and a is not None:
                            ep_s.append(s)
                            ep_a.append(a)
                    if len(ep_s) > 1:
                        episodes.append((
                            np.array(ep_s, dtype=np.float32),
                            np.array(ep_a, dtype=np.float32),
                        ))
                    if max_episodes and len(episodes) >= max_episodes:
                        return episodes

            return episodes

        @staticmethod
        def _resolve_state_cols(df):
            """
            Pick state columns in an order consistent with STATE_KEYS so
            that training features match what extract_state() produces at
            inference time from the live environment.
            """
            all_cols = list(df.columns)

            # 1) Per-key columns (e.g. observation.state.robot0_base_pos)
            ordered = []
            for key in STATE_KEYS:
                matches = [c for c in all_cols if c.endswith(key)]
                if matches:
                    ordered.append(matches[0])
            if ordered:
                return ordered

            # 2) Single observation.state column (array-valued)
            exact = [c for c in all_cols if c == "observation.state"]
            if exact:
                return exact

            # 3) Any observation.state.* columns (sorted for consistency)
            obs_state = sorted(c for c in all_cols if c.startswith("observation.state"))
            if obs_state:
                return obs_state

            # 4) Fallback heuristic
            fallback = [
                c for c in all_cols
                if "gripper" in c or "base" in c or "eef" in c
            ]
            return fallback or []

        @staticmethod
        def _row_to_vec(row, cols):
            parts = []
            for c in cols:
                val = row[c]
                if isinstance(val, np.ndarray):
                    parts.extend(val.flatten().tolist())
                elif isinstance(val, (list, tuple)):
                    parts.extend([float(v) for v in val])
                elif isinstance(val, (int, float, np.floating)):
                    parts.append(float(val))
            return np.array(parts, dtype=np.float32) if parts else None

        @staticmethod
        def _pad_window(data, t, horizon, side="right"):
            """Extract a window from data, padding at the boundary."""
            T = len(data)
            if side == "right":
                end = min(T, t + horizon)
                seq = data[t:end]
                if len(seq) < horizon:
                    pad = np.tile(seq[-1:], (horizon - len(seq), 1))
                    seq = np.concatenate([seq, pad], axis=0)
            else:
                start = max(0, t - horizon + 1)
                seq = data[start:t + 1]
                if len(seq) < horizon:
                    pad = np.tile(seq[0:1], (horizon - len(seq), 1))
                    seq = np.concatenate([pad, seq], axis=0)
            return seq.astype(np.float32)

        def _generate_synthetic(self):
            self.state_dim = 16
            self.action_dim = 12
            rng = np.random.default_rng(42)
            for _ in range(10):
                T = 100
                states = np.cumsum(
                    rng.standard_normal((T, self.state_dim)) * 0.1, axis=0
                ).astype(np.float32)
                actions = (rng.standard_normal((T, self.action_dim)) * 0.1).astype(
                    np.float32
                )
                for t in range(T):
                    obs = self._pad_window(states, t, self.obs_horizon, "left")
                    act = self._pad_window(actions, t, self.action_horizon, "right")
                    self.windows.append((obs, act))

        # -- torch Dataset interface --------------------------------------

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, idx):
            obs_seq, act_seq = self.windows[idx]
            obs_norm = (obs_seq - self.state_mean) / self.state_std
            act_norm = (act_seq - self.action_mean) / self.action_std
            return torch.from_numpy(obs_norm), torch.from_numpy(act_norm)

    dataset = CabinetDemoDataset(
        dataset_path, obs_horizon, action_horizon,
        max_episodes=config.get("max_episodes", None),
    )

    val_frac = config.get("val_fraction", 0.1)
    val_size = max(1, int(len(dataset) * val_frac))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

    state_dim = dataset.state_dim
    action_dim = dataset.action_dim

    # ----------------------------------------------------------------
    # 2. Build model
    # ----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = ActionChunkingPolicy(
        state_dim, action_dim, obs_horizon, action_horizon,
        hidden_dim=hidden_dim, n_blocks=n_blocks, dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Policy:     ActionChunkingPolicy (residual MLP)")
    print(f"Parameters: {n_params:,}")

    # ----------------------------------------------------------------
    # 3. Training loop
    # ----------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"],
        eta_min=config["learning_rate"] * 0.01,
    )
    grad_clip = config.get("grad_clip", 1.0)

    checkpoint_dir = config.get("checkpoint_dir", "/tmp/cabinet_policy_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print_section("Training")
    print(f"Epochs:       {config['epochs']}")
    print(f"Batch size:   {config['batch_size']}")
    print(f"LR:           {config['learning_rate']}")
    print(f"Train size:   {train_size}")
    print(f"Val size:     {val_size}")

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(checkpoint_dir, "best_policy.pt")

    def _make_ckpt(epoch, train_loss, val_loss):
        return {
            "policy_type": "action_chunking",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "loss": val_loss,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "obs_horizon": obs_horizon,
            "action_horizon": action_horizon,
            "action_exec_horizon": action_exec_horizon,
            "hidden_dim": hidden_dim,
            "n_blocks": n_blocks,
            "dropout": dropout,
            "state_keys": STATE_KEYS,
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
            "action_mean": dataset.action_mean,
            "action_std": dataset.action_std,
        }

    avg_train = float("inf")
    avg_val = float("inf")

    for epoch in range(config["epochs"]):
        # --- train ---
        model.train()
        train_loss_sum, n_train = 0.0, 0
        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            pred = model(obs_batch)
            loss = nn.functional.mse_loss(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss_sum += loss.item()
            n_train += 1

        scheduler.step()
        avg_train = train_loss_sum / max(n_train, 1)

        # --- validate ---
        model.eval()
        val_loss_sum, n_val = 0.0, 0
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                obs_batch = obs_batch.to(device)
                act_batch = act_batch.to(device)
                pred = model(obs_batch)
                val_loss_sum += nn.functional.mse_loss(pred, act_batch).item()
                n_val += 1
        avg_val = val_loss_sum / max(n_val, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch + 1:4d}/{config['epochs']}  "
                f"Train: {avg_train:.6f}  Val: {avg_val:.6f}  LR: {lr:.2e}"
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(_make_ckpt(epoch, avg_train, avg_val), best_ckpt_path)

    final_path = os.path.join(checkpoint_dir, "final_policy.pt")
    torch.save(_make_ckpt(config["epochs"], avg_train, avg_val), final_path)

    print(f"\nTraining complete!")
    print(f"Best val loss:    {best_val_loss:.6f}")
    print(f"Best checkpoint:  {best_ckpt_path}")
    print(f"Final checkpoint: {final_path}")

    print_section("Next Steps")
    print(
        "Evaluate your policy:\n"
        f"  python 07_evaluate_policy.py --checkpoint {best_ckpt_path}\n"
        "\n"
        "Visualize rollouts:\n"
        f"  python 08_visualize_policy_rollout.py --checkpoint {best_ckpt_path}\n"
        "\n"
        "Sweep the action horizon K (try 4, 8, 16) and compare:\n"
        "  python 06_train_policy.py --action_horizon 4\n"
        "  python 06_train_policy.py --action_horizon 16\n"
    )


def print_diffusion_policy_instructions():
    """Print instructions for using the official Diffusion Policy repo."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Train an action-chunking BC policy for OpenCabinet"
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--obs_horizon", type=int, default=5,
        help="Number of past observations as context",
    )
    parser.add_argument(
        "--action_horizon", type=int, default=10,
        help="Number of future actions to predict (K)",
    )
    parser.add_argument(
        "--action_exec_horizon", type=int, default=5,
        help="Actions to execute before re-planning",
    )
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument(
        "--checkpoint_dir", type=str, default="/tmp/cabinet_policy_checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides other args)",
    )
    parser.add_argument(
        "--use_diffusion_policy", action="store_true",
        help="Print instructions for using the official Diffusion Policy repo",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Action Chunking Policy Training")
    print("=" * 60)

    if args.use_diffusion_policy:
        print_diffusion_policy_instructions()
        return

    if args.config:
        config = load_config(args.config)
    else:
        config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "obs_horizon": args.obs_horizon,
            "action_horizon": args.action_horizon,
            "action_exec_horizon": args.action_exec_horizon,
            "hidden_dim": args.hidden_dim,
            "n_blocks": args.n_blocks,
            "checkpoint_dir": args.checkpoint_dir,
        }

    train_policy(config)


if __name__ == "__main__":
    main()
