"""
Shared utilities for the OpenCabinet policy training and evaluation.

Contains model definitions, normalization, state extraction, and data loading
used by 06_train_policy.py, 07_evaluate_policy.py, and 08_visualize_policy_rollout.py.
"""

import math
import os

import numpy as np

# ── Key mappings ──────────────────────────────────────────────────────────
# LeRobot parquet column names  ↔  robosuite observation keys.
# Both training and evaluation MUST use the same features in the same order.

LEROBOT_STATE_KEYS = [
    "state.end_effector_position_relative",  # 3D
    "state.end_effector_rotation_relative",  # 4D
    "state.gripper_qpos",                    # 2D
]

ROBOSUITE_STATE_KEYS = [
    "robot0_base_to_eef_pos",   # 3D
    "robot0_base_to_eef_quat",  # 4D
    "robot0_gripper_qpos",      # 2D
]

LEROBOT_ACTION_KEYS = [
    "action.end_effector_position",   # 3D
    "action.end_effector_rotation",   # 3D
    "action.gripper_close",           # 1D
    "action.base_motion",             # 4D
    "action.control_mode",            # 1D
]


# ── Data loading ──────────────────────────────────────────────────────────

def _extract_column(df, col_name):
    """Extract values from a parquet column, handling arrays, scalars, and sub-columns."""
    if col_name in df.columns:
        vals = df[col_name].values
        if len(vals) > 0:
            if isinstance(vals[0], (np.ndarray, list)):
                return np.stack([np.asarray(v, dtype=np.float32) for v in vals])
            return vals.astype(np.float32).reshape(-1, 1)

    sub_cols = sorted(
        [c for c in df.columns if c.startswith(col_name + ".")],
        key=lambda c: int(c.rsplit(".", 1)[-1]) if c.rsplit(".", 1)[-1].isdigit() else 0,
    )
    if sub_cols:
        return np.column_stack([df[c].values.astype(np.float32) for c in sub_cols])

    return None


def load_dataset_arrays(dataset_path, max_episodes=None):
    """Load state-action data from LeRobot parquet files.

    Returns (states, actions, episode_ids) as numpy arrays.
    """
    import pyarrow.parquet as pq

    data_dir = os.path.join(dataset_path, "data")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(dataset_path, "lerobot", "data")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found under: {dataset_path}")

    chunk_dir = os.path.join(data_dir, "chunk-000")
    if not os.path.exists(chunk_dir):
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    parquet_files = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {chunk_dir}")

    all_states, all_actions, all_ep_ids = [], [], []

    for pf in parquet_files:
        df = pq.read_table(os.path.join(chunk_dir, pf)).to_pandas()

        ep_col = None
        for name in ["episode_index", "episode_id", "episode"]:
            if name in df.columns:
                ep_col = df[name].values.astype(int)
                break
        if ep_col is None:
            ep_col = np.zeros(len(df), dtype=int)

        state_parts = []
        for key in LEROBOT_STATE_KEYS:
            arr = _extract_column(df, key)
            if arr is not None:
                state_parts.append(arr)

        if not state_parts:
            for c in sorted(c for c in df.columns if c.startswith("state.")):
                arr = _extract_column(df, c)
                if arr is not None:
                    state_parts.append(arr)

        action_parts = []
        for key in LEROBOT_ACTION_KEYS:
            arr = _extract_column(df, key)
            if arr is not None:
                action_parts.append(arr)

        if not action_parts:
            for c in sorted(c for c in df.columns if c.startswith("action.")):
                arr = _extract_column(df, c)
                if arr is not None:
                    action_parts.append(arr)

        if state_parts and action_parts:
            all_states.append(np.hstack(state_parts).astype(np.float32))
            all_actions.append(np.hstack(action_parts).astype(np.float32))
            all_ep_ids.append(ep_col)

        if max_episodes is not None:
            unique_eps = set()
            for ea in all_ep_ids:
                unique_eps.update(ea.tolist())
            if len(unique_eps) >= max_episodes:
                break

    if not all_states:
        raise RuntimeError(
            "Could not extract state-action pairs from the dataset.\n"
            "Run 04_download_dataset.py to download the data first."
        )

    return np.concatenate(all_states), np.concatenate(all_actions), np.concatenate(all_ep_ids)


# ── Normalization ─────────────────────────────────────────────────────────

def compute_norm_params(data):
    """Compute min-max normalization params for mapping data → [-1, 1]."""
    flat = data.reshape(-1, data.shape[-1]) if data.ndim > 2 else data
    return {
        "min": flat.min(axis=0).astype(np.float64),
        "max": flat.max(axis=0).astype(np.float64),
    }


def normalize(data, params):
    """Normalize data to [-1, 1] using min-max."""
    scale = params["max"] - params["min"]
    scale = np.where(scale < 1e-8, 1.0, scale)
    return ((data - params["min"]) / scale * 2.0 - 1.0).astype(np.float32)


def denormalize(data, params):
    """Denormalize from [-1, 1] back to original range."""
    scale = params["max"] - params["min"]
    scale = np.where(scale < 1e-8, 1.0, scale)
    if hasattr(data, "numpy"):
        result = (data.cpu().numpy() + 1.0) / 2.0 * scale + params["min"]
    else:
        result = (data + 1.0) / 2.0 * scale + params["min"]
    return result.astype(np.float32)


def _norm_to_serializable(params):
    """Convert norm params to JSON-safe lists for checkpoint saving."""
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in params.items()}


def _norm_from_checkpoint(params):
    """Restore norm params from checkpoint (lists → numpy)."""
    return {k: np.array(v, dtype=np.float64) if isinstance(v, list) else v for k, v in params.items()}


# ── State extraction (evaluation) ────────────────────────────────────────

def extract_state_from_obs(obs):
    """Extract a state vector from robosuite observations using the fixed key order."""
    parts = []
    for key in ROBOSUITE_STATE_KEYS:
        if key in obs:
            parts.append(obs[key].flatten().astype(np.float32))
    if not parts:
        return np.zeros(9, dtype=np.float32)
    return np.concatenate(parts)


# ── Model definitions ─────────────────────────────────────────────────────

def build_simple_policy(state_dim, action_dim, hidden_dim=256):
    """Construct a SimplePolicy MLP (requires torch)."""
    import torch.nn as nn

    class SimplePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),
            )

        def forward(self, state):
            return self.net(state)

    return SimplePolicy()


def build_diffusion_policy(state_dim, action_dim, chunk_size=8,
                           hidden_dim=256, n_diffusion_steps=50):
    """Construct a minimal DiffusionPolicy with action chunking (requires torch)."""
    import torch
    import torch.nn as nn

    class SinusoidalPosEmb(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, t):
            half = self.dim // 2
            emb = math.log(10000) / (half - 1)
            emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
            emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
            return torch.cat([emb.sin(), emb.cos()], dim=-1)

    class DiffusionPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.chunk_size = chunk_size
            self.n_diffusion_steps = n_diffusion_steps
            self.output_dim = action_dim * chunk_size

            self.time_emb = nn.Sequential(
                SinusoidalPosEmb(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.state_enc = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            in_dim = self.output_dim + hidden_dim * 2
            self.noise_pred = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.Mish(),
                nn.Linear(512, 512),
                nn.Mish(),
                nn.Linear(512, 512),
                nn.Mish(),
                nn.Linear(512, self.output_dim),
            )

            betas = torch.linspace(1e-4, 0.02, n_diffusion_steps)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)

            self.register_buffer("betas", betas)
            self.register_buffer("alphas", alphas)
            self.register_buffer("alphas_cumprod", alphas_cumprod)
            self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
            self.register_buffer("sqrt_one_minus_ac", (1.0 - alphas_cumprod).sqrt())

        def _predict_noise(self, noisy_action, state, timestep):
            t_emb = self.time_emb(timestep)
            s_emb = self.state_enc(state)
            x = torch.cat([noisy_action, s_emb, t_emb], dim=-1)
            return self.noise_pred(x)

        def compute_loss(self, state, action_chunk):
            """Forward pass for training: add noise to actions, predict it."""
            B = state.shape[0]
            action_flat = action_chunk.reshape(B, -1)

            t = torch.randint(0, self.n_diffusion_steps, (B,), device=state.device)
            noise = torch.randn_like(action_flat)

            noisy = (
                self.sqrt_alphas_cumprod[t].unsqueeze(-1) * action_flat
                + self.sqrt_one_minus_ac[t].unsqueeze(-1) * noise
            )

            pred = self._predict_noise(noisy, state, t)
            return nn.functional.mse_loss(pred, noise)

        @torch.no_grad()
        def sample(self, state):
            """DDPM reverse process: iteratively denoise from random noise."""
            B = state.shape[0]
            x = torch.randn(B, self.output_dim, device=state.device)

            for t in reversed(range(self.n_diffusion_steps)):
                t_batch = torch.full((B,), t, device=state.device, dtype=torch.long)
                eps = self._predict_noise(x, state, t_batch)

                alpha = self.alphas[t]
                alpha_bar = self.alphas_cumprod[t]
                beta = self.betas[t]

                mean = (1.0 / alpha.sqrt()) * (
                    x - (beta / (1.0 - alpha_bar).sqrt()) * eps
                )

                if t > 0:
                    x = mean + beta.sqrt() * torch.randn_like(x)
                else:
                    x = mean

            return x.reshape(B, self.chunk_size, self.action_dim)

    return DiffusionPolicy()


# ── Policy loading ────────────────────────────────────────────────────────

def load_policy_checkpoint(checkpoint_path, device=None):
    """Load a trained policy from a checkpoint file.

    Returns (model, ckpt_dict, state_norm, action_norm).
    For MLP policies state_norm and action_norm are None.
    """
    import torch

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    policy_type = ckpt.get("policy_type", "mlp")

    if policy_type == "diffusion":
        model = build_diffusion_policy(
            state_dim=ckpt["state_dim"],
            action_dim=ckpt["action_dim"],
            chunk_size=ckpt["chunk_size"],
            hidden_dim=ckpt["hidden_dim"],
            n_diffusion_steps=ckpt["n_diffusion_steps"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        state_norm = _norm_from_checkpoint(ckpt["state_norm"])
        action_norm = _norm_from_checkpoint(ckpt["action_norm"])
        return model, ckpt, state_norm, action_norm

    else:
        model = build_simple_policy(ckpt["state_dim"], ckpt["action_dim"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt, None, None
