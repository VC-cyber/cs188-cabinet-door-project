"""
Shared policy architectures and inference utilities for OpenCabinet.

Used by 06_train_policy.py, 07_evaluate_policy.py, and 08_visualize_policy_rollout.py.
"""

import numpy as np
import torch
import torch.nn as nn


# Canonical low-dim observation keys for PandaOmron in OpenCabinet.
# This order MUST match the feature order in the LeRobot parquet files
# (observation.state column): base first, gripper last.
# Total: 3 + 4 + 3 + 4 + 2 = 16 dims.
STATE_KEYS = [
    "robot0_base_pos",          # (3,)
    "robot0_base_quat",         # (4,)
    "robot0_base_to_eef_pos",   # (3,)
    "robot0_base_to_eef_quat",  # (4,)
    "robot0_gripper_qpos",      # (2,)
]

# Action ordering mismatch between LeRobot parquet and env.step().
#
# Env (composite controller) order:
#   [arm_pos(3), arm_rot(3), gripper(1), base(3), torso(1), mode(1)]
#   right[0:6], right_gripper[6], base[7:10], torso[10], mode[11]
#
# Parquet (LeRobot dataset) order:
#   [base(3), torso(1), mode(1), arm_pos(3), arm_rot(3), gripper(1)]
#
# PARQUET_TO_ENV[i] gives the parquet index that maps to env index i.
PARQUET_TO_ENV_ACTION = [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]


def reorder_action_for_env(action):
    """Convert an action from parquet/model order to env.step() order."""
    return action[PARQUET_TO_ENV_ACTION]


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


class ActionChunkingPolicy(nn.Module):
    """
    Residual MLP that maps an observation window to an action chunk.

    Input:  (batch, obs_horizon, state_dim)  -- recent observation history
    Output: (batch, action_horizon, action_dim) -- future action chunk
    """

    def __init__(self, state_dim, action_dim, obs_horizon, action_horizon,
                 hidden_dim=512, n_blocks=4, dropout=0.1):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        input_dim = state_dim * obs_horizon
        output_dim = action_dim * action_horizon

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, drop=dropout) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, obs_seq):
        B = obs_seq.shape[0]
        x = obs_seq.reshape(B, -1)
        x = self.input_proj(x)
        x = self.blocks(x)
        x = self.output_proj(x)
        return x.reshape(B, self.action_horizon, self.action_dim)


class SimplePolicy(nn.Module):
    """Original single-step MLP (kept for loading old checkpoints)."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
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


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_policy_from_checkpoint(checkpoint_path, device):
    """
    Load a trained policy from a checkpoint file.

    Returns (model, ckpt_dict).  Works with both the new action-chunking
    format and the old single-step SimplePolicy format.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    policy_type = ckpt.get("policy_type", "simple")

    if policy_type == "action_chunking":
        model = ActionChunkingPolicy(
            state_dim, action_dim,
            obs_horizon=ckpt["obs_horizon"],
            action_horizon=ckpt["action_horizon"],
            hidden_dim=ckpt.get("hidden_dim", 512),
            n_blocks=ckpt.get("n_blocks", 4),
            dropout=ckpt.get("dropout", 0.1),
        ).to(device)
    else:
        model = SimplePolicy(state_dim, action_dim).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded policy from: {checkpoint_path}")
    print(f"  Type: {policy_type}")
    print(f"  Trained for {ckpt['epoch']} epochs, loss={ckpt['loss']:.6f}")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")
    if policy_type == "action_chunking":
        print(f"  Obs horizon: {ckpt['obs_horizon']}, "
              f"Action horizon: {ckpt['action_horizon']}, "
              f"Exec horizon: {ckpt.get('action_exec_horizon', ckpt['action_horizon'])}")

    return model, ckpt


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def extract_state(obs, state_dim, state_keys=None):
    """
    Extract a fixed-size state vector from environment observations
    using only the canonical STATE_KEYS in a deterministic order.

    This ensures the features at inference match exactly what the
    model was trained on from the demonstration parquet files.
    """
    if state_keys is None:
        state_keys = STATE_KEYS

    parts = []
    for key in state_keys:
        if key in obs and isinstance(obs[key], np.ndarray):
            parts.append(obs[key].flatten())

    if not parts:
        return np.zeros(state_dim, dtype=np.float32)

    state = np.concatenate(parts).astype(np.float32)

    if len(state) < state_dim:
        state = np.pad(state, (0, state_dim - len(state)))
    elif len(state) > state_dim:
        state = state[:state_dim]
    return state


class ActionChunkingInference:
    """
    Manages observation history and action buffer for chunked inference.

    On each call to predict():
      - If the action buffer still has actions, pop and return the next one.
      - Otherwise, run the model on the recent observation window, fill the
        buffer with the predicted chunk, and return the first action.
    """

    def __init__(self, model, ckpt, device):
        self.model = model
        self.device = device
        self.obs_horizon = ckpt["obs_horizon"]
        self.action_exec_horizon = ckpt.get("action_exec_horizon", ckpt["action_horizon"])
        self.state_dim = ckpt["state_dim"]
        self.state_keys = ckpt.get("state_keys", STATE_KEYS)

        self.state_mean = torch.from_numpy(ckpt["state_mean"]).to(device)
        self.state_std = torch.from_numpy(ckpt["state_std"]).to(device)
        self.action_mean = torch.from_numpy(ckpt["action_mean"]).to(device)
        self.action_std = torch.from_numpy(ckpt["action_std"]).to(device)

        self.obs_buffer = []
        self.action_buffer = []

    def reset(self):
        self.obs_buffer.clear()
        self.action_buffer.clear()

    def predict(self, state):
        """Return the next action given a raw (unnormalized) state vector."""
        self.obs_buffer.append(state.copy())
        if len(self.obs_buffer) > self.obs_horizon:
            self.obs_buffer = self.obs_buffer[-self.obs_horizon:]

        # Use buffered action if available
        if self.action_buffer:
            return self.action_buffer.pop(0)

        # Pad observation history to obs_horizon
        obs_list = list(self.obs_buffer)
        while len(obs_list) < self.obs_horizon:
            obs_list.insert(0, obs_list[0])

        obs_tensor = torch.from_numpy(
            np.stack(obs_list)
        ).unsqueeze(0).to(self.device)
        obs_norm = (obs_tensor - self.state_mean) / self.state_std

        with torch.no_grad():
            pred_norm = self.model(obs_norm)

        pred = pred_norm * self.action_std + self.action_mean
        actions = pred.squeeze(0).cpu().numpy()

        self.action_buffer = list(actions[1:self.action_exec_horizon])
        return actions[0]
