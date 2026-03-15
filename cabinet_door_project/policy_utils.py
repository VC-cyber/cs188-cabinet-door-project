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
    # Use 16D proprio to match observation.state in dataset.
    # Keep stable order for both train/eval.
    "robot0_base_pos",          # 3D
    "robot0_base_quat",         # 4D
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

# Optional handle features from 05b augmentation (order matches training state)
LEROBOT_HANDLE_STATE_KEYS = [
    "observation.handle_pos",           # 3D
    "observation.handle_to_eef_pos",   # 3D
]
# Optional: "observation.door_openness" (1D) can be appended for 16-dim state

# Observation keys for handle at eval (set by wrapper when using augmented state)
ROBOSUITE_HANDLE_STATE_KEYS = [
    "robot0_handle_pos",
    "robot0_handle_to_eef_pos",
]


def _get_handle_feature_mode():
    """Handle feature mode: both | relative_only | none."""
    mode = os.environ.get("CABINET_HANDLE_FEATURE_MODE", "both").strip().lower()
    if mode not in {"both", "relative_only", "none"}:
        mode = "both"
    return mode


def _selected_lerobot_handle_keys():
    mode = _get_handle_feature_mode()
    if mode == "both":
        return LEROBOT_HANDLE_STATE_KEYS
    if mode == "relative_only":
        return ["observation.handle_to_eef_pos"]
    return []


def _selected_robosuite_handle_keys():
    mode = _get_handle_feature_mode()
    if mode == "both":
        return ROBOSUITE_HANDLE_STATE_KEYS
    if mode == "relative_only":
        return ["robot0_handle_to_eef_pos"]
    return []


def get_dataset_path():
    """Get the path to the OpenCabinet dataset (shared by 05b, 06, etc.)."""
    try:
        import robocasa  # noqa: F401
        from robocasa.utils.dataset_registry_utils import get_ds_path
    except ImportError:
        return None
    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        return None
    return path


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

    Prefers augmented/ dir (from 05b_augment_handle_data.py) when present, so state
    includes handle_pos and handle_to_eef_pos (15 dims). Otherwise uses data/chunk-000 (9 dims).

    Returns (states, actions, episode_ids) as numpy arrays.
    """
    import pyarrow.parquet as pq

    dataset_path = os.path.abspath(dataset_path)
    aug_dir = os.path.join(dataset_path, "augmented")
    data_dir = os.path.join(dataset_path, "data")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(dataset_path, "lerobot", "data")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found under: {dataset_path}")

    # Prefer augmented parquet files when available
    if os.path.isdir(aug_dir):
        parquet_files = sorted(f for f in os.listdir(aug_dir) if f.endswith(".parquet"))
        chunk_dir = aug_dir
    else:
        chunk_dir = os.path.join(data_dir, "chunk-000")
        if not os.path.exists(chunk_dir):
            raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")
        parquet_files = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {chunk_dir}")

    all_states, all_actions, all_ep_ids = [], [], []
    use_handle = chunk_dir == aug_dir

    for pf in parquet_files:
        df = pq.read_table(os.path.join(chunk_dir, pf)).to_pandas()

        ep_col = None
        for name in ["episode_index", "episode_id", "episode"]:
            if name in df.columns:
                ep_col = df[name].values.astype(int)
                break
        if ep_col is None:
            ep_col = np.zeros(len(df), dtype=int)

        # State: proprio (9) then optional handle (6) = [proprio, handle_pos, handle_to_eef_pos]
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
        if not state_parts and "observation.state" in df.columns:
            arr = _extract_column(df, "observation.state")
            if arr is not None:
                state_parts.append(arr)

        if use_handle:
            for key in _selected_lerobot_handle_keys():
                arr = _extract_column(df, key)
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
        if not action_parts and "action" in df.columns:
            arr = _extract_column(df, "action")
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


# ── Action reordering and gripper binarization ───────────────────────────
# Dataset order (verified in WORKING_SETUP + parquet stats):
# [base_motion(3), torso(1), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)] = 12
# Env order:
# [eef_pos(3), eef_rot(3), gripper(1), base_motion(3), torso(1), base_mode(1)] = 12
# Gripper and base_mode: binarize at 0.0 (>=0 -> 1, <0 -> -1), not 0.5 (WORKING_SETUP).

def dataset_action_to_env_action(policy_action, env_action_dim):
    """Convert policy output (dataset order) to env action order.

    Dataset action vector is assumed to be:
      [base_x, base_y, base_yaw, torso, control_mode, eef_x, eef_y, eef_z, rot_x, rot_y, rot_z, gripper]
    """
    policy_action = np.atleast_1d(np.asarray(policy_action, dtype=np.float64))
    # Policy has 12 dims: base(0:3), torso(3), control_mode(4), eef_pos(5:8), eef_rot(8:11), gripper(11)
    if len(policy_action) >= 12:
        base_motion_3 = policy_action[0:3]
        torso = policy_action[3]
        control_mode_raw = policy_action[4]
        eef_pos = policy_action[5:8]
        eef_rot = policy_action[8:11]
        gripper_raw = policy_action[11]
        gripper_bin = 1.0 if gripper_raw >= 0.0 else -1.0
        base_mode_bin = 1.0 if control_mode_raw >= 0.0 else -1.0
        env_action = np.concatenate([
            eef_pos,
            eef_rot,
            [gripper_bin],
            base_motion_3,
            [torso],
            [base_mode_bin],
        ]).astype(np.float32)
    else:
        env_action = np.zeros(env_action_dim, dtype=np.float32)
        copy_len = min(len(policy_action), env_action_dim)
        env_action[:copy_len] = policy_action[:copy_len]
        if copy_len > 6:
            env_action[6] = 1.0 if env_action[6] >= 0.0 else -1.0
        if copy_len > 11:
            env_action[11] = 1.0 if env_action[11] >= 0.0 else -1.0

    if len(env_action) < env_action_dim:
        env_action = np.pad(env_action, (0, env_action_dim - len(env_action)))
    elif len(env_action) > env_action_dim:
        env_action = env_action[:env_action_dim]
    return env_action


# ── State extraction (evaluation) ────────────────────────────────────────

def get_state_dim_from_obs(obs):
    """Return state dimension that extract_state_from_obs would produce (9 or 15 with handle)."""
    dim = 0
    for key in ROBOSUITE_STATE_KEYS:
        if key in obs:
            dim += np.size(obs[key])
    for key in _selected_robosuite_handle_keys():
        if key in obs:
            dim += np.size(obs[key])
    return dim if dim > 0 else 9


def extract_state_from_obs(obs, state_dim=None):
    """Extract a state vector from robosuite observations using the fixed key order.

    If obs contains handle keys from wrapper, appends those according to
    CABINET_HANDLE_FEATURE_MODE so eval matches training feature selection.
    """
    parts = []
    for key in ROBOSUITE_STATE_KEYS:
        if key in obs:
            parts.append(obs[key].flatten().astype(np.float32))
    for key in _selected_robosuite_handle_keys():
        if key in obs:
            parts.append(obs[key].flatten().astype(np.float32))
    if not parts:
        return np.zeros(9, dtype=np.float32)
    out = np.concatenate(parts)
    if state_dim is not None and len(out) != state_dim:
        if len(out) < state_dim:
            out = np.pad(out, (0, state_dim - len(out)))
        else:
            out = out[:state_dim]
    return out


# ── Relaxed success (one door open) ───────────────────────────────────────

DOOR_OPEN_THRESHOLD_RAD = 0.3  # ~17 degrees; any hinge beyond this counts as success


def check_success_relaxed(env):
    """Return True if any cabinet door joint is open beyond DOOR_OPEN_THRESHOLD_RAD.

    Use instead of env._check_success() so one open door counts as success.
    Falls back to env._check_success() if we cannot read door joints.
    """
    try:
        sim = env.sim
        if sim is None:
            return env._check_success()
        fxtr = getattr(env, "fxtr", None)
        if fxtr is None:
            return env._check_success()
        fixture_name = getattr(fxtr, "name", None) or ""
        model = sim.model
        data = sim.data
        for i in range(model.njnt):
            jname = model.joint(i).name
            if fixture_name in jname and "door" in jname:
                addr = model.joint(i).qposadr[0]
                qpos = data.qpos[addr]
                jrange = model.jnt_range[i]
                jmin, jmax = jrange[0], jrange[1]
                if jmax - jmin > 1e-8:
                    dist_closed = min(abs(qpos - jmin), abs(qpos - jmax))
                    if abs(jmin) < abs(jmax):
                        dist_closed = abs(qpos - jmin)
                    else:
                        dist_closed = abs(qpos - jmax)
                    if dist_closed > DOOR_OPEN_THRESHOLD_RAD:
                        return True
        return env._check_success()
    except Exception:
        return env._check_success()


# ── Observation wrapper for handle state at eval ──────────────────────────

def _get_handle_state_from_env(env):
    """Get handle_pos (3) and handle_to_eef_pos (3) from env sim; return None on failure.

    Matches 05b augmentation logic:
    - Build handle<->door-joint associations
    - Prefer handles whose doors are not yet open (openness < 0.9)
    - Among candidates, choose nearest to current EEF
    """
    try:
        sim = env.sim
        fxtr = getattr(env, "fxtr", None)
        if sim is None or fxtr is None:
            return None
        fixture_name = getattr(fxtr, "name", None) or ""
        model = sim.model
        data = sim.data
        # Find handle bodies
        handle_bodies = []
        for i in range(model.nbody):
            name = model.body(i).name
            if fixture_name in name and "handle" in name:
                handle_bodies.append(name)
        if not handle_bodies:
            return None

        # Find door joints for this fixture
        door_joints = []
        for i in range(model.njnt):
            jname = model.joint(i).name
            if fixture_name in jname and "door" in jname:
                door_joints.append((jname, i))

        # Build handle->joint map (left/right-aware)
        if len(handle_bodies) == 1 or len(door_joints) == 1:
            handle_to_joint_map = {hb: door_joints for hb in handle_bodies}
        else:
            handle_to_joint_map = {}
            for hb in handle_bodies:
                hbl = hb.lower()
                if "left" in hbl:
                    matched = [(jn, ji) for jn, ji in door_joints if "left" in jn.lower()]
                elif "right" in hbl:
                    matched = [(jn, ji) for jn, ji in door_joints if "right" in jn.lower()]
                else:
                    matched = []
                handle_to_joint_map[hb] = matched if matched else door_joints

        def _door_openness_for_handle(hb):
            joints = handle_to_joint_map.get(hb, [])
            if not joints:
                return 0.0
            vals = []
            for _, jidx in joints:
                addr = model.joint(jidx).qposadr[0]
                qpos = data.qpos[addr]
                jmin, jmax = model.jnt_range[jidx]
                if jmax - jmin > 1e-8:
                    # Closed position is bound closest to 0 (same as 05b)
                    if abs(jmin) < abs(jmax):
                        norm = abs(qpos - jmin) / (jmax - jmin)
                    else:
                        norm = abs(qpos - jmax) / (jmax - jmin)
                else:
                    norm = 0.0
                vals.append(float(np.clip(norm, 0.0, 1.0)))
            return float(np.mean(vals))

        eef_pos = data.body("gripper0_right_eef").xpos.copy()
        # Prefer unopened doors first (same OPEN_THRESHOLD as 05b)
        OPEN_THRESHOLD = 0.90
        active = [hb for hb in handle_bodies if _door_openness_for_handle(hb) < OPEN_THRESHOLD]
        candidates = active if active else handle_bodies

        # Use nearest handle among candidates
        handle_pos = None
        best_dist = float("inf")
        for hb in candidates:
            pos = data.body(hb).xpos.copy()
            d = np.linalg.norm(pos - eef_pos)
            if d < best_dist:
                best_dist = d
                handle_pos = pos
        if handle_pos is None:
            return None
        # WORKING_SETUP convention: handle_to_eef = eef_pos - handle_pos
        handle_to_eef = eef_pos - handle_pos
        return {"robot0_handle_pos": handle_pos.astype(np.float32), "robot0_handle_to_eef_pos": handle_to_eef.astype(np.float32)}
    except Exception:
        return None


class HandleObservationWrapper:
    """Wraps env so obs includes robot0_handle_pos and robot0_handle_to_eef_pos for 15-dim state."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        self._inject_handle(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._inject_handle(obs)
        return obs, reward, done, info

    def _inject_handle(self, obs):
        h = _get_handle_state_from_env(self._env)
        if h:
            obs["robot0_handle_pos"] = h["robot0_handle_pos"]
            obs["robot0_handle_to_eef_pos"] = h["robot0_handle_to_eef_pos"]


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


def build_bc_unet_policy(state_dim, action_dim, chunk_size=16, hidden_dim=256, n_channels=32):
    """BC policy with 1D convolutional U-Net backbone; predicts action chunk from state.

    Input: state (B, state_dim). Output: (B, chunk_size, action_dim).
    Small (~few M params) to avoid overfitting on ~100 demos.
    """
    import torch
    import torch.nn as nn

    class Conv1dBlock(nn.Module):
        def __init__(self, in_c, out_c, kernel_size=3):
            super().__init__()
            self.conv = nn.Conv1d(in_c, out_c, kernel_size, padding=kernel_size // 2)
            self.norm = nn.GroupNorm(min(8, out_c), out_c)
            self.act = nn.Mish()

        def forward(self, x):
            return self.act(self.norm(self.conv(x)))

    class BCUnetPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.chunk_size = chunk_size
            self.output_dim = action_dim * chunk_size

            self.state_enc = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # Condition as extra channels: (B, n_channels + action_dim, chunk_size)
            self.proj = nn.Linear(hidden_dim, n_channels * chunk_size)

            # 1D U-Net: input (B, n_channels + action_dim, chunk_size), output (B, action_dim, chunk_size)
            c = n_channels
            self.enc1 = nn.Sequential(
                Conv1dBlock(n_channels + action_dim, c),
                Conv1dBlock(c, c),
            )
            self.enc2 = nn.Sequential(
                nn.Conv1d(c, c * 2, 3, stride=2, padding=1),
                nn.GroupNorm(8, c * 2),
                nn.Mish(),
                Conv1dBlock(c * 2, c * 2),
            )
            self.enc3 = nn.Sequential(
                nn.Conv1d(c * 2, c * 4, 3, stride=2, padding=1),
                nn.GroupNorm(8, c * 4),
                nn.Mish(),
                Conv1dBlock(c * 4, c * 4),
            )
            self.bottleneck = nn.Sequential(
                Conv1dBlock(c * 4, c * 4),
                Conv1dBlock(c * 4, c * 4),
            )
            self.dec3 = nn.Sequential(
                Conv1dBlock(c * 4 + c * 2, c * 2),
                Conv1dBlock(c * 2, c * 2),
            )
            self.dec2 = nn.Sequential(
                Conv1dBlock(c * 2 + c, c),
                Conv1dBlock(c, c),
            )
            self.dec1 = nn.Sequential(
                Conv1dBlock(c + n_channels, c),
                nn.Conv1d(c, action_dim, 1),
            )

        def forward(self, state):
            B = state.shape[0]
            s_emb = self.state_enc(state)
            cond = self.proj(s_emb).reshape(B, n_channels, self.chunk_size)
            zeros = torch.zeros(B, self.action_dim, self.chunk_size, device=state.device, dtype=state.dtype)
            x = torch.cat([cond, zeros], dim=1)
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            b = self.bottleneck(e3)
            d3 = nn.functional.interpolate(b, size=e2.shape[2], mode="linear", align_corners=False)
            d3 = self.dec3(torch.cat([d3, e2], dim=1))
            d2 = nn.functional.interpolate(d3, size=e1.shape[2], mode="linear", align_corners=False)
            d2 = self.dec2(torch.cat([d2, e1], dim=1))
            out = self.dec1(torch.cat([d2, x[:, :n_channels]], dim=1))
            return out.permute(0, 2, 1)

        @torch.no_grad()
        def sample(self, state):
            return self.forward(state)

    return BCUnetPolicy()


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

    elif policy_type == "bc_unet":
        model = build_bc_unet_policy(
            state_dim=ckpt["state_dim"],
            action_dim=ckpt["action_dim"],
            chunk_size=ckpt["chunk_size"],
            hidden_dim=ckpt.get("hidden_dim", 256),
            n_channels=ckpt.get("n_channels", 32),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        state_norm = _norm_from_checkpoint(ckpt["state_norm"]) if ckpt.get("state_norm") else None
        action_norm = _norm_from_checkpoint(ckpt["action_norm"]) if ckpt.get("action_norm") else None
        return model, ckpt, state_norm, action_norm

    else:
        model = build_simple_policy(ckpt["state_dim"], ckpt["action_dim"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt, None, None
