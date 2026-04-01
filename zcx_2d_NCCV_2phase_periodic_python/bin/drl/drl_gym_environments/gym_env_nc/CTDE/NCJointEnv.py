import os
import glob
import re
from collections import deque
from typing import Callable, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _latest_restart_step(restart_dir: str) -> int:
    steps = []
    for f in glob.glob(os.path.join(restart_dir, "*_rst_*.xml")):
        m = re.search(r"_rst_(\d+)\.xml$", f.replace("\\", "/"))
        if m:
            steps.append(int(m.group(1)))
    mx = max(steps) if steps else -1
    return mx if mx > 0 else -1


def actions_to_segment_temps(
        raw_actions: np.ndarray,
        baseline_T: float = 2.0,
        ampl: float = 0.75,
        Tmin: float = 0.0,
        Tmax: float = 4.0,
) -> np.ndarray:
    """Map joint actions in [-1, 1] to physically safe segment temperatures."""
    a = np.asarray(raw_actions, dtype=np.float32).reshape(-1)
    a = np.clip(a, -1.0, 1.0)
    centered = a - float(np.mean(a))
    K2 = max(1.0, float(np.max(np.abs(centered))))
    temps = baseline_T + ampl * centered / K2
    return np.clip(temps, Tmin, Tmax).astype(np.float32)


class NCJointEnv(gym.Env):
    """
    CTDE joint env aligned with marl reward/training semantics:
    - action: joint action (n_seg,)
    - obs: per-agent local patch observation, shape (n_seg, n_rows*(n_cols/n_seg)*3)
    - reward: shared mean reward computed from marl's flux/KE shaping
    """

    metadata = {"render_modes": []}

    def __init__(
            self,
            solver_factory: Callable[[int, int, int], object],
            group_id: int,
            training_root: str,
            n_seg: int = 10,
            n_rows: int = 8,
            n_cols: int = 30,
            avg_len: int = 4,
            warmup_time: float = 400.0,
            delta_time: float = 2.0,
            max_steps_per_episode: int = 200,
            beta: float = 0.0015,
            baseline_T: float = 2.0,
            flux_base: float = 0.19,
            flux_scale: float = 0.018,
            ke_base: float = 0.04,
            ke_scale: float = 0.004,
            w_flux: float = 1.0,
            w_ke: float = 0.0,
    ):
        super().__init__()
        self.solver_factory = solver_factory
        self.group_id = int(group_id)

        self.training_root = os.path.abspath(training_root)
        self.case_root = _mkdir(os.path.join(self.training_root, f"CFD_n{self.group_id}"))
        for name in ("input", "output", "reload", "restart"):
            _mkdir(os.path.join(self.case_root, name))
        self.restart_dir = os.path.join(self.case_root, "restart")

        self.n_seg = int(n_seg)
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.avg_len = int(avg_len)
        self._probe_dim = 3
        if self.n_cols % self.n_seg != 0:
            raise ValueError(f"Need n_cols % n_seg == 0. n_cols={self.n_cols}, n_seg={self.n_seg}")

        self.cols_per_seg = self.n_cols // self.n_seg
        self.local_dim = self.n_rows * self.cols_per_seg * self._probe_dim

        self.warmup_time = float(warmup_time)
        self.delta_time = float(delta_time)
        self.max_steps = int(max_steps_per_episode)

        self.beta = float(beta)
        self.baseline_T = float(baseline_T)
        self.flux_base = float(flux_base)
        self.flux_scale = float(flux_scale)
        self.ke_base = float(ke_base)
        self.ke_scale = float(ke_scale)
        self.w_flux = float(w_flux)
        self.w_ke = float(w_ke)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_seg,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(self.n_seg, self.local_dim), dtype=np.float32
        )

        self.episode = 1
        self.step_count = 0
        self.sim_time = float(self.warmup_time)
        self.restart_step = -1
        self.nc = None

        self._probe_hist = deque(maxlen=self.avg_len)
        self._fluxes_hist = deque(maxlen=self.avg_len)
        self._ke_hist = deque(maxlen=self.avg_len)

        self.log_root = os.path.join(self.training_root, f"logs_env_{self.group_id}")
        self._step_curve_inited = False
        self._episode_return = 0.0

    def _probe_index_col_major(self, row: int, col: int) -> int:
        return col * self.n_rows + row

    def _snapshot_grid(self, sim) -> np.ndarray:
        grid = np.empty((self.n_rows, self.n_cols, self._probe_dim), dtype=np.float32)
        for col in range(self.n_cols):
            for row in range(self.n_rows):
                idx = self._probe_index_col_major(row, col)
                grid[row, col, 0] = float(sim.get_local_velocity(idx, 0))
                grid[row, col, 1] = float(sim.get_local_velocity(idx, 1))
                grid[row, col, 2] = float(sim.get_local_temperature(idx))
        return grid

    def _update_probe_average(self, sim) -> np.ndarray:
        snap = self._snapshot_grid(sim).astype(np.float32)
        self._probe_hist.append(snap)
        return np.mean(np.stack(list(self._probe_hist), axis=0), axis=0).astype(np.float32)

    def _joint_obs_from_grid(self, grid: np.ndarray) -> np.ndarray:
        if grid.shape != (self.n_rows, self.n_cols, self._probe_dim):
            raise ValueError(f"grid must be {(self.n_rows, self.n_cols, self._probe_dim)}, got {grid.shape}")

        obs = np.empty((self.n_seg, self.local_dim), dtype=np.float32)
        for seg_index in range(self.n_seg):
            x0 = seg_index * self.cols_per_seg
            x1 = (seg_index + 1) * self.cols_per_seg
            patch = grid[:, x0:x1, :]
            obs[seg_index] = patch.reshape(-1)
        return obs

    def _mixes_flux_vec(self, gen_flux: float, local_flux: np.ndarray) -> np.ndarray:
        return ((1.0 - self.beta) * float(gen_flux) + self.beta * local_flux.astype(np.float32) * float(self.n_seg))

    def _rewards_from_history(self) -> np.ndarray:
        fluxes_bar = np.mean(np.stack(list(self._fluxes_hist), axis=0), axis=0)
        ke_bar = float(np.mean(self._ke_hist))

        x_fluxes = (self.flux_base - fluxes_bar) / (2.0 * self.flux_scale)
        x_fluxes = np.clip(x_fluxes, -10.0, 10.0).astype(np.float32)
        r_fluxes = np.tanh(x_fluxes).astype(np.float32)

        x_ke = (self.ke_base - ke_bar) / (2.0 * self.ke_scale)
        x_ke = float(np.clip(x_ke, -10.0, 10.0))
        r_ke = float(np.tanh(x_ke))

        rewards = self.w_flux * r_fluxes + self.w_ke * r_ke
        return rewards.astype(np.float32)

    def _ensure_baseline_once(self) -> None:
        os.chdir(self.case_root)
        if _latest_restart_step(self.restart_dir) > 0:
            return
        sim = self.solver_factory(self.group_id, 1, 0)
        sim.set_segment_temperatures([self.baseline_T] * self.n_seg)
        sim.run_case(float(self.warmup_time))

    def _step_curve_dir(self) -> str:
        return os.path.join(self.log_root, "mean_reward_curve")

    def _step_curve_file(self, episode: int) -> str:
        return os.path.join(self._step_curve_dir(), f"episode_{int(episode):06d}.txt")

    def _episode_curve_dir(self) -> str:
        return os.path.join(self.log_root, "mean_reward_by_episode")

    def _episode_curve_file(self) -> str:
        return os.path.join(self._episode_curve_dir(), f"reward_env_{int(self.group_id)}.txt")

    def init_step_curve_file(self, episode: int) -> str:
        os.makedirs(self._step_curve_dir(), exist_ok=True)
        f = self._step_curve_file(int(episode))
        with open(f, "w", encoding="utf-8") as fp:
            header = ["actuation", "sim_time", "mean_reward"]
            header += [f"reward_seg{i}" for i in range(self.n_seg)]
            fp.write(",".join(header) + "\n")
        self._step_curve_inited = True
        return f

    def append_step_mean_reward(
            self,
            episode: int,
            actuation: int,
            sim_time: float,
            mean_reward: float,
            rewards_vec: np.ndarray,
    ) -> None:
        os.makedirs(self._step_curve_dir(), exist_ok=True)
        f = self._step_curve_file(int(episode))
        rv = np.asarray(rewards_vec, dtype=np.float32).reshape(-1)
        if rv.size != self.n_seg:
            raise ValueError(f"rewards_vec size mismatch: got {rv.size}, expected {self.n_seg}")

        row = [str(int(actuation)),
               f"{float(sim_time):.6f}",
               f"{float(mean_reward):.6f}"] + [f"{float(x):.6f}" for x in rv.tolist()]
        with open(f, "a", encoding="utf-8", buffering=1) as fp:
            fp.write(",".join(row) + "\n")

    def append_episode_mean_return(self, episode: int, mean_return: float) -> None:
        os.makedirs(self._episode_curve_dir(), exist_ok=True)
        f = self._episode_curve_file()
        with open(f, "a", encoding="utf-8", buffering=1) as fp:
            fp.write(f"episode: {int(episode)}  mean_return: {float(mean_return):.6f}\n")

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        os.chdir(self.case_root)

        print(f"[env {self.group_id}] ===== Episode {self.episode} start =====")

        self.step_count = 0
        self.sim_time = float(self.warmup_time)
        self._episode_return = 0.0
        self._step_curve_inited = False

        self._ensure_baseline_once()
        self.restart_step = _latest_restart_step(self.restart_dir)
        if self.restart_step < 0:
            raise RuntimeError(f"No restart files found in {self.restart_dir}")

        self.nc = self.solver_factory(self.group_id, self.episode, int(self.restart_step))
        self.nc.run_case(float(self.warmup_time))

        self._probe_hist.clear()
        snap0 = self._snapshot_grid(self.nc)
        for _ in range(self.avg_len):
            self._probe_hist.append(snap0.copy())
        grid0 = np.mean(np.stack(list(self._probe_hist), axis=0), axis=0).astype(np.float32)

        gen_flux0 = float(self.nc.get_global_heat_flux())
        local_flux0 = np.asarray([float(self.nc.get_local_phi_flux(i)) for i in range(self.n_seg)], dtype=np.float32)
        gen_ke0 = float(self.nc.get_global_kinetic_energy())
        mixes0 = self._mixes_flux_vec(gen_flux0, local_flux0)

        self._fluxes_hist.clear()
        self._ke_hist.clear()
        for _ in range(self.avg_len):
            self._fluxes_hist.append(mixes0.copy())
            self._ke_hist.append(float(gen_ke0))

        obs0 = self._joint_obs_from_grid(grid0)
        self.init_step_curve_file(self.episode)
        info = {
            "episode": int(self.episode),
            "group_id": int(self.group_id),
            "actuation": 0,
            "sim_time": float(self.sim_time),
        }
        return obs0, info

    def step(self, action: np.ndarray):
        os.chdir(self.case_root)

        episode_now = int(self.episode)
        actuation = int(self.step_count + 1)

        raw_actions = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw_actions.size != self.n_seg:
            raise ValueError(f"Action must have length {self.n_seg}, got {raw_actions.shape}")

        seg_temps = actions_to_segment_temps(raw_actions, baseline_T=self.baseline_T)
        self.nc.set_segment_temperatures(seg_temps.tolist())

        end_time = float(self.sim_time + self.delta_time)
        self.nc.run_case(end_time)
        self.sim_time = end_time

        grid_time_avg = self._update_probe_average(self.nc)
        obs = self._joint_obs_from_grid(grid_time_avg)

        gen_flux = float(self.nc.get_global_heat_flux())
        local_flux = np.asarray([float(self.nc.get_local_phi_flux(i)) for i in range(self.n_seg)], dtype=np.float32)
        gen_ke = float(self.nc.get_global_kinetic_energy())

        self._fluxes_hist.append(self._mixes_flux_vec(gen_flux, local_flux))
        self._ke_hist.append(gen_ke)
        rewards_all = self._rewards_from_history()
        reward = float(np.mean(rewards_all))
        self._episode_return += reward

        if not self._step_curve_inited:
            self.init_step_curve_file(episode_now)
        self.append_step_mean_reward(
            episode=episode_now,
            actuation=actuation,
            sim_time=float(self.sim_time),
            mean_reward=reward,
            rewards_vec=rewards_all,
        )

        self.step_count += 1
        terminated = False
        truncated = (self.step_count >= self.max_steps)

        info: Dict[str, object] = {
            "episode": episode_now,
            "group_id": int(self.group_id),
            "actuation": actuation,
            "sim_time": float(self.sim_time),
            "raw_actions": raw_actions.astype(np.float32),
            "temps": seg_temps.astype(np.float32),
            "gen_flux": gen_flux,
            "local_flux": local_flux,
            "gen_ke": gen_ke,
            "rewards_all": rewards_all,
        }

        if truncated:
            self.append_episode_mean_return(episode_now, self._episode_return)
            self.episode += 1
            self.nc = None
            self._step_curve_inited = False

        return obs, reward, terminated, truncated, info

    def close(self):
        self.nc = None
        return
