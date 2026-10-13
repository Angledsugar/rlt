"""Gymnasium environment for ALOHA pen uncapping.

Registers as ``gym_aloha/AlohaPenUncap-v0``.
Follows the same API contract as the upstream gym-aloha environments
so it can be used as a drop-in replacement in the RLT training loop.
"""

import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces

from gym_aloha.constants import ACTIONS, DT, JOINTS

from rlt.envs.pen_uncap_task import PEN_POSE, PenUncapTask, sample_pen_pose

_ASSETS_DIR = None


def _get_assets_dir():
    """Return the gym-aloha assets directory (where scene.xml / vx300s*.xml live)."""
    global _ASSETS_DIR
    if _ASSETS_DIR is None:
        from gym_aloha.constants import ASSETS_DIR
        _ASSETS_DIR = ASSETS_DIR
    return _ASSETS_DIR


def _pen_uncap_xml_path():
    """Return the path to our pen_uncap XML, symlinking robot assets as needed."""
    import shutil
    from pathlib import Path

    local_assets = Path(__file__).parent / "assets"
    gym_assets = _get_assets_dir()

    # Symlink shared gym-aloha assets so MuJoCo <include> resolves them.
    for name in (
        "scene.xml",
        "vx300s_dependencies.xml",
        "vx300s_left.xml",
        "vx300s_right.xml",
        "tabletop.stl",
    ):
        dst = local_assets / name
        src = gym_assets / name
        if not dst.exists() and src.exists():
            dst.symlink_to(src)

    # Also link all .stl mesh files referenced by vx300s_dependencies.xml.
    for stl in gym_assets.glob("*.stl"):
        dst = local_assets / stl.name
        if not dst.exists():
            dst.symlink_to(stl)

    return str(local_assets / "bimanual_viperx_pen_uncap.xml")


class PenUncapEnv(gym.Env):
    """Gymnasium environment for ALOHA bimanual pen uncapping."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
    ):
        super().__init__()
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height

        xml_path = _pen_uncap_xml_path()
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PenUncapTask()
        self._env = control.Environment(
            physics, task, time_limit=float("inf"), control_timestep=DT, n_sub_steps=None, flat_observation=False,
        )

        if obs_type == "pixels":
            self.observation_space = spaces.Dict({
                "top": spaces.Box(0, 255, (observation_height, observation_width, 3), np.uint8),
            })
        elif obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict({
                "pixels": spaces.Dict({
                    "top": spaces.Box(0, 255, (observation_height, observation_width, 3), np.uint8),
                }),
                "agent_pos": spaces.Box(-1000.0, 1000.0, (len(JOINTS),), np.float64),
            })

        self.action_space = spaces.Box(-1, 1, (len(ACTIONS),), np.float32)

    def render(self):
        return self._env.physics.render(
            height=self.visualization_height,
            width=self.visualization_width,
            camera_id="top",
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        PEN_POSE[0] = sample_pen_pose(seed)
        raw_obs = self._env.reset()
        return self._format_obs(raw_obs.observation), {"is_success": False}

    def step(self, action):
        assert action.ndim == 1
        _, reward, _, raw_obs = self._env.step(action)
        terminated = is_success = reward == 4
        obs = self._format_obs(raw_obs)
        return obs, reward, terminated, False, {"is_success": is_success}

    def _format_obs(self, raw_obs):
        if self.obs_type == "pixels":
            return {"top": raw_obs["images"]["top"].copy()}
        elif self.obs_type == "pixels_agent_pos":
            return {
                "pixels": {"top": raw_obs["images"]["top"].copy()},
                "agent_pos": raw_obs["qpos"],
            }

    def close(self):
        pass
