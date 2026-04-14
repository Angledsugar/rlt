"""Simple simulation environment for testing the RLT pipeline.

This is a dummy environment that generates random observations
and rewards. It is used to verify the training pipeline without
requiring a real robot or simulator.
"""

import numpy as np

from rlt.envs.base import BaseEnv


class DummySimEnv(BaseEnv):
    """Dummy simulation environment for pipeline testing."""

    def __init__(
        self,
        action_dim: int = 14,
        proprio_dim: int = 14,
        max_steps: int = 100,
        image_size: tuple[int, int] = (224, 224),
        max_token_len: int = 200,
        prompt: str = "do something",
    ):
        self._action_dim = action_dim
        self._proprio_dim = proprio_dim
        self._max_steps = max_steps
        self._image_size = image_size
        self._max_token_len = max_token_len
        self._prompt = prompt
        self._step_count = 0
        self._rng = np.random.default_rng(42)

    def reset(self) -> dict:
        self._step_count = 0
        return self._get_obs()

    def step(self, action_chunk: np.ndarray) -> tuple[dict, float, bool, dict]:
        self._step_count += action_chunk.shape[0]
        done = self._step_count >= self._max_steps
        # Sparse binary reward: random success at end of episode.
        reward = float(self._rng.random() > 0.5) if done else 0.0
        obs = self._get_obs()
        return obs, reward, done, {}

    def get_observation_dict(self, obs: dict) -> dict:
        return {
            "images": obs["images"],
            "image_masks": obs["image_masks"],
            "state": obs["state"],
            "tokenized_prompt": obs["tokenized_prompt"],
            "tokenized_prompt_mask": obs["tokenized_prompt_mask"],
        }

    def get_policy_observation(self, obs: dict) -> dict:
        return obs["policy_observation"]

    def get_proprio(self, obs: dict) -> np.ndarray:
        return obs["state"][0]

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def proprio_dim(self) -> int:
        return self._proprio_dim

    def _get_obs(self) -> dict:
        """Generate a random observation."""
        h, w = self._image_size
        images = {
            "base_0_rgb": self._rng.random((1, h, w, 3)).astype(np.float32),
            "left_wrist_0_rgb": self._rng.random((1, h, w, 3)).astype(np.float32),
            "right_wrist_0_rgb": self._rng.random((1, h, w, 3)).astype(np.float32),
        }
        state = self._rng.random((1, self._proprio_dim)).astype(np.float32)
        return {
            "images": {
                "base_0_rgb": images["base_0_rgb"],
                "left_wrist_0_rgb": images["left_wrist_0_rgb"],
                "right_wrist_0_rgb": images["right_wrist_0_rgb"],
            },
            "image_masks": {
                "base_0_rgb": np.ones((1,), dtype=bool),
                "left_wrist_0_rgb": np.ones((1,), dtype=bool),
                "right_wrist_0_rgb": np.ones((1,), dtype=bool),
            },
            "state": state,
            "tokenized_prompt": np.zeros((1, self._max_token_len), dtype=np.int32),
            "tokenized_prompt_mask": np.zeros((1, self._max_token_len), dtype=bool),
            "policy_observation": {
                "images": {
                    "cam_high": np.moveaxis(images["base_0_rgb"][0], -1, 0),
                    "cam_left_wrist": np.moveaxis(images["left_wrist_0_rgb"][0], -1, 0),
                    "cam_right_wrist": np.moveaxis(images["right_wrist_0_rgb"][0], -1, 0),
                },
                "state": state[0],
                "prompt": self._prompt,
            },
        }
