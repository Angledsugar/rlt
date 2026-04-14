"""Abstract base class for RLT environments.

Defines the interface that all environments must implement
for use with the RLT online RL training loop.
"""

import abc

import numpy as np


class BaseEnv(abc.ABC):
    """Abstract environment interface for RLT.

    Environments must provide:
    - Standard gym-like step/reset interface.
    - Observation dict compatible with the VLA (openpi format).
    - Proprioceptive state extraction.
    - Sparse binary reward signal.
    """

    @abc.abstractmethod
    def reset(self) -> dict:
        """Reset environment and return initial observation.

        Returns:
            observation: Environment-specific observation dict.
        """

    @abc.abstractmethod
    def step(self, action_chunk: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Execute an action chunk in the environment.

        The action chunk is executed step by step at the control frequency.

        Args:
            action_chunk: [C, action_dim] - Action chunk to execute.

        Returns:
            (observation, reward, done, info)
            - reward: Sparse binary (+1 success, 0 otherwise).
            - done: Whether the episode has ended.
        """

    @abc.abstractmethod
    def get_observation_dict(self, obs: dict) -> dict:
        """Convert environment observation to VLA-compatible format.

        Must return a dict with keys matching openpi's Observation:
        - images: Dict[str, ndarray] with camera views.
        - image_masks: Dict[str, ndarray] with validity masks.
        - state: ndarray with robot state.
        - tokenized_prompt: ndarray with tokenized language instruction.
        - tokenized_prompt_mask: ndarray with prompt mask.

        Args:
            obs: Raw environment observation.

        Returns:
            VLA-compatible observation dict.
        """

    @abc.abstractmethod
    def get_proprio(self, obs: dict) -> np.ndarray:
        """Extract proprioceptive state from observation.

        Args:
            obs: Raw environment observation.

        Returns:
            proprio: [proprio_dim] - Proprioceptive state vector.
        """

    @property
    @abc.abstractmethod
    def action_dim(self) -> int:
        """Per-step action dimension."""

    @property
    @abc.abstractmethod
    def proprio_dim(self) -> int:
        """Proprioceptive state dimension."""
