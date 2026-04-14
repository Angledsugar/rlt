"""ALOHA environment wrapper for RLT online RL.

Wraps an ALOHA robot (real or simulated) to provide the BaseEnv interface
for the RLT training loop. Handles:
1. Observation formatting (cameras, proprioception) for VLA.
2. Action chunk execution at 50Hz.
3. Sparse binary reward from human supervisor or automatic detection.

For simulation testing, uses gym-aloha environments.
For real robot, connects to the ALOHA hardware via ROS/direct serial.
"""

import logging

import einops
import numpy as np

from rlt.envs.base import BaseEnv

logger = logging.getLogger("rlt")

# ALOHA action/state dimensions.
ALOHA_ACTION_DIM = 14
ALOHA_PROPRIO_DIM = 14
ALOHA_IMAGE_SIZE = (224, 224)

# Camera name mapping: ALOHA -> openpi internal.
CAMERA_MAPPING = {
    "cam_high": "base_0_rgb",
    "cam_left_wrist": "left_wrist_0_rgb",
    "cam_right_wrist": "right_wrist_0_rgb",
}


class AlohaSimEnv(BaseEnv):
    """ALOHA simulation environment wrapper using gym-aloha.

    Uses gym_aloha environments for testing the RLT pipeline
    without requiring real robot hardware.
    """

    def __init__(
        self,
        task: str = "gym_aloha/AlohaTransferCube-v0",
        prompt: str = "uncap the pen",
        max_episode_steps: int = 400,
        adapt_to_pi: bool = False,
        tokenizer_path: str | None = None,
    ):
        """Initialize ALOHA sim environment.

        Args:
            task: Gym environment ID.
            prompt: Language instruction for VLA.
            max_episode_steps: Maximum steps per episode.
            adapt_to_pi: Apply pi-space joint transforms.
            tokenizer_path: Path to SentencePiece tokenizer for prompt encoding.
        """
        import gymnasium as gym

        self._env = gym.make(task, max_episode_steps=max_episode_steps)
        self._prompt = prompt
        self._adapt_to_pi = adapt_to_pi
        self._step_count = 0
        self._max_steps = max_episode_steps

        # Pre-encode the prompt tokens.
        self._tokenized_prompt = None
        self._tokenized_prompt_mask = None
        if tokenizer_path is not None:
            self._setup_tokenizer(tokenizer_path)

    def _setup_tokenizer(self, tokenizer_path: str) -> None:
        """Load SentencePiece tokenizer and encode prompt."""
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.Load(tokenizer_path)
        tokens = sp.Encode(self._prompt)
        max_len = 200  # pi0.5 max_token_len
        padded = np.zeros(max_len, dtype=np.int32)
        mask = np.zeros(max_len, dtype=bool)
        length = min(len(tokens), max_len)
        padded[:length] = tokens[:length]
        mask[:length] = True
        self._tokenized_prompt = padded
        self._tokenized_prompt_mask = mask

    def reset(self) -> dict:
        obs, info = self._env.reset()
        self._step_count = 0
        return self._wrap_obs(obs)

    def step(self, action_chunk: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Execute action chunk step by step.

        Args:
            action_chunk: [C, 14] action chunk.

        Returns:
            (obs, reward, done, info)
        """
        total_reward = 0.0
        done = False
        info = {}

        for t in range(len(action_chunk)):
            action = action_chunk[t]
            if self._adapt_to_pi:
                action = self._action_to_aloha(action)

            obs, reward, terminated, truncated, step_info = self._env.step(action)
            total_reward += reward
            self._step_count += 1
            done = terminated or truncated

            if done:
                info["success"] = step_info.get("is_success", reward > 0)
                break

        # Sparse binary reward: +1 at end if successful.
        sparse_reward = 1.0 if info.get("success", False) else 0.0
        if not done:
            sparse_reward = 0.0

        return self._wrap_obs(obs), sparse_reward, done, info

    def get_observation_dict(self, obs: dict) -> dict:
        return {
            "images": obs["images"],
            "image_masks": obs["image_masks"],
            "state": obs["state"],
            "tokenized_prompt": obs["tokenized_prompt"],
            "tokenized_prompt_mask": obs["tokenized_prompt_mask"],
        }

    def get_policy_observation(self, obs: dict) -> dict:
        """Return raw ALOHA-format observation for openpi policy transforms."""
        return obs["policy_observation"]

    def get_proprio(self, obs: dict) -> np.ndarray:
        state = obs["state"]
        if state.ndim == 2:
            return state[0]
        return state

    @property
    def action_dim(self) -> int:
        return ALOHA_ACTION_DIM

    @property
    def proprio_dim(self) -> int:
        return ALOHA_PROPRIO_DIM

    def _wrap_obs(self, raw_obs) -> dict:
        """Convert gym observation to VLA-compatible format."""
        images = {}
        image_masks = {}
        policy_images = {}

        if isinstance(raw_obs, dict) and "pixels" in raw_obs:
            # gym-aloha format: raw_obs["pixels"] is a dict of camera images.
            pixels = raw_obs["pixels"]
            for aloha_cam, pi_cam in CAMERA_MAPPING.items():
                if aloha_cam in pixels:
                    img = self._process_image(np.asarray(pixels[aloha_cam]))
                    images[pi_cam] = img[None]  # Add batch dim.
                    image_masks[pi_cam] = np.array([True])
                    policy_images[aloha_cam] = np.moveaxis(img, -1, 0)
                else:
                    images[pi_cam] = np.zeros((1, *ALOHA_IMAGE_SIZE, 3), dtype=np.uint8)
                    image_masks[pi_cam] = np.array([False])
                    if aloha_cam == "cam_high":
                        policy_images[aloha_cam] = np.zeros((3, *ALOHA_IMAGE_SIZE), dtype=np.uint8)

            state = np.asarray(raw_obs.get("agent_pos", np.zeros(ALOHA_PROPRIO_DIM)), dtype=np.float32)
        elif isinstance(raw_obs, np.ndarray):
            # Flat observation: just state, no images.
            for pi_cam in CAMERA_MAPPING.values():
                images[pi_cam] = np.zeros((1, *ALOHA_IMAGE_SIZE, 3), dtype=np.uint8)
                image_masks[pi_cam] = np.array([False])
            state = raw_obs[:ALOHA_PROPRIO_DIM].astype(np.float32)
            policy_images["cam_high"] = np.zeros((3, *ALOHA_IMAGE_SIZE), dtype=np.uint8)
        else:
            raise ValueError(f"Unexpected observation type: {type(raw_obs)}")

        policy_state = state.copy()

        if self._adapt_to_pi:
            state = self._state_to_pi(state)

        if state.ndim == 1:
            state = state[None]  # Add batch dim.

        # Tokenized prompt.
        if self._tokenized_prompt is not None:
            tok_prompt = self._tokenized_prompt[None]
            tok_mask = self._tokenized_prompt_mask[None]
        else:
            tok_prompt = np.zeros((1, 200), dtype=np.int32)
            tok_mask = np.zeros((1, 200), dtype=bool)

        return {
            "images": images,
            "image_masks": image_masks,
            "state": state,
            "tokenized_prompt": tok_prompt,
            "tokenized_prompt_mask": tok_mask,
            "policy_observation": {
                "images": policy_images,
                "state": policy_state,
                "prompt": self._prompt,
            },
        }

    def _process_image(self, img: np.ndarray) -> np.ndarray:
        """Resize and format image."""
        import cv2

        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = einops.rearrange(img, "c h w -> h w c")
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * np.clip(img, 0, 1)).astype(np.uint8)
        h, w = ALOHA_IMAGE_SIZE
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return img

    def _state_to_pi(self, state: np.ndarray) -> np.ndarray:
        """Convert ALOHA state to pi-space."""
        from rlt.data.aloha_dataset import _gripper_to_angular
        from rlt.data.aloha_dataset import _joint_flip_mask

        state = _joint_flip_mask() * state
        if state.ndim == 1:
            state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
        else:
            state[:, [6, 13]] = _gripper_to_angular(state[:, [6, 13]])
        return state

    def _action_to_aloha(self, action: np.ndarray) -> np.ndarray:
        """Convert pi-space action back to ALOHA space for execution."""
        from rlt.data.aloha_dataset import _gripper_from_angular
        from rlt.data.aloha_dataset import _joint_flip_mask

        action = _joint_flip_mask() * action
        if action.ndim == 1:
            action[[6, 13]] = _gripper_from_angular(action[[6, 13]])
        else:
            action[:, [6, 13]] = _gripper_from_angular(action[:, [6, 13]])
        return action
