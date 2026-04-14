"""Off-policy replay buffer for action chunk transitions.

Stores transitions of the form:
    <x_t, a_{t:t+C}, reward, x_{t+1}>

where x_t = (z_rl(s_t), s^P_t) is the RL state (RL token + proprioception),
and a_{t:t+C} is the action chunk.

Supports:
- Action chunk subsampling (stride=2) for increased data efficiency.
- Mixed storage of autonomous rollouts and human interventions.
- Sparse binary rewards (+1 success, 0 otherwise).
"""

import numpy as np


class ReplayBuffer:
    """Numpy-based circular replay buffer for off-policy RL."""

    def __init__(
        self,
        capacity: int,
        rl_token_dim: int,
        proprio_dim: int,
        action_chunk_dim: int,
    ):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions.
            rl_token_dim: Dimension of RL token.
            proprio_dim: Dimension of proprioceptive state.
            action_chunk_dim: Flattened action chunk dimension (action_dim * C).
        """
        self.capacity = capacity
        self.size = 0
        self.ptr = 0

        state_dim = rl_token_dim + proprio_dim

        # Pre-allocate arrays.
        self.rl_tokens = np.zeros((capacity, rl_token_dim), dtype=np.float32)
        self.proprios = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_chunk_dim), dtype=np.float32)
        self.reference_actions = np.zeros((capacity, action_chunk_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_rl_tokens = np.zeros((capacity, rl_token_dim), dtype=np.float32)
        self.next_proprios = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        rl_token: np.ndarray,
        proprio: np.ndarray,
        action: np.ndarray,
        reference_action: np.ndarray,
        reward: float,
        next_rl_token: np.ndarray,
        next_proprio: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition to the buffer.

        Args:
            rl_token: [rl_token_dim] - Current RL token.
            proprio: [proprio_dim] - Current proprioception.
            action: [action_chunk_dim] - Executed action chunk (flattened).
            reference_action: [action_chunk_dim] - VLA reference action chunk.
            reward: Scalar reward (+1 for success, 0 otherwise).
            next_rl_token: [rl_token_dim] - Next RL token.
            next_proprio: [proprio_dim] - Next proprioception.
            done: Whether the episode is done.
        """
        idx = self.ptr

        self.rl_tokens[idx] = rl_token
        self.proprios[idx] = proprio
        self.actions[idx] = action
        self.reference_actions[idx] = reference_action
        self.rewards[idx] = reward
        self.next_rl_tokens[idx] = next_rl_token
        self.next_proprios[idx] = next_proprio
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_episode(
        self,
        rl_tokens: np.ndarray,
        proprios: np.ndarray,
        actions: np.ndarray,
        reference_actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        stride: int = 2,
    ) -> None:
        """Add an entire episode with action chunk subsampling.

        Paper Section V: subsampling with stride=2 to increase data.

        Args:
            rl_tokens: [T, rl_token_dim]
            proprios: [T, proprio_dim]
            actions: [T, action_chunk_dim]
            reference_actions: [T, action_chunk_dim]
            rewards: [T]
            dones: [T]
            stride: Subsampling stride.
        """
        T = len(rl_tokens)
        for t in range(0, T - 1, stride):
            next_t = min(t + stride, T - 1)
            self.add(
                rl_token=rl_tokens[t],
                proprio=proprios[t],
                action=actions[t],
                reference_action=reference_actions[t],
                reward=rewards[t],
                next_rl_token=rl_tokens[next_t],
                next_proprio=proprios[next_t],
                done=dones[t],
            )

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> dict:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.
            rng: Numpy random generator (optional).

        Returns:
            Dict with keys: rl_token, proprio, action, reference_action,
                           reward, next_rl_token, next_proprio, done.
        """
        if rng is None:
            rng = np.random.default_rng()

        indices = rng.integers(0, self.size, size=batch_size)

        return {
            "rl_token": self.rl_tokens[indices],
            "proprio": self.proprios[indices],
            "action": self.actions[indices],
            "reference_action": self.reference_actions[indices],
            "reward": self.rewards[indices],
            "next_rl_token": self.next_rl_tokens[indices],
            "next_proprio": self.next_proprios[indices],
            "done": self.dones[indices],
        }

    def __len__(self) -> int:
        return self.size
