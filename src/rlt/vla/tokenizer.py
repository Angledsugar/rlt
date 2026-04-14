"""VLA embedding extraction and action chunk tokenization.

Handles the interface between the frozen VLA and the RL token module:
1. Extracting VLA embeddings (dropping language tokens as per paper footnote 1).
2. Converting VLA outputs into action chunks for the RL policy.
"""

import jax
import jax.numpy as jnp


class VLATokenizer:
    """Extracts and processes VLA embeddings for RL token creation."""

    def __init__(self, action_dim: int, action_chunk_length: int):
        """Initialize tokenizer.

        Args:
            action_dim: Per-step action dimension (e.g., 14 for bimanual).
            action_chunk_length: Number of steps per action chunk C (paper: 10).
        """
        self.action_dim = action_dim
        self.action_chunk_length = action_chunk_length

    def flatten_action_chunk(self, action_chunk: jax.Array) -> jax.Array:
        """Flatten action chunk from [batch, C, action_dim] to [batch, C*action_dim].

        Args:
            action_chunk: [batch, C, action_dim]

        Returns:
            flattened: [batch, C * action_dim]
        """
        batch_size = action_chunk.shape[0]
        return action_chunk.reshape(batch_size, -1)

    def unflatten_action_chunk(self, flat_action: jax.Array) -> jax.Array:
        """Unflatten action from [batch, C*action_dim] to [batch, C, action_dim].

        Args:
            flat_action: [batch, C * action_dim]

        Returns:
            action_chunk: [batch, C, action_dim]
        """
        batch_size = flat_action.shape[0]
        return flat_action.reshape(batch_size, self.action_chunk_length, self.action_dim)

    def subsample_action_chunk(
        self,
        action_chunk: jax.Array,
        stride: int = 2,
    ) -> list[jax.Array]:
        """Subsample action chunk with given stride for replay buffer storage.

        Paper Section V: "we pick a stride of 2 and save transitions
        corresponding to <x_0, a_{0:C}, x_2, a_{2:C+2}, ...>"

        Args:
            action_chunk: [C, action_dim] single action chunk.
            stride: Subsampling stride (default 2).

        Returns:
            List of subsampled starting indices.
        """
        indices = list(range(0, self.action_chunk_length, stride))
        return indices

    def extract_rl_state(
        self,
        rl_token: jax.Array,
        proprio: jax.Array,
    ) -> jax.Array:
        """Combine RL token and proprioception into RL state vector.

        The RL state x = (z_rl, s^P) as defined in the paper.

        Args:
            rl_token: [batch, token_dim] - RL token from encoder.
            proprio: [batch, proprio_dim] - Proprioceptive state.

        Returns:
            state: [batch, token_dim + proprio_dim]
        """
        return jnp.concatenate([rl_token, proprio], axis=-1)
