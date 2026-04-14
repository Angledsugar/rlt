"""Action chunking utilities.

Implements action chunk creation and subsampling as described in Section V.
The RL policy uses action chunks of length C=10 with 14-dimensional actions,
resulting in 140-dimensional chunked actions.

Subsampling with stride=2 creates overlapping transitions:
    <x_0, a_{0:C}>, <x_2, a_{2:C+2}>, <x_4, a_{4:C+4}>, ...
"""

import jax.numpy as jnp
import numpy as np


def chunk_actions(
    actions: np.ndarray,
    chunk_length: int,
    stride: int = 1,
) -> list[np.ndarray]:
    """Create action chunks from a sequence of single-step actions.

    Args:
        actions: [T, action_dim] - Sequence of actions.
        chunk_length: C - Number of steps per chunk.
        stride: Step between chunk start positions.

    Returns:
        List of [chunk_length, action_dim] action chunks.
    """
    T = len(actions)
    chunks = []
    for start in range(0, T - chunk_length + 1, stride):
        chunk = actions[start : start + chunk_length]
        chunks.append(chunk)
    return chunks


def unchunk_actions(
    chunked_action: np.ndarray | jnp.ndarray,
    action_dim: int,
    chunk_length: int,
) -> np.ndarray | jnp.ndarray:
    """Reshape flattened action chunk back to [chunk_length, action_dim].

    Args:
        chunked_action: [chunk_length * action_dim] - Flattened action chunk.
        action_dim: Per-step action dimension.
        chunk_length: Number of steps per chunk.

    Returns:
        [chunk_length, action_dim] - Reshaped action chunk.
    """
    return chunked_action.reshape(chunk_length, action_dim)
