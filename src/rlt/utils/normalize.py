"""Running normalization for states and actions.

Maintains running statistics for online normalization
of RL token, proprioception, and action values.
"""

import numpy as np


class RunningNormalizer:
    """Online running mean/std normalizer."""

    def __init__(self, dim: int, eps: float = 1e-8):
        """Initialize normalizer.

        Args:
            dim: Feature dimension.
            eps: Small constant for numerical stability.
        """
        self.dim = dim
        self.eps = eps
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        """Update running statistics with a batch of observations.

        Uses Welford's online algorithm for numerical stability.

        Args:
            x: [batch, dim] - Batch of observations.
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize observations using running statistics.

        Args:
            x: [..., dim] - Observations to normalize.

        Returns:
            Normalized observations.
        """
        std = np.sqrt(self.var + self.eps)
        return (x - self.mean) / std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize observations.

        Args:
            x: [..., dim] - Normalized observations.

        Returns:
            Original-scale observations.
        """
        std = np.sqrt(self.var + self.eps)
        return x * std + self.mean
