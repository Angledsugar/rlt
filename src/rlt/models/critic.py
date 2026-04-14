"""TD3 Critic network (Q_psi).

Implements the twin Q-network critic from Section IV-B (Eq.3).
The critic takes the state (RL token + proprioception) and action chunk
as input and estimates Q-value.

    Q_psi(x, a_{1:C}) in R

Uses TD3-style twin Q-networks following Fujimoto et al.
The critic additionally takes in the reference action chunk (masked at 50%).
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from rlt.config.actor_critic import CriticConfig


class Critic(nnx.Module):
    """Single Q-network for state-action value estimation."""

    def __init__(
        self,
        config: CriticConfig,
        rl_token_dim: int,
        proprio_dim: int,
        action_chunk_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        # Input: rl_token + proprio + action_chunk + reference_action_chunk.
        input_dim = rl_token_dim + proprio_dim + action_chunk_dim + action_chunk_dim

        layers = []
        in_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim
        self.layers = layers
        self.output_layer = nnx.Linear(in_dim, 1, rngs=rngs)

    def __call__(
        self,
        rl_token: jax.Array,
        proprio: jax.Array,
        action: jax.Array,
        reference_action: jax.Array,
    ) -> jax.Array:
        """Estimate Q-value.

        Args:
            rl_token: [batch, rl_token_dim]
            proprio: [batch, proprio_dim]
            action: [batch, action_chunk_dim] - Flattened action chunk.
            reference_action: [batch, action_chunk_dim] - VLA reference (possibly masked).

        Returns:
            q_value: [batch, 1]
        """
        x = jnp.concatenate([rl_token, proprio, action, reference_action], axis=-1)
        for layer in self.layers:
            x = layer(x)
            x = nnx.relu(x)
        return self.output_layer(x)


class DoubleCritic(nnx.Module):
    """Twin Q-networks (TD3 style).

    Uses the minimum of two Q-networks to reduce overestimation bias.
    Follows Fujimoto et al. and paper Section IV-B.
    """

    def __init__(
        self,
        config: CriticConfig,
        rl_token_dim: int,
        proprio_dim: int,
        action_chunk_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.q1 = Critic(config, rl_token_dim, proprio_dim, action_chunk_dim, rngs=rngs)
        self.q2 = Critic(config, rl_token_dim, proprio_dim, action_chunk_dim, rngs=rngs)

    def __call__(
        self,
        rl_token: jax.Array,
        proprio: jax.Array,
        action: jax.Array,
        reference_action: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute both Q-values.

        Returns:
            (q1, q2): tuple of [batch, 1] Q-value estimates.
        """
        q1 = self.q1(rl_token, proprio, action, reference_action)
        q2 = self.q2(rl_token, proprio, action, reference_action)
        return q1, q2

    def min_q(
        self,
        rl_token: jax.Array,
        proprio: jax.Array,
        action: jax.Array,
        reference_action: jax.Array,
    ) -> jax.Array:
        """Compute minimum of twin Q-values (for target computation)."""
        q1, q2 = self(rl_token, proprio, action, reference_action)
        return jnp.minimum(q1, q2)
