"""RL Actor network (pi_theta).

Implements the lightweight MLP actor from Section IV-B (Eq.4-5).
The actor takes the RL token representation and proprioceptive state as input,
and is conditioned on the VLA reference action chunk to produce refined actions.

    pi_theta(a_{1:C} | x, a_hat_{1:C}) = N(mu_theta(x, a_hat), sigma^2 I)

where x = (z_rl, s^P) combines RL token and proprioception,
and conditioning on a_hat exposes the actor to VLA predictions directly.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from rlt.config.actor_critic import ActorConfig


class Actor(nnx.Module):
    """Gaussian MLP actor for online RL.

    Input: [rl_token, proprioception, reference_action_chunk]
    Output: mean of Gaussian policy over action chunks.
    """

    def __init__(
        self,
        config: ActorConfig,
        rl_token_dim: int,
        proprio_dim: int,
        action_chunk_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.action_chunk_dim = action_chunk_dim

        # Input dim: rl_token + proprioception + reference action chunk.
        input_dim = rl_token_dim + proprio_dim + action_chunk_dim

        # Build MLP layers.
        layers = []
        in_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nnx.Linear(in_dim, hidden_dim, rngs=rngs))
            in_dim = hidden_dim
        self.layers = layers
        self.output_layer = nnx.Linear(in_dim, action_chunk_dim, rngs=rngs)

        # Fixed log standard deviation.
        self.log_std = nnx.Param(jnp.full((action_chunk_dim,), jnp.log(config.init_std)))

    def __call__(
        self,
        rl_token: jax.Array,
        proprio: jax.Array,
        reference_action: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass.

        Args:
            rl_token: [batch, rl_token_dim] - RL token from frozen VLA.
            proprio: [batch, proprio_dim] - Proprioceptive state.
            reference_action: [batch, action_chunk_dim] - VLA reference action chunk (flattened).

        Returns:
            action_mean: [batch, action_chunk_dim] - Mean of Gaussian policy.
            action_std: [batch, action_chunk_dim] - Std of Gaussian policy.
        """
        x = jnp.concatenate([rl_token, proprio, reference_action], axis=-1)
        for layer in self.layers:
            x = layer(x)
            x = nnx.relu(x)
        action_mean = self.output_layer(x)
        action_std = jnp.broadcast_to(jnp.exp(self.log_std.value), action_mean.shape)
        return action_mean, action_std

    def sample(
        self,
        rng: jax.Array,
        rl_token: jax.Array,
        proprio: jax.Array,
        reference_action: jax.Array,
    ) -> jax.Array:
        """Sample action from the Gaussian policy.

        Args:
            rng: PRNG key.
            rl_token: [batch, rl_token_dim]
            proprio: [batch, proprio_dim]
            reference_action: [batch, action_chunk_dim]

        Returns:
            action: [batch, action_chunk_dim] - Sampled action chunk.
        """
        mean, std = self(rl_token, proprio, reference_action)
        noise = jax.random.normal(rng, mean.shape)
        return mean + std * noise

    def get_mean(
        self,
        rl_token: jax.Array,
        proprio: jax.Array,
        reference_action: jax.Array,
    ) -> jax.Array:
        """Get deterministic action (mean of policy).

        Used during evaluation.
        """
        mean, _ = self(rl_token, proprio, reference_action)
        return mean
