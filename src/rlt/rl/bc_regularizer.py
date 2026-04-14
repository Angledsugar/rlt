"""BC (Behavior Cloning) regularization for the RL actor.

Implements the BC regularization term from Eq.5:
    L_bc = beta * ||a_{1:C} - a_hat_{1:C}||^2

where a_{1:C} is the actor's action chunk and a_hat_{1:C} is the VLA reference.
The coefficient beta controls how strongly the actor is regularized toward
the sampled VLA action.

Also implements reference action dropout (Section IV-B):
For a random subset of transitions, the reference action chunk is replaced
with zeros to prevent the actor from simply copying the reference.
"""

import jax
import jax.numpy as jnp


def compute_bc_loss(
    actor_action: jax.Array,
    reference_action: jax.Array,
) -> jax.Array:
    """Compute BC regularization loss.

    Args:
        actor_action: [batch, action_chunk_dim] - Actor's predicted action.
        reference_action: [batch, action_chunk_dim] - VLA reference action.

    Returns:
        bc_loss: Scalar mean squared error.
    """
    return jnp.mean(jnp.sum(jnp.square(actor_action - reference_action), axis=-1))


def apply_reference_dropout(
    rng: jax.Array,
    reference_action: jax.Array,
    dropout_rate: float = 0.5,
) -> jax.Array:
    """Apply reference action dropout.

    For a random subset of transitions, replace the reference action with zeros.
    This prevents the actor from simply copying the VLA action and forces it
    to maintain an independent action-generation pathway.

    Args:
        rng: PRNG key.
        reference_action: [batch, action_chunk_dim] - VLA reference actions.
        dropout_rate: Probability of zeroing out reference (default 0.5 per paper).

    Returns:
        masked_reference: [batch, action_chunk_dim] - Masked reference actions.
    """
    keep_mask = jax.random.bernoulli(
        rng,
        1.0 - dropout_rate,
        shape=(reference_action.shape[0], 1),
    )
    return reference_action * keep_mask
