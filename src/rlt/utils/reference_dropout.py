"""Reference action dropout utility.

Implements reference action dropout from Section IV-B.
For a random subset of transitions in each training batch,
the reference chunk is replaced with zeros before passing to the actor.
This forces the actor to maintain an independent action-generation pathway.
"""

import jax
import jax.numpy as jnp


def apply_reference_dropout(
    rng: jax.Array,
    reference_action: jax.Array,
    dropout_rate: float = 0.5,
) -> jax.Array:
    """Apply reference action dropout.

    Args:
        rng: PRNG key.
        reference_action: [batch, action_chunk_dim]
        dropout_rate: Probability of zeroing out (default 0.5).

    Returns:
        Masked reference action.
    """
    keep_mask = jax.random.bernoulli(
        rng,
        1.0 - dropout_rate,
        shape=(reference_action.shape[0], 1),
    )
    return reference_action * keep_mask
