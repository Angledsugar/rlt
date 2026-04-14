"""Phase 1: RL Token training.

Trains the encoder-decoder transformer (phi) to compress VLA embeddings
into a compact RL token representation.

Loss function (Eq.2):
    L_co = E_D [ sum_i || (d_theta(z_rl, z_hat_{1:i-1}))_i - z_i ||^2 ]

Optionally combined with VLA supervised fine-tuning:
    phi, theta_vla = argmin L_co(phi) + alpha * L_vla(theta_vla)

After training, both theta_vla and phi are frozen for online RL.
"""

import logging

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from rlt.config.base import RLTConfig
from rlt.models.rl_token import RLTokenModule
from rlt.training.utils import create_checkpoint_manager
from rlt.training.utils import log_metrics
from rlt.vla.backbone import VLABackbone

logger = logging.getLogger("rlt")


def create_rl_token_train_state(
    config: RLTConfig,
    rng: jax.Array,
) -> tuple[RLTokenModule, nnx.Optimizer]:
    """Create RL token module and optimizer.

    Args:
        config: RLT configuration.
        rng: PRNG key.

    Returns:
        (rl_token_module, optimizer)
    """
    rl_token = RLTokenModule(config.rl_token, rngs=nnx.Rngs(rng))
    optimizer = nnx.Optimizer(rl_token, optax.adam(config.rl_token.learning_rate))
    return rl_token, optimizer


def rl_token_train_step(
    rl_token: RLTokenModule,
    optimizer: nnx.Optimizer,
    vla_embeddings: jax.Array,
) -> jax.Array:
    """Single training step for RL token encoder-decoder.

    Args:
        rl_token: RL token module.
        optimizer: Optimizer state.
        vla_embeddings: [batch, seq_len, vla_embed_dim] - Extracted VLA embeddings.

    Returns:
        loss: Scalar reconstruction loss.
    """
    def loss_fn(model):
        return model.reconstruction_loss(vla_embeddings)

    loss, grads = nnx.value_and_grad(loss_fn)(rl_token)
    optimizer.update(grads)
    return loss


def train_rl_token(
    config: RLTConfig,
    demo_data_path: str,
    rng: jax.Array,
) -> RLTokenModule:
    """Train the RL token encoder-decoder on demonstration data.

    Phase 1 of the RLT pipeline:
    1. Load demonstration dataset.
    2. Extract VLA embeddings from frozen VLA.
    3. Train encoder-decoder to minimize reconstruction loss.

    Args:
        config: RLT configuration.
        demo_data_path: Path to demonstration data.
        rng: PRNG key.

    Returns:
        Trained RLTokenModule (frozen after this phase).
    """
    rng, model_rng, vla_rng = jax.random.split(rng, 3)

    logger.info("Phase 1: Training RL token encoder-decoder")
    logger.info("Config: %s", config.rl_token)

    # Load frozen VLA backbone.
    vla = VLABackbone(config.vla_checkpoint, rng=vla_rng)

    # Create RL token module and optimizer.
    rl_token, optimizer = create_rl_token_train_state(config, model_rng)

    # Create checkpoint manager.
    ckpt_manager = create_checkpoint_manager("checkpoints/rl_token")

    # Training loop.
    # In practice, this would iterate over the demonstration dataset.
    # Here we show the structure; actual data loading depends on the task.
    for step in range(config.rl_token.num_train_steps):
        rng, step_rng = jax.random.split(rng)

        # TODO: Load batch of demonstrations and extract VLA embeddings.
        # vla_embeddings = vla.extract_embeddings(demo_batch)
        # For now, create placeholder to show the training structure.
        vla_embeddings = jax.random.normal(
            step_rng,
            (config.rl_token.batch_size, 50, config.rl_token.vla_embed_dim),
        )

        loss = rl_token_train_step(rl_token, optimizer, vla_embeddings)

        if step % 100 == 0:
            log_metrics(step, {"rl_token/reconstruction_loss": float(loss)})

        if step % 1000 == 0 and step > 0:
            ckpt_manager.save(step, args=nnx.state(rl_token))
            logger.info("Step %d: loss = %.6f", step, float(loss))

    # Save final checkpoint.
    ckpt_manager.save(config.rl_token.num_train_steps, args=nnx.state(rl_token))
    logger.info("RL token training complete.")

    return rl_token
