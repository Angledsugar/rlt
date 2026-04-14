"""End-to-end RLT training script for ALOHA pen_uncap task.

Runs both phases:
  Phase 1: Train RL token encoder-decoder on pen_uncap demonstrations.
  Phase 2: Online RL with frozen VLA + frozen RL token.

Usage:
    # Phase 1 only (RL token training on demo data):
    uv run python scripts/train_pen_uncap.py --phase 1

    # Phase 2 only (online RL, requires Phase 1 checkpoint):
    uv run python scripts/train_pen_uncap.py --phase 2 --rl-token-checkpoint checkpoints/pen_uncap/rl_token/5000

    # Both phases sequentially:
    uv run python scripts/train_pen_uncap.py --phase both
"""

# ruff: noqa: I001

import os

# GPU visibility: use only CUDA 0,1.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
# Enable HF Transfer for faster downloads (Rust-based multi-connection).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import logging
import sys
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro

from rlt.config.base import RLTConfig
from rlt.config.actor_critic import ActorConfig
from rlt.config.actor_critic import CriticConfig
from rlt.config.rl_token import RLTokenConfig
from rlt.config.training import TrainingConfig
from rlt.models.rl_token import RLTokenModule
from rlt.training.utils import create_checkpoint_manager
from rlt.training.utils import log_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("rlt")


def make_pen_uncap_config() -> RLTConfig:
    """Create RLT config for pen_uncap task."""
    return RLTConfig(
        task_name="aloha_pen_uncap",
        vla_checkpoint="gs://openpi-assets/checkpoints/pi05_base/params",
        action_dim=14,
        action_chunk_length=10,
        proprio_dim=14,
        rl_token=RLTokenConfig(
            token_dim=2048,
            vla_embed_dim=2048,
            learning_rate=1e-4,
            num_train_steps=5000,
            batch_size=32,
        ),
        actor=ActorConfig(hidden_dims=(256, 256), learning_rate=3e-4),
        critic=CriticConfig(hidden_dims=(256, 256), learning_rate=3e-4),
        training=TrainingConfig(
            beta=1.0,
            reference_action_dropout=0.5,
            num_warmup_episodes=50,
            max_env_steps=50_000,
            wandb_project="rlt-pen-uncap",
        ),
    )


def train_rl_token_pen_uncap(
    config: RLTConfig,
    rng: jax.Array,
    embeddings_dir: str = "data/pen_uncap_embeddings",
    checkpoint_dir: str = "checkpoints/pen_uncap/rl_token",
) -> RLTokenModule:
    """Phase 1: Train RL token on pre-extracted VLA embeddings.

    Requires Phase 0 (extract_embeddings.py) to have been run first.
    Loads .npy files — no VLA needed, lightweight and fast.
    """
    logger.info("=" * 60)
    logger.info("Phase 1: RL Token Training (from pre-extracted embeddings)")
    logger.info("=" * 60)
    sys.stdout.flush()

    rng, model_rng = jax.random.split(rng)

    # Load pre-extracted embeddings.
    emb_path = Path(embeddings_dir)
    embeddings_all = np.load(emb_path / "embeddings.npy")
    logger.info("Loaded embeddings: %s", embeddings_all.shape)
    sys.stdout.flush()

    num_frames, seq_len, embed_dim = embeddings_all.shape
    assert embed_dim == config.rl_token.vla_embed_dim, (
        f"Embedding dim mismatch: file={embed_dim}, config={config.rl_token.vla_embed_dim}"
    )

    # Create RL token module.
    rl_token = RLTokenModule(config.rl_token, rngs=nnx.Rngs(model_rng))
    optimizer = nnx.Optimizer(rl_token, optax.adam(config.rl_token.learning_rate))

    # Checkpoint manager.
    ckpt_manager = create_checkpoint_manager(checkpoint_dir)

    # Training loop — random batches from pre-extracted embeddings.
    batch_size = config.rl_token.batch_size
    np_rng = np.random.default_rng(config.training.seed)

    for step in range(config.rl_token.num_train_steps):
        # Sample random batch of embeddings.
        indices = np_rng.integers(0, num_frames, size=batch_size)
        vla_embeddings = jnp.asarray(embeddings_all[indices])

        # Train RL token encoder-decoder.
        def loss_fn(model):
            return model.reconstruction_loss(vla_embeddings)

        loss, grads = nnx.value_and_grad(loss_fn)(rl_token)
        optimizer.update(grads)

        if step % 100 == 0:
            log_metrics(step, {"rl_token/reconstruction_loss": float(loss)})

        if step % 1000 == 0 and step > 0:
            ckpt_manager.save(step, args=nnx.state(rl_token))
            logger.info("Checkpoint saved at step %d", step)

    # Save final.
    ckpt_manager.save(config.rl_token.num_train_steps, args=nnx.state(rl_token))
    logger.info("Phase 1 complete. Final checkpoint: %s/%d", checkpoint_dir, config.rl_token.num_train_steps)

    return rl_token


def train_online_rl_pen_uncap(
    config: RLTConfig,
    rl_token_module: RLTokenModule,
    rng: jax.Array,
) -> None:
    """Phase 2: Online RL on pen_uncap task.

    Uses the frozen VLA + frozen RL token to train a lightweight
    actor-critic online.
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Online RL for pen_uncap")
    logger.info("=" * 60)

    from rlt.envs.aloha_env import AlohaSimEnv
    from rlt.training.train_online_rl import train_online_rl

    # For sim testing, use gym-aloha. For real robot, replace with real env.
    env = AlohaSimEnv(
        task="gym_aloha/AlohaTransferCube-v0",
        prompt="uncap the pen",
        # train_online_rl uses openpi policy output transforms, so actions are
        # already in the environment/Aloha action space.
        adapt_to_pi=False,
    )

    train_online_rl(config, rl_token_module, env, rng)


def main(
    phase: str = "both",
    embeddings_dir: str = "data/pen_uncap_embeddings",
    rl_token_checkpoint: str | None = None,
    seed: int = 42,
) -> None:
    """Run RLT training for ALOHA pen_uncap task.

    Workflow:
        Phase 0: uv run python scripts/extract_embeddings.py  (separate script)
        Phase 1: Train RL token from pre-extracted embeddings
        Phase 2: Online RL with frozen VLA + RL token

    Args:
        phase: Which phase to run: "1", "2", or "both".
        embeddings_dir: Directory with pre-extracted .npy files from Phase 0.
        rl_token_checkpoint: Path to trained RL token checkpoint (required for phase 2 only).
        seed: Random seed.
    """
    config = make_pen_uncap_config()
    rng = jax.random.PRNGKey(seed)

    if phase in ("1", "both"):
        rng, phase1_rng = jax.random.split(rng)
        rl_token_module = train_rl_token_pen_uncap(config, phase1_rng, embeddings_dir=embeddings_dir)

    if phase in ("2", "both"):
        rng, phase2_rng = jax.random.split(rng)

        if phase == "2":
            # Load RL token from checkpoint.
            if rl_token_checkpoint is None:
                raise ValueError("--rl-token-checkpoint required for phase 2")
            rng, model_rng = jax.random.split(rng)
            rl_token_module = RLTokenModule(config.rl_token, rngs=nnx.Rngs(model_rng))
            checkpointer = ocp.StandardCheckpointer()
            state = checkpointer.restore(rl_token_checkpoint, target=nnx.state(rl_token_module))
            nnx.update(rl_token_module, state)

        train_online_rl_pen_uncap(config, rl_token_module, phase2_rng)


if __name__ == "__main__":
    tyro.cli(main)
