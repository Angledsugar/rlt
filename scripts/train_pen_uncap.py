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

import dataclasses
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
    # VLA outputs bfloat16 which numpy stores as raw bytes (|V2). Convert to float32.
    if embeddings_all.dtype.kind == "V":
        embeddings_all = embeddings_all.view(jnp.bfloat16).astype(np.float32)
    logger.info("Loaded embeddings: %s (dtype=%s)", embeddings_all.shape, embeddings_all.dtype)
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
            ckpt_manager.save(step, args=ocp.args.StandardSave(nnx.state(rl_token)))
            logger.info("Checkpoint saved at step %d", step)

    # Save final.
    ckpt_manager.save(config.rl_token.num_train_steps, args=ocp.args.StandardSave(nnx.state(rl_token)))
    logger.info("Phase 1 complete. Final checkpoint: %s/%d", checkpoint_dir, config.rl_token.num_train_steps)

    return rl_token


def train_online_rl_pen_uncap(
    config: RLTConfig,
    rl_token_module: RLTokenModule,
    rng: jax.Array,
    checkpoint_dir: str = "checkpoints/pen_uncap/online_rl",
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

    # Use pen_uncap simulation environment.
    # max_episode_steps=100 keeps episodes short (~10 VLA calls/episode)
    # since each VLA forward pass takes ~15s on RTX 3090.
    env = AlohaSimEnv(
        task="gym_aloha/AlohaPenUncap-v0",
        prompt="uncap the pen",
        adapt_to_pi=False,
        max_episode_steps=100,
    )

    train_online_rl(config, rl_token_module, env, rng, checkpoint_dir=checkpoint_dir)


def main(
    phase: str = "both",
    embeddings_dir: str = "data/pen_uncap_embeddings",
    rl_token_checkpoint: str | None = None,
    seed: int = 42,
    # --- Phase 1 overrides ---
    rl_token_lr: float | None = None,
    rl_token_batch_size: int | None = None,
    rl_token_num_steps: int | None = None,
    # --- Phase 2 overrides ---
    beta: float | None = None,
    reference_action_dropout: float | None = None,
    utd_ratio: int | None = None,
    num_warmup_episodes: int | None = None,
    actor_lr: float | None = None,
    critic_lr: float | None = None,
    action_chunk_stride: int | None = None,
    max_env_steps: int | None = None,
    # --- Output ---
    run_name: str | None = None,
    checkpoint_base: str = "checkpoints/pen_uncap",
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
        rl_token_lr: Override RL token learning rate.
        rl_token_batch_size: Override RL token batch size.
        rl_token_num_steps: Override RL token training steps.
        beta: Override BC regularization coefficient.
        reference_action_dropout: Override reference action dropout.
        utd_ratio: Override update-to-data ratio.
        num_warmup_episodes: Override warmup episodes.
        actor_lr: Override actor learning rate.
        critic_lr: Override critic learning rate.
        action_chunk_stride: Override action chunk stride.
        max_env_steps: Override max environment steps.
        run_name: Experiment run name (used for checkpoint/log dirs).
        checkpoint_base: Base directory for checkpoints.
    """
    config = make_pen_uncap_config()

    # Apply overrides by rebuilding frozen dataclasses.
    rl_token_overrides = {}
    if rl_token_lr is not None:
        rl_token_overrides["learning_rate"] = rl_token_lr
    if rl_token_batch_size is not None:
        rl_token_overrides["batch_size"] = rl_token_batch_size
    if rl_token_num_steps is not None:
        rl_token_overrides["num_train_steps"] = rl_token_num_steps
    if rl_token_overrides:
        config = dataclasses.replace(config, rl_token=dataclasses.replace(config.rl_token, **rl_token_overrides))

    actor_overrides = {}
    if actor_lr is not None:
        actor_overrides["learning_rate"] = actor_lr
    if actor_overrides:
        config = dataclasses.replace(config, actor=dataclasses.replace(config.actor, **actor_overrides))

    critic_overrides = {}
    if critic_lr is not None:
        critic_overrides["learning_rate"] = critic_lr
    if critic_overrides:
        config = dataclasses.replace(config, critic=dataclasses.replace(config.critic, **critic_overrides))

    training_overrides = {}
    if beta is not None:
        training_overrides["beta"] = beta
    if reference_action_dropout is not None:
        training_overrides["reference_action_dropout"] = reference_action_dropout
    if utd_ratio is not None:
        training_overrides["utd_ratio"] = utd_ratio
    if num_warmup_episodes is not None:
        training_overrides["num_warmup_episodes"] = num_warmup_episodes
    if action_chunk_stride is not None:
        training_overrides["action_chunk_stride"] = action_chunk_stride
    if max_env_steps is not None:
        training_overrides["max_env_steps"] = max_env_steps
    if training_overrides:
        config = dataclasses.replace(config, training=dataclasses.replace(config.training, **training_overrides))

    # Resolve checkpoint dirs.
    suffix = f"/{run_name}" if run_name else ""
    rl_token_ckpt_dir = f"{checkpoint_base}/rl_token{suffix}"
    online_rl_ckpt_dir = f"{checkpoint_base}/online_rl{suffix}"

    logger.info("Config: %s", config)
    logger.info("Checkpoint dirs: rl_token=%s, online_rl=%s", rl_token_ckpt_dir, online_rl_ckpt_dir)
    rng = jax.random.PRNGKey(seed)

    if phase in ("1", "both"):
        rng, phase1_rng = jax.random.split(rng)
        rl_token_module = train_rl_token_pen_uncap(
            config, phase1_rng,
            embeddings_dir=embeddings_dir,
            checkpoint_dir=rl_token_ckpt_dir,
        )

    if phase in ("2", "both"):
        rng, phase2_rng = jax.random.split(rng)

        if phase == "2":
            # Load RL token from checkpoint.
            if rl_token_checkpoint is None:
                raise ValueError("--rl-token-checkpoint required for phase 2")
            rng, model_rng = jax.random.split(rng)
            rl_token_module = RLTokenModule(config.rl_token, rngs=nnx.Rngs(model_rng))
            if rl_token_checkpoint.endswith(".pkl"):
                import pickle
                with open(rl_token_checkpoint, "rb") as f:
                    state_np = pickle.load(f)
                state = jax.tree.map(jnp.asarray, state_np)
            else:
                checkpointer = ocp.StandardCheckpointer()
                state = checkpointer.restore(rl_token_checkpoint, target=nnx.state(rl_token_module))
            nnx.update(rl_token_module, state)

        train_online_rl_pen_uncap(config, rl_token_module, phase2_rng, online_rl_ckpt_dir)


if __name__ == "__main__":
    tyro.cli(main)
