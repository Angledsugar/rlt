"""Script to run online RL training (Phase 2).

Usage:
    uv run python scripts/train_online_rl.py --rl-token-checkpoint checkpoints/rl_token/10000
    uv run python scripts/train_online_rl.py --config configs/ethernet.yaml
"""

import logging

import jax
import tyro

from rlt.config.base import RLTConfig
from rlt.training.train_online_rl import train_online_rl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


def main(
    rl_token_checkpoint: str = "checkpoints/rl_token/10000",
    checkpoint_path: str = "gs://openpi-assets/checkpoints/pi05_base/params",
    max_env_steps: int = 100_000,
    beta: float = 1.0,
    num_warmup_episodes: int = 50,
    seed: int = 42,
) -> None:
    """Run online RL training with pre-trained RL token.

    Args:
        rl_token_checkpoint: Path to trained RL token checkpoint.
        checkpoint_path: Path to pi0.5-base VLA checkpoint.
        max_env_steps: Maximum environment steps.
        beta: BC regularization coefficient.
        num_warmup_episodes: Number of warmup episodes with VLA.
        seed: Random seed.
    """
    import flax.nnx as nnx
    import orbax.checkpoint as ocp

    from rlt.config.training import TrainingConfig
    from rlt.envs.sim_env import DummySimEnv
    from rlt.models.rl_token import RLTokenModule

    config = RLTConfig(
        vla_checkpoint=checkpoint_path,
        training=TrainingConfig(
            max_env_steps=max_env_steps,
            beta=beta,
            num_warmup_episodes=num_warmup_episodes,
            seed=seed,
        ),
    )

    rng = jax.random.PRNGKey(seed)
    rng, model_rng = jax.random.split(rng)

    # Load trained RL token module.
    rl_token_module = RLTokenModule(config.rl_token, rngs=nnx.Rngs(model_rng))
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(rl_token_checkpoint, target=nnx.state(rl_token_module))
    nnx.update(rl_token_module, state)

    # Create environment (replace with real env for actual training).
    env = DummySimEnv(
        action_dim=config.action_dim,
        proprio_dim=config.proprio_dim,
    )

    train_online_rl(config, rl_token_module, env, rng)


if __name__ == "__main__":
    tyro.cli(main)
