"""Evaluation script for trained RLT policy.

Usage:
    uv run python scripts/evaluate.py --actor-checkpoint checkpoints/online_rl/final
"""

import logging

import flax.nnx as nnx
import jax
import numpy as np
import orbax.checkpoint as ocp
import tyro

from rlt.config.base import RLTConfig
from rlt.models.actor import Actor
from rlt.models.rl_token import RLTokenModule
from rlt.vla.backbone import VLABackbone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("rlt")


def main(
    actor_checkpoint: str,
    rl_token_checkpoint: str,
    checkpoint_path: str = "gs://openpi-assets/checkpoints/pi05_base/params",
    num_episodes: int = 10,
    seed: int = 42,
) -> None:
    """Evaluate a trained RLT policy.

    Args:
        actor_checkpoint: Path to trained actor checkpoint.
        rl_token_checkpoint: Path to trained RL token checkpoint.
        checkpoint_path: Path to pi0.5-base checkpoint.
        num_episodes: Number of evaluation episodes.
        seed: Random seed.
    """
    config = RLTConfig(vla_checkpoint=checkpoint_path)
    rng = jax.random.PRNGKey(seed)

    # Load models.
    rng, vla_rng, token_rng, actor_rng = jax.random.split(rng, 4)

    vla = VLABackbone(checkpoint_path)
    vla.load_from_openpi_config(
        config_name="pi05_aloha_pen_uncap",
        checkpoint_dir=checkpoint_path,
    )

    rl_token_module = RLTokenModule(config.rl_token, rngs=nnx.Rngs(token_rng))
    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(rl_token_checkpoint, target=nnx.state(rl_token_module))
    nnx.update(rl_token_module, state)

    actor = Actor(
        config.actor,
        rl_token_dim=config.rl_token.token_dim,
        proprio_dim=config.proprio_dim,
        action_chunk_dim=config.chunked_action_dim,
        rngs=nnx.Rngs(actor_rng),
    )
    state = checkpointer.restore(actor_checkpoint, target=nnx.state(actor))
    nnx.update(actor, state)

    # Create environment.
    from rlt.envs.sim_env import DummySimEnv
    env = DummySimEnv(action_dim=config.action_dim, proprio_dim=config.proprio_dim)

    # Evaluate.
    successes = 0
    total_steps = 0

    for ep in range(num_episodes):
        rng, ep_rng = jax.random.split(rng)
        obs = env.reset()
        done = False
        ep_steps = 0

        while not done:
            rng, vla_rng, actor_rng = jax.random.split(rng, 3)
            if hasattr(env, "get_policy_observation"):
                obs_dict = env.get_policy_observation(obs)
            else:
                obs_dict = env.get_observation_dict(obs)

            vla_embeddings = vla.extract_policy_embeddings(obs_dict)
            rl_token = rl_token_module.encode(vla_embeddings)
            reference_chunk = vla.sample_policy_actions(vla_rng, obs_dict)
            reference_chunk = reference_chunk[:, : config.action_chunk_length, :]
            reference_flat = reference_chunk.reshape(1, -1)
            proprio = env.get_proprio(obs)

            action_flat = actor.get_mean(rl_token, proprio[None], reference_flat)
            action_chunk = np.array(action_flat).reshape(config.action_chunk_length, config.action_dim)

            obs, reward, done, info = env.step(action_chunk)
            ep_steps += config.action_chunk_length

        if reward > 0:
            successes += 1
        total_steps += ep_steps
        logger.info("Episode %d: steps=%d, success=%s", ep + 1, ep_steps, reward > 0)

    logger.info("Evaluation: %d/%d successes (%.1f%%)", successes, num_episodes, 100 * successes / num_episodes)
    logger.info("Average steps per episode: %.1f", total_steps / num_episodes)


if __name__ == "__main__":
    tyro.cli(main)
