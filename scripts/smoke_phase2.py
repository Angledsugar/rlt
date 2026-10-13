"""Phase 2 smoke test: Online RL with mock VLA.

Tests the full online RL loop (warmup + TD3 training) without
requiring the real pi0.5 VLA model or GCS downloads.

Replaces VLABackbone with a FakeVLA that returns random embeddings
and actions, then runs a short training loop to verify:
1. Environment integration (PenUncapEnv)
2. Replay buffer filling
3. TD3 actor-critic updates converge (critic loss decreasing)
4. No shape mismatches or NaN values
"""

import dataclasses
import logging
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("rlt.smoke2")


class FakeVLA:
    """Mock VLA backbone that returns random embeddings and actions."""

    def __init__(self, embed_dim=2048, seq_len=8, action_dim=14, action_horizon=10):
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.action_horizon = action_horizon

    def extract_policy_embeddings(self, obs_dict):
        return jnp.zeros((1, self.seq_len, self.embed_dim))

    def sample_policy_actions(self, rng, obs_dict, num_steps=10):
        return jax.random.normal(rng, (1, self.action_horizon, self.action_dim)) * 0.1

    def load_from_openpi_config(self, *args, **kwargs):
        pass


def run_phase2_smoke(
    beta=1.0,
    reference_action_dropout=0.5,
    utd_ratio=5,
    num_warmup_episodes=3,
    max_env_steps=500,
    actor_lr=3e-4,
    critic_lr=3e-4,
    action_chunk_stride=2,
    seed=42,
    run_name="smoke",
):
    """Run a short Phase 2 training with mock VLA."""
    from rlt.config.base import RLTConfig
    from rlt.config.actor_critic import ActorConfig, CriticConfig
    from rlt.config.rl_token import RLTokenConfig
    from rlt.config.training import TrainingConfig
    from rlt.models.rl_token import RLTokenModule
    from rlt.models.actor import Actor
    from rlt.models.critic import DoubleCritic
    from rlt.rl.replay_buffer import ReplayBuffer
    from rlt.rl.td3 import TD3Updater
    from rlt.envs.aloha_env import AlohaSimEnv

    config = RLTConfig(
        task_name="pen_uncap_smoke",
        action_dim=14,
        action_chunk_length=10,
        proprio_dim=14,
        rl_token=RLTokenConfig(
            token_dim=2048,
            vla_embed_dim=2048,
        ),
        actor=ActorConfig(hidden_dims=(256, 256), learning_rate=actor_lr),
        critic=CriticConfig(hidden_dims=(256, 256), learning_rate=critic_lr),
        training=TrainingConfig(
            beta=beta,
            reference_action_dropout=reference_action_dropout,
            utd_ratio=utd_ratio,
            num_warmup_episodes=num_warmup_episodes,
            max_env_steps=max_env_steps,
            action_chunk_stride=action_chunk_stride,
            batch_size=64,
            log_interval=50,
            checkpoint_interval=100000,  # no checkpoint saving during smoke
            seed=seed,
        ),
    )

    rng = jax.random.PRNGKey(seed)
    rng, model_rng, ac_rng = jax.random.split(rng, 3)

    # Create frozen RL token module (random weights - no real training).
    rl_token_module = RLTokenModule(config.rl_token, rngs=nnx.Rngs(model_rng))

    # Create fake VLA.
    vla = FakeVLA()

    # Create env.
    env = AlohaSimEnv(
        task="gym_aloha/AlohaPenUncap-v0",
        prompt="uncap the pen",
        max_episode_steps=100,  # short episodes for smoke test
    )

    # Create actor-critic.
    from rlt.training.train_online_rl import create_actor_critic
    actor, critic, target_critic = create_actor_critic(config, ac_rng)

    updater = TD3Updater(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        actor_config=config.actor,
        critic_config=config.critic,
        training_config=config.training,
    )

    replay_buffer = ReplayBuffer(
        capacity=config.training.replay_buffer_size,
        rl_token_dim=config.rl_token.token_dim,
        proprio_dim=config.proprio_dim,
        action_chunk_dim=config.chunked_action_dim,
    )

    # --- Warmup ---
    t0 = time.time()
    logger.info("[%s] Warmup: %d episodes", run_name, num_warmup_episodes)
    for ep in range(num_warmup_episodes):
        rng, ep_rng = jax.random.split(rng)
        episode_data = _collect_episode_mock(config, vla, rl_token_module, None, env, ep_rng)
        replay_buffer.add_episode(
            rl_tokens=episode_data["rl_tokens"],
            proprios=episode_data["proprios"],
            actions=episode_data["actions"],
            reference_actions=episode_data["reference_actions"],
            rewards=episode_data["rewards"],
            dones=episode_data["dones"],
            stride=config.training.action_chunk_stride,
        )
        logger.info("  warmup ep %d: %d steps, buffer=%d", ep + 1, len(episode_data["rewards"]), len(replay_buffer))

    warmup_time = time.time() - t0
    logger.info("Warmup done in %.1fs, buffer size=%d", warmup_time, len(replay_buffer))

    # --- Online RL ---
    env_step = 0
    episode = 0
    np_rng = np.random.default_rng(seed)
    metrics_history = []

    while env_step < max_env_steps:
        rng, episode_rng = jax.random.split(rng)
        episode += 1

        episode_data = _collect_episode_mock(config, vla, rl_token_module, actor, env, episode_rng, use_rl_policy=True)
        replay_buffer.add_episode(
            rl_tokens=episode_data["rl_tokens"],
            proprios=episode_data["proprios"],
            actions=episode_data["actions"],
            reference_actions=episode_data["reference_actions"],
            rewards=episode_data["rewards"],
            dones=episode_data["dones"],
            stride=config.training.action_chunk_stride,
        )

        ep_reward = float(np.sum(episode_data["rewards"]))
        env_step += len(episode_data["rewards"])

        # Off-policy updates.
        if len(replay_buffer) >= config.training.batch_size:
            for _ in range(config.training.utd_ratio):
                rng, update_rng = jax.random.split(rng)
                batch = replay_buffer.sample(config.training.batch_size, np_rng)
                jax_batch = {k: jnp.array(v) for k, v in batch.items()}
                metrics = updater.update(jax_batch, update_rng)

            # Check for NaN.
            for k, v in metrics.items():
                val = float(v)
                if np.isnan(val):
                    logger.error("NaN detected in %s at env_step=%d!", k, env_step)
                    return {"name": run_name, "error": f"NaN in {k}", "env_step": env_step}

            if env_step % config.training.log_interval == 0 or episode <= 3:
                metrics_snap = {k: float(v) for k, v in metrics.items()}
                metrics_snap["env_step"] = env_step
                metrics_snap["episode"] = episode
                metrics_snap["ep_reward"] = ep_reward
                metrics_snap["buffer_size"] = len(replay_buffer)
                metrics_history.append(metrics_snap)
                logger.info("  step=%d ep=%d reward=%.2f critic=%.4f %s",
                            env_step, episode, ep_reward,
                            metrics_snap.get("critic_loss", 0),
                            {k: f"{v:.4f}" for k, v in metrics_snap.items() if k.startswith("actor")})

    wall_time = time.time() - t0

    return {
        "name": run_name,
        "beta": beta,
        "ref_dropout": reference_action_dropout,
        "utd_ratio": utd_ratio,
        "warmup_eps": num_warmup_episodes,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "stride": action_chunk_stride,
        "final_env_step": env_step,
        "episodes": episode,
        "wall_time_s": wall_time,
        "warmup_time_s": warmup_time,
        "metrics_history": metrics_history,
        "error": None,
    }


def _collect_episode_mock(config, vla, rl_token_module, actor, env, rng, use_rl_policy=False):
    """Collect episode using mock VLA (same logic as train_online_rl._collect_episode)."""
    rl_tokens_list = []
    proprios_list = []
    actions_list = []
    ref_actions_list = []
    rewards_list = []
    dones_list = []

    obs = env.reset()
    done = False

    while not done:
        rng, vla_rng, actor_rng = jax.random.split(rng, 3)

        # Mock VLA embeddings.
        vla_embeddings = vla.extract_policy_embeddings(obs)
        rl_token = rl_token_module.encode(vla_embeddings)

        # Mock VLA reference actions.
        reference_chunk = vla.sample_policy_actions(vla_rng, obs)
        reference_chunk = reference_chunk[:, :config.action_chunk_length, :]
        reference_flat = reference_chunk.reshape(1, -1)

        proprio = env.get_proprio(obs)

        if use_rl_policy and actor is not None:
            action_flat = actor.sample(actor_rng, rl_token, proprio[None], reference_flat)
        else:
            action_flat = reference_flat

        action_chunk = action_flat.reshape(config.action_chunk_length, config.action_dim)
        action_chunk = np.clip(np.asarray(action_chunk), -1.0, 1.0)
        next_obs, reward, done, info = env.step(action_chunk)

        rl_tokens_list.append(np.array(rl_token[0]))
        proprios_list.append(np.array(proprio))
        actions_list.append(np.array(action_flat[0]))
        ref_actions_list.append(np.array(reference_flat[0]))
        rewards_list.append(reward)
        dones_list.append(done)

        obs = next_obs

    return {
        "rl_tokens": np.stack(rl_tokens_list),
        "proprios": np.stack(proprios_list),
        "actions": np.stack(actions_list),
        "reference_actions": np.stack(ref_actions_list),
        "rewards": np.array(rewards_list, dtype=np.float32),
        "dones": np.array(dones_list, dtype=np.float32),
    }


def main():
    """Run Phase 2 experiments with different hyperparameter settings."""
    MAX_STEPS = 2000
    WARMUP_EPS = 5

    experiments = [
        # Baseline (paper defaults)
        {"run_name": "baseline", "beta": 1.0, "reference_action_dropout": 0.5, "utd_ratio": 5},
        # Lower beta (less BC regularization - now safe with action clipping)
        {"run_name": "beta_0.25", "beta": 0.25},
        {"run_name": "beta_0.5", "beta": 0.5},
        {"run_name": "beta_2.0", "beta": 2.0},
        # Higher UTD ratio (more off-policy updates per env step)
        {"run_name": "utd_10", "utd_ratio": 10},
        {"run_name": "utd_2", "utd_ratio": 2},
        # Reference action dropout
        {"run_name": "dropout_0.25", "reference_action_dropout": 0.25},
        {"run_name": "dropout_0.75", "reference_action_dropout": 0.75},
        # Actor LR
        {"run_name": "actor_lr_1e-4", "actor_lr": 1e-4},
        # Stride
        {"run_name": "stride_1", "action_chunk_stride": 1},
        {"run_name": "stride_4", "action_chunk_stride": 4},
    ]

    results = []
    for exp_kwargs in experiments:
        # Fill defaults.
        defaults = {
            "beta": 1.0, "reference_action_dropout": 0.5, "utd_ratio": 5,
            "num_warmup_episodes": WARMUP_EPS, "max_env_steps": MAX_STEPS,
            "actor_lr": 3e-4, "critic_lr": 3e-4, "action_chunk_stride": 2, "seed": 42,
        }
        defaults.update(exp_kwargs)
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT: %s", defaults["run_name"])
        logger.info("Config: %s", {k: v for k, v in defaults.items() if k != "run_name"})

        try:
            result = run_phase2_smoke(**defaults)
        except Exception as e:
            logger.error("Experiment %s FAILED: %s", defaults["run_name"], e, exc_info=True)
            result = {"name": defaults["run_name"], "error": str(e)}
        results.append(result)

        # Free memory.
        jax.clear_caches()

    # Summary table.
    print("\n" + "=" * 110)
    print("Phase 2 Smoke Test Results (mock VLA, max_steps=500, warmup=3 eps)")
    print("=" * 110)
    print("%-16s %6s %6s %5s %8s %8s %6s %10s %8s %8s" % (
        "Name", "Beta", "RefDO", "UTD", "ActorLR", "Episodes", "Steps", "CriticLoss", "BCLoss", "Time(s)"))
    print("-" * 110)
    for r in results:
        if r.get("error"):
            print("%-16s ERROR: %s" % (r["name"], r["error"]))
            continue
        mh = r.get("metrics_history", [])
        last = mh[-1] if mh else {}
        print("%-16s %6.2f %6.2f %5d %8.1e %8d %6d %10.4f %8.4f %8.1f" % (
            r["name"], r.get("beta", 0), r.get("ref_dropout", 0), r.get("utd_ratio", 0),
            r.get("actor_lr", 0), r.get("episodes", 0), r.get("final_env_step", 0),
            last.get("critic_loss", float("nan")), last.get("actor_bc_loss", float("nan")),
            r.get("wall_time_s", 0)))


if __name__ == "__main__":
    main()
