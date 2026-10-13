"""Phase 2: Online RL training loop.

Implements Algorithm 1 from the paper.
After the initial RL token training, the VLA and RL token module are frozen.
A lightweight actor and critic are trained online using TD3.

The training loop:
1. Warmup: Pre-fill replay buffer with VLA rollouts.
2. Online RL: Alternate between environment rollouts and off-policy updates.
   - At each step, the VLA produces reference action chunks.
   - The RL token module extracts compressed state representations.
   - The actor refines actions conditioned on the reference.
   - The critic evaluates state-action values.
   - Human interventions can optionally override actions.

Key components:
- TD backup for critic (Eq.3)
- Gaussian actor with BC regularization (Eq.5)
- Reference action dropout (Section IV-B)
- Action chunk subsampling (stride=2)
"""

import logging

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from rlt.config.base import RLTConfig
from rlt.models.actor import Actor
from rlt.models.critic import DoubleCritic
from rlt.models.rl_token import RLTokenModule
from rlt.rl.replay_buffer import ReplayBuffer
from rlt.rl.td3 import TD3Updater
from rlt.training.utils import create_checkpoint_manager
from rlt.training.utils import log_metrics
from rlt.vla.backbone import VLABackbone

logger = logging.getLogger("rlt")


def create_actor_critic(
    config: RLTConfig,
    rng: jax.Array,
) -> tuple[Actor, DoubleCritic, DoubleCritic]:
    """Create actor and twin critic networks.

    Args:
        config: RLT configuration.
        rng: PRNG key.

    Returns:
        (actor, critic, target_critic)
    """
    actor_rng, critic_rng, target_rng = jax.random.split(rng, 3)

    rl_token_dim = config.rl_token.token_dim
    proprio_dim = config.proprio_dim
    action_chunk_dim = config.chunked_action_dim

    actor = Actor(
        config.actor,
        rl_token_dim=rl_token_dim,
        proprio_dim=proprio_dim,
        action_chunk_dim=action_chunk_dim,
        rngs=nnx.Rngs(actor_rng),
    )

    critic = DoubleCritic(
        config.critic,
        rl_token_dim=rl_token_dim,
        proprio_dim=proprio_dim,
        action_chunk_dim=action_chunk_dim,
        rngs=nnx.Rngs(critic_rng),
    )

    # Target critic (initialized with same weights as critic).
    target_critic = DoubleCritic(
        config.critic,
        rl_token_dim=rl_token_dim,
        proprio_dim=proprio_dim,
        action_chunk_dim=action_chunk_dim,
        rngs=nnx.Rngs(target_rng),
    )
    # Copy critic params to target.
    nnx.update(target_critic, nnx.state(critic))

    return actor, critic, target_critic


def train_online_rl(
    config: RLTConfig,
    rl_token_module: RLTokenModule,
    env,
    rng: jax.Array,
    checkpoint_dir: str = "checkpoints/online_rl",
) -> Actor:
    """Run the online RL training loop (Algorithm 1).

    Args:
        config: RLT configuration.
        rl_token_module: Trained (frozen) RL token module.
        env: Environment instance (must implement BaseEnv interface).
        rng: PRNG key.

    Returns:
        Trained actor.
    """
    rng, vla_rng, ac_rng = jax.random.split(rng, 3)

    logger.info("Phase 2: Online RL training")
    logger.info("Training config: %s", config.training)

    # Load frozen VLA with openpi policy transforms so prompt/state/action
    # preprocessing matches the checkpoint's training/inference pipeline.
    logger.info("Loading VLA backbone...")
    vla = VLABackbone(config.vla_checkpoint)
    vla.load_from_openpi_config(
        config_name="pi05_aloha_pen_uncap",
        checkpoint_dir=config.vla_checkpoint,
    )
    logger.info("VLA backbone loaded.")

    # Create actor-critic.
    actor, critic, target_critic = create_actor_critic(config, ac_rng)

    # Create TD3 updater.
    updater = TD3Updater(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        actor_config=config.actor,
        critic_config=config.critic,
        training_config=config.training,
    )

    # Create replay buffer.
    replay_buffer = ReplayBuffer(
        capacity=config.training.replay_buffer_size,
        rl_token_dim=config.rl_token.token_dim,
        proprio_dim=config.proprio_dim,
        action_chunk_dim=config.chunked_action_dim,
    )

    # Checkpoint manager.
    ckpt_manager = create_checkpoint_manager(checkpoint_dir)

    # --- Warmup phase (Algorithm 1, line 3) ---
    logger.info("Warmup: collecting %d episodes with base VLA policy", config.training.num_warmup_episodes)
    rng = _run_warmup(
        config, vla, rl_token_module, env, replay_buffer, rng,
    )
    logger.info("Warmup complete. Buffer size: %d", len(replay_buffer))

    # --- Online RL loop (Algorithm 1, lines 5-19) ---
    logger.info("Starting online RL loop")
    env_step = 0
    episode = 0
    np_rng = np.random.default_rng(config.training.seed)

    while env_step < config.training.max_env_steps:
        rng, episode_rng = jax.random.split(rng)
        episode += 1

        # Collect one episode.
        episode_data = _collect_episode(
            config, vla, rl_token_module, actor, env, episode_rng,
            use_rl_policy=True,
        )

        # Add to replay buffer with subsampling.
        replay_buffer.add_episode(
            rl_tokens=episode_data["rl_tokens"],
            proprios=episode_data["proprios"],
            actions=episode_data["actions"],
            reference_actions=episode_data["reference_actions"],
            rewards=episode_data["rewards"],
            dones=episode_data["dones"],
            stride=config.training.action_chunk_stride,
        )

        env_step += len(episode_data["rewards"])
        ep_reward = float(episode_data["rewards"].sum())
        logger.info("Episode %d: steps=%d, reward=%.2f, total_env_steps=%d, buffer=%d",
                    episode, len(episode_data["rewards"]), ep_reward, env_step, len(replay_buffer))

        # Off-policy updates (paper: UTD ratio = 5).
        if len(replay_buffer) >= config.training.batch_size:
            for _ in range(config.training.utd_ratio):
                rng, update_rng = jax.random.split(rng)
                batch = replay_buffer.sample(config.training.batch_size, np_rng)
                # Convert to JAX arrays.
                jax_batch = {k: jnp.array(v) for k, v in batch.items()}
                metrics = updater.update(jax_batch, update_rng)

            log_metrics(env_step, metrics)

        # Save checkpoint.
        if env_step % config.training.checkpoint_interval == 0:
            ckpt_manager.save(env_step, args=ocp.args.StandardSave(nnx.state(actor)))
            logger.info("Checkpoint saved at step %d", env_step)

    # Save final checkpoint.
    ckpt_manager.save(env_step, args=ocp.args.StandardSave(nnx.state(actor)))
    logger.info("Online RL training complete after %d steps, %d episodes", env_step, episode)

    return actor


def _run_warmup(
    config: RLTConfig,
    vla: VLABackbone,
    rl_token_module: RLTokenModule,
    env,
    replay_buffer: ReplayBuffer,
    rng: jax.Array,
) -> jax.Array:
    """Pre-fill replay buffer using the base VLA policy (warmup phase).

    Algorithm 1, line 3: Roll out VLA reference policy for N_warm episodes.
    """
    for ep in range(config.training.num_warmup_episodes):
        rng, ep_rng = jax.random.split(rng)
        episode_data = _collect_episode(
            config, vla, rl_token_module, None, env, ep_rng,
            use_rl_policy=False,
        )
        replay_buffer.add_episode(
            rl_tokens=episode_data["rl_tokens"],
            proprios=episode_data["proprios"],
            actions=episode_data["actions"],
            reference_actions=episode_data["reference_actions"],
            rewards=episode_data["rewards"],
            dones=episode_data["dones"],
            stride=config.training.action_chunk_stride,
        )
        logger.info("Warmup episode %d/%d, steps=%d, buffer size: %d",
                    ep + 1, config.training.num_warmup_episodes,
                    len(episode_data["rewards"]), len(replay_buffer))
    return rng


def _collect_episode(
    config: RLTConfig,
    vla: VLABackbone,
    rl_token_module: RLTokenModule,
    actor: Actor | None,
    env,
    rng: jax.Array,
    use_rl_policy: bool = False,
) -> dict[str, np.ndarray]:
    """Collect a single episode.

    Algorithm 1, lines 6-12:
    - Sample VLA reference action chunk.
    - Extract RL token.
    - Execute action (VLA reference or RL policy).
    - Store transition.

    Args:
        config: RLT config.
        vla: Frozen VLA backbone.
        rl_token_module: Frozen RL token module.
        actor: RL actor (None during warmup).
        env: Environment.
        rng: PRNG key.
        use_rl_policy: Whether to use RL policy (False during warmup).

    Returns:
        Dict of episode arrays.
    """
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

        # Get raw policy-format observation for openpi transforms.
        if hasattr(env, "get_policy_observation"):
            obs_dict = env.get_policy_observation(obs)
        else:
            obs_dict = env.get_observation_dict(obs)

        # Extract VLA embeddings and RL token (Algorithm 1, line 8).
        vla_embeddings = vla.extract_policy_embeddings(obs_dict)
        rl_token = rl_token_module.encode(vla_embeddings)

        # Sample VLA reference action chunk (Algorithm 1, line 7).
        reference_chunk = vla.sample_policy_actions(vla_rng, obs_dict)
        if reference_chunk.shape[1] < config.action_chunk_length:
            raise ValueError(
                f"VLA returned horizon {reference_chunk.shape[1]}, "
                f"but config.action_chunk_length={config.action_chunk_length}"
            )
        if reference_chunk.shape[2] != config.action_dim:
            raise ValueError(
                f"VLA returned action_dim {reference_chunk.shape[2]}, "
                f"but config.action_dim={config.action_dim}"
            )
        reference_chunk = reference_chunk[:, : config.action_chunk_length, :]
        reference_flat = reference_chunk.reshape(1, -1)

        # Get proprioception.
        proprio = env.get_proprio(obs)

        # Decide action (Algorithm 1, lines 9).
        if use_rl_policy and actor is not None:
            # RL policy: refine action conditioned on VLA reference.
            action_flat = actor.sample(
                actor_rng,
                rl_token,
                proprio[None],
                reference_flat,
            )
        else:
            # Warmup: use VLA reference directly.
            action_flat = reference_flat

        # Execute action chunk (Algorithm 1, line 10).
        action_chunk = action_flat.reshape(config.action_chunk_length, config.action_dim)
        # Clip actions to valid range to avoid MuJoCo physics instability.
        action_chunk = np.clip(np.asarray(action_chunk), -1.0, 1.0)
        next_obs, reward, done, info = env.step(action_chunk)

        # Store transition data.
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
