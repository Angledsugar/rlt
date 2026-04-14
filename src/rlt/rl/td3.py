"""TD3-based critic and actor update for online RL.

Implements the training procedure from Section IV-B:
- Critic: TD backup with twin Q-networks (Eq.3)
- Actor: Q-value maximization + BC regularization (Eq.5)

    L_Q = E_{(x,a,r,x')} [ (Q - Q_target(x,a))^2 ]
    Q_target = sum_{t'=1}^{C} gamma^{t'-1} * r_{t'} + gamma^C * E_{a'~pi} [ Q_target(x', a') ]

    L_pi(theta) = E_{s~B} [ -Q_psi(x, a) + beta * ||a - a_hat||^2 ]
    where a ~ pi_theta(.|s, a_hat)
"""

import functools

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from rlt.config.actor_critic import ActorConfig
from rlt.config.actor_critic import CriticConfig
from rlt.config.training import TrainingConfig
from rlt.models.actor import Actor
from rlt.models.critic import DoubleCritic


class TD3Updater:
    """Manages TD3-style actor-critic updates."""

    def __init__(
        self,
        actor: Actor,
        critic: DoubleCritic,
        target_critic: DoubleCritic,
        actor_config: ActorConfig,
        critic_config: CriticConfig,
        training_config: TrainingConfig,
    ):
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.actor_config = actor_config
        self.critic_config = critic_config
        self.training_config = training_config

        # Optimizers.
        self.actor_optimizer = nnx.Optimizer(actor, optax.adam(actor_config.learning_rate))
        self.critic_optimizer = nnx.Optimizer(critic, optax.adam(critic_config.learning_rate))

        self._update_step = 0

    def update_critic(
        self,
        batch: dict[str, jax.Array],
        rng: jax.Array,
    ) -> jax.Array:
        """Update critic using TD3 backup (Eq.3).

        Args:
            batch: Dict with keys from replay buffer.
            rng: PRNG key.

        Returns:
            critic_loss: Scalar loss value.
        """
        gamma = self.critic_config.discount
        action_chunk_length = self.training_config.batch_size  # will use config

        # Compute target Q-value.
        next_action_mean, _ = self.actor(
            batch["next_rl_token"],
            batch["next_proprio"],
            batch["reference_action"],  # Use same reference for next state.
        )
        target_q = self.target_critic.min_q(
            batch["next_rl_token"],
            batch["next_proprio"],
            next_action_mean,
            batch["reference_action"],
        )
        # TD target: r + gamma^C * (1 - done) * Q_target(s', a').
        td_target = batch["reward"] + (gamma ** 10) * (1.0 - batch["done"]) * target_q
        td_target = jax.lax.stop_gradient(td_target)

        def critic_loss_fn(critic):
            q1, q2 = critic(
                batch["rl_token"],
                batch["proprio"],
                batch["action"],
                batch["reference_action"],
            )
            loss = jnp.mean(jnp.square(q1 - td_target)) + jnp.mean(jnp.square(q2 - td_target))
            return loss

        loss, grads = nnx.value_and_grad(critic_loss_fn)(self.critic)
        self.critic_optimizer.update(grads)

        return loss

    def update_actor(
        self,
        batch: dict[str, jax.Array],
        rng: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Update actor using Q-value maximization + BC regularization (Eq.5).

        L_pi = E[ -Q(x, a) + beta * ||a - a_hat||^2 ]

        Args:
            batch: Dict with keys from replay buffer.
            rng: PRNG key.

        Returns:
            (actor_loss, bc_loss): Loss components.
        """
        beta = self.training_config.beta
        reference_action = batch["reference_action"]

        # Apply reference action dropout (paper Section IV-B).
        dropout_rng, sample_rng = jax.random.split(rng)
        dropout_mask = jax.random.bernoulli(
            dropout_rng,
            1.0 - self.training_config.reference_action_dropout,
            shape=(reference_action.shape[0], 1),
        )
        masked_reference = reference_action * dropout_mask

        def actor_loss_fn(actor):
            action_mean, _ = actor(
                batch["rl_token"],
                batch["proprio"],
                masked_reference,
            )
            # Q-value term.
            q_value = self.critic.q1(
                batch["rl_token"],
                batch["proprio"],
                action_mean,
                masked_reference,
            )
            q_loss = -jnp.mean(q_value)

            # BC regularization term.
            bc_loss = jnp.mean(jnp.sum(jnp.square(action_mean - reference_action), axis=-1))

            total_loss = q_loss + beta * bc_loss
            return total_loss, (q_loss, bc_loss)

        (loss, (q_loss, bc_loss)), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(self.actor)
        self.actor_optimizer.update(grads)

        return q_loss, bc_loss

    def update_target_critic(self) -> None:
        """Soft update target critic parameters.

        theta_target = tau * theta + (1 - tau) * theta_target
        """
        tau = self.critic_config.tau
        critic_state = nnx.state(self.critic)
        target_state = nnx.state(self.target_critic)

        new_target_state = jax.tree.map(
            lambda c, t: tau * c + (1.0 - tau) * t,
            critic_state,
            target_state,
        )
        nnx.update(self.target_critic, new_target_state)

    def update(
        self,
        batch: dict[str, jax.Array],
        rng: jax.Array,
    ) -> dict[str, jax.Array]:
        """Perform one full update step (critic + possibly actor).

        Paper: 2 critic updates per actor update.

        Args:
            batch: Dict from replay buffer.
            rng: PRNG key.

        Returns:
            Dict of logged metrics.
        """
        rng, critic_rng, actor_rng = jax.random.split(rng, 3)

        # Critic update.
        critic_loss = self.update_critic(batch, critic_rng)

        metrics = {"critic_loss": critic_loss}

        # Actor update (every critic_updates_per_actor steps).
        self._update_step += 1
        if self._update_step % self.critic_config.critic_updates_per_actor == 0:
            q_loss, bc_loss = self.update_actor(batch, actor_rng)
            self.update_target_critic()
            metrics["actor_q_loss"] = q_loss
            metrics["actor_bc_loss"] = bc_loss

        return metrics
