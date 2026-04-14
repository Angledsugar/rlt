"""Tests for Actor and Critic networks."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from rlt.config.actor_critic import ActorConfig
from rlt.config.actor_critic import CriticConfig
from rlt.models.actor import Actor
from rlt.models.critic import Critic
from rlt.models.critic import DoubleCritic

RL_TOKEN_DIM = 64
PROPRIO_DIM = 14
ACTION_CHUNK_DIM = 140  # 14 * 10


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def actor_config():
    return ActorConfig(hidden_dims=(32, 32))


@pytest.fixture
def critic_config():
    return CriticConfig(hidden_dims=(32, 32))


class TestActor:
    def test_output_shape(self, actor_config, rng):
        actor = Actor(actor_config, RL_TOKEN_DIM, PROPRIO_DIM, ACTION_CHUNK_DIM, rngs=nnx.Rngs(rng))
        rl_token = jax.random.normal(rng, (4, RL_TOKEN_DIM))
        proprio = jax.random.normal(rng, (4, PROPRIO_DIM))
        ref_action = jax.random.normal(rng, (4, ACTION_CHUNK_DIM))
        mean, std = actor(rl_token, proprio, ref_action)
        assert mean.shape == (4, ACTION_CHUNK_DIM)
        assert std.shape == (4, ACTION_CHUNK_DIM)
        assert jnp.all(std > 0)

    def test_sample(self, actor_config, rng):
        actor = Actor(actor_config, RL_TOKEN_DIM, PROPRIO_DIM, ACTION_CHUNK_DIM, rngs=nnx.Rngs(rng))
        rl_token = jax.random.normal(rng, (4, RL_TOKEN_DIM))
        proprio = jax.random.normal(rng, (4, PROPRIO_DIM))
        ref_action = jax.random.normal(rng, (4, ACTION_CHUNK_DIM))
        action = actor.sample(rng, rl_token, proprio, ref_action)
        assert action.shape == (4, ACTION_CHUNK_DIM)

    def test_get_mean(self, actor_config, rng):
        actor = Actor(actor_config, RL_TOKEN_DIM, PROPRIO_DIM, ACTION_CHUNK_DIM, rngs=nnx.Rngs(rng))
        rl_token = jax.random.normal(rng, (4, RL_TOKEN_DIM))
        proprio = jax.random.normal(rng, (4, PROPRIO_DIM))
        ref_action = jax.random.normal(rng, (4, ACTION_CHUNK_DIM))
        mean1 = actor.get_mean(rl_token, proprio, ref_action)
        mean2 = actor.get_mean(rl_token, proprio, ref_action)
        assert jnp.allclose(mean1, mean2)


class TestCritic:
    def test_output_shape(self, critic_config, rng):
        critic = Critic(critic_config, RL_TOKEN_DIM, PROPRIO_DIM, ACTION_CHUNK_DIM, rngs=nnx.Rngs(rng))
        rl_token = jax.random.normal(rng, (4, RL_TOKEN_DIM))
        proprio = jax.random.normal(rng, (4, PROPRIO_DIM))
        action = jax.random.normal(rng, (4, ACTION_CHUNK_DIM))
        ref_action = jax.random.normal(rng, (4, ACTION_CHUNK_DIM))
        q = critic(rl_token, proprio, action, ref_action)
        assert q.shape == (4, 1)


class TestDoubleCritic:
    def test_twin_q(self, critic_config, rng):
        double_critic = DoubleCritic(critic_config, RL_TOKEN_DIM, PROPRIO_DIM, ACTION_CHUNK_DIM, rngs=nnx.Rngs(rng))
        rl_token = jax.random.normal(rng, (4, RL_TOKEN_DIM))
        proprio = jax.random.normal(rng, (4, PROPRIO_DIM))
        action = jax.random.normal(rng, (4, ACTION_CHUNK_DIM))
        ref_action = jax.random.normal(rng, (4, ACTION_CHUNK_DIM))
        q1, q2 = double_critic(rl_token, proprio, action, ref_action)
        assert q1.shape == (4, 1)
        assert q2.shape == (4, 1)
        # Twin Q-networks should give different outputs.
        assert not jnp.allclose(q1, q2)

    def test_min_q(self, critic_config, rng):
        double_critic = DoubleCritic(critic_config, RL_TOKEN_DIM, PROPRIO_DIM, ACTION_CHUNK_DIM, rngs=nnx.Rngs(rng))
        rl_token = jax.random.normal(rng, (4, RL_TOKEN_DIM))
        proprio = jax.random.normal(rng, (4, PROPRIO_DIM))
        action = jax.random.normal(rng, (4, ACTION_CHUNK_DIM))
        ref_action = jax.random.normal(rng, (4, ACTION_CHUNK_DIM))
        q1, q2 = double_critic(rl_token, proprio, action, ref_action)
        min_q = double_critic.min_q(rl_token, proprio, action, ref_action)
        expected = jnp.minimum(q1, q2)
        assert jnp.allclose(min_q, expected)
