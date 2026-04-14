"""Tests for RL Token encoder-decoder."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from rlt.config.rl_token import RLTokenConfig
from rlt.models.rl_token import RLTokenDecoder
from rlt.models.rl_token import RLTokenEncoder
from rlt.models.rl_token import RLTokenModule


@pytest.fixture
def config():
    return RLTokenConfig(
        token_dim=64,
        encoder_num_layers=1,
        encoder_num_heads=4,
        encoder_mlp_dim=128,
        decoder_num_layers=1,
        decoder_num_heads=4,
        decoder_mlp_dim=128,
        vla_embed_dim=32,
    )


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


class TestRLTokenEncoder:
    def test_output_shape(self, config, rng):
        encoder = RLTokenEncoder(config, rngs=nnx.Rngs(rng))
        vla_embeddings = jax.random.normal(rng, (2, 10, config.vla_embed_dim))
        rl_token = encoder(vla_embeddings)
        assert rl_token.shape == (2, config.token_dim)

    def test_deterministic(self, config, rng):
        encoder = RLTokenEncoder(config, rngs=nnx.Rngs(rng))
        vla_embeddings = jax.random.normal(rng, (2, 10, config.vla_embed_dim))
        out1 = encoder(vla_embeddings)
        out2 = encoder(vla_embeddings)
        assert jnp.allclose(out1, out2)


class TestRLTokenDecoder:
    def test_output_shape(self, config, rng):
        decoder = RLTokenDecoder(config, rngs=nnx.Rngs(rng))
        rl_token = jax.random.normal(rng, (2, config.token_dim))
        target_embeddings = jax.random.normal(rng, (2, 10, config.token_dim))
        reconstructed = decoder(rl_token, target_embeddings)
        assert reconstructed.shape == (2, 10, config.vla_embed_dim)


class TestRLTokenModule:
    def test_forward(self, config, rng):
        module = RLTokenModule(config, rngs=nnx.Rngs(rng))
        vla_embeddings = jax.random.normal(rng, (2, 10, config.vla_embed_dim))
        reconstructed, rl_token = module(vla_embeddings)
        assert rl_token.shape == (2, config.token_dim)
        assert reconstructed.shape == (2, 10, config.vla_embed_dim)

    def test_encode(self, config, rng):
        module = RLTokenModule(config, rngs=nnx.Rngs(rng))
        vla_embeddings = jax.random.normal(rng, (2, 10, config.vla_embed_dim))
        rl_token = module.encode(vla_embeddings)
        assert rl_token.shape == (2, config.token_dim)

    def test_reconstruction_loss(self, config, rng):
        module = RLTokenModule(config, rngs=nnx.Rngs(rng))
        vla_embeddings = jax.random.normal(rng, (2, 10, config.vla_embed_dim))
        loss = module.reconstruction_loss(vla_embeddings)
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert loss >= 0
