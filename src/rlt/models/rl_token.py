"""RL Token encoder-decoder transformer.

Implements the RL token extraction mechanism from Section IV-A and Figure 2.
The encoder compresses VLA internal embeddings into a compact 1 x token_dim
representation. The decoder autoregressively reconstructs the original embeddings
from the RL token.

Architecture:
    encoder g_θ: [z_1:M, e_cl] -> z_rl (Eq.1)
    decoder h_θ: z_rl -> reconstructed z_1:M (Eq.2)

where z_1:M are the final-layer token embeddings from the VLA,
e_cl is a learned CLS-like token appended to the sequence,
and z_rl is the encoder output at the special token position.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from rlt.config.rl_token import RLTokenConfig


class TransformerBlock(nnx.Module):
    """A single transformer block with pre-norm (RMSNorm) attention and MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, *, rngs: nnx.Rngs):
        self.attn_norm = nnx.RMSNorm(embed_dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            decode=False,
            rngs=rngs,
        )
        self.mlp_norm = nnx.RMSNorm(embed_dim, rngs=rngs)
        self.mlp_up = nnx.Linear(embed_dim, mlp_dim, rngs=rngs)
        self.mlp_down = nnx.Linear(mlp_dim, embed_dim, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        # Pre-norm self-attention.
        h = self.attn_norm(x)
        h = self.attn(h, mask=mask)
        x = x + h

        # Pre-norm feed-forward.
        h = self.mlp_norm(x)
        h = self.mlp_up(h)
        h = nnx.gelu(h)
        h = self.mlp_down(h)
        x = x + h
        return x


class CausalTransformerBlock(nnx.Module):
    """Transformer block with causal (autoregressive) masking."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, *, rngs: nnx.Rngs):
        self.attn_norm = nnx.RMSNorm(embed_dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            decode=False,
            rngs=rngs,
        )
        self.mlp_norm = nnx.RMSNorm(embed_dim, rngs=rngs)
        self.mlp_up = nnx.Linear(embed_dim, mlp_dim, rngs=rngs)
        self.mlp_down = nnx.Linear(mlp_dim, embed_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        seq_len = x.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

        # Pre-norm causal self-attention.
        h = self.attn_norm(x)
        h = self.attn(h, mask=causal_mask)
        x = x + h

        # Pre-norm feed-forward.
        h = self.mlp_norm(x)
        h = self.mlp_up(h)
        h = nnx.gelu(h)
        h = self.mlp_down(h)
        x = x + h
        return x


class RLTokenEncoder(nnx.Module):
    """Encoder g_θ that compresses VLA embeddings into a single RL token.

    Takes the sequence of VLA token embeddings z_1:M, appends a learnable
    CLS token e_cl, processes through transformer layers, and outputs
    the RL token z_rl at the CLS position (Eq.1).
    """

    def __init__(self, config: RLTokenConfig, *, rngs: nnx.Rngs):
        self.config = config
        # Learnable CLS token: e_cl (appended to the VLA embedding sequence).
        self.cls_token = nnx.Param(jax.random.normal(rngs.params(), (1, 1, config.token_dim)))

        # Project VLA embeddings to encoder dimension if needed.
        self.input_proj = nnx.Linear(config.vla_embed_dim, config.token_dim, rngs=rngs)

        # Encoder transformer layers.
        self.layers = [
            TransformerBlock(
                embed_dim=config.token_dim,
                num_heads=config.encoder_num_heads,
                mlp_dim=config.encoder_mlp_dim,
                rngs=rngs,
            )
            for _ in range(config.encoder_num_layers)
        ]

        self.output_norm = nnx.RMSNorm(config.token_dim, rngs=rngs)

    def __call__(self, vla_embeddings: jax.Array) -> jax.Array:
        """Encode VLA embeddings into RL token.

        Args:
            vla_embeddings: [batch, seq_len, vla_embed_dim] - VLA final-layer embeddings.

        Returns:
            rl_token: [batch, token_dim] - Compressed RL token representation.
        """
        batch_size = vla_embeddings.shape[0]

        # Project VLA embeddings to encoder dim.
        x = self.input_proj(vla_embeddings)

        # Append learnable CLS token.
        cls = jnp.broadcast_to(self.cls_token.value, (batch_size, 1, self.config.token_dim))
        x = jnp.concatenate([x, cls], axis=1)

        # Process through transformer layers.
        for layer in self.layers:
            x = layer(x)

        x = self.output_norm(x)

        # Extract RL token at the last position (CLS token position).
        rl_token = x[:, -1, :]  # [batch, token_dim]
        return rl_token


class RLTokenDecoder(nnx.Module):
    """Decoder h_θ that autoregressively reconstructs VLA embeddings from RL token.

    Takes the RL token z_rl and reconstructs the original VLA embeddings
    z_1:M using an autoregressive transformer with a linear output projection (Eq.2).
    Uses stop-gradient on targets: z_hat_i = sg(z_i).
    """

    def __init__(self, config: RLTokenConfig, *, rngs: nnx.Rngs):
        self.config = config

        # Decoder transformer layers (causal / autoregressive).
        self.layers = [
            CausalTransformerBlock(
                embed_dim=config.token_dim,
                num_heads=config.decoder_num_heads,
                mlp_dim=config.decoder_mlp_dim,
                rngs=rngs,
            )
            for _ in range(config.decoder_num_layers)
        ]

        self.output_norm = nnx.RMSNorm(config.token_dim, rngs=rngs)

        # Linear output projection d_θ to reconstruct VLA embedding dim.
        self.output_proj = nnx.Linear(config.token_dim, config.vla_embed_dim, rngs=rngs)

    def __call__(self, rl_token: jax.Array, target_embeddings: jax.Array) -> jax.Array:
        """Autoregressively reconstruct VLA embeddings from RL token.

        During training, the decoder receives the RL token and the (stop-gradient)
        target embeddings shifted right by one position.

        Args:
            rl_token: [batch, token_dim] - The RL token from the encoder.
            target_embeddings: [batch, seq_len, token_dim] - Stop-gradient VLA embeddings
                (already projected to token_dim by caller).

        Returns:
            reconstructed: [batch, seq_len, vla_embed_dim] - Reconstructed embeddings.
        """
        # Prepend RL token as the first element of the decoder input sequence.
        # Shift targets right: input is [z_rl, z_1, z_2, ..., z_{M-1}]
        # to predict [z_1, z_2, ..., z_M].
        rl_token_expanded = rl_token[:, None, :]  # [batch, 1, token_dim]
        # Use stop-gradient targets shifted right.
        decoder_input = jnp.concatenate(
            [rl_token_expanded, target_embeddings[:, :-1, :]], axis=1
        )

        x = decoder_input
        for layer in self.layers:
            x = layer(x)

        x = self.output_norm(x)
        reconstructed = self.output_proj(x)
        return reconstructed


class RLTokenModule(nnx.Module):
    """Combined RL token encoder-decoder module.

    Wraps the encoder and decoder together for joint training.
    Implements the reconstruction loss L_co (Eq.2).
    """

    def __init__(self, config: RLTokenConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.encoder = RLTokenEncoder(config, rngs=rngs)
        self.decoder = RLTokenDecoder(config, rngs=rngs)

        # Projection for target embeddings to decoder dim (for shifted input).
        self.target_proj = nnx.Linear(config.vla_embed_dim, config.token_dim, rngs=rngs)

    def encode(self, vla_embeddings: jax.Array) -> jax.Array:
        """Extract RL token from VLA embeddings.

        Args:
            vla_embeddings: [batch, seq_len, vla_embed_dim]

        Returns:
            rl_token: [batch, token_dim]
        """
        return self.encoder(vla_embeddings)

    def __call__(self, vla_embeddings: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass: encode then decode, return (reconstructed, rl_token).

        Args:
            vla_embeddings: [batch, seq_len, vla_embed_dim]

        Returns:
            reconstructed: [batch, seq_len, vla_embed_dim]
            rl_token: [batch, token_dim]
        """
        # Encode.
        rl_token = self.encoder(vla_embeddings)

        # Project target embeddings to decoder dim (with stop-gradient).
        target_proj = self.target_proj(jax.lax.stop_gradient(vla_embeddings))

        # Decode.
        reconstructed = self.decoder(rl_token, target_proj)

        return reconstructed, rl_token

    def reconstruction_loss(self, vla_embeddings: jax.Array) -> jax.Array:
        """Compute autoregressive reconstruction loss L_co (Eq.2).

        L_co = E_D [ sum_i || (d_θ(z_rl, z_hat_{1:i-1}))_i - z_i ||^2 ]

        Args:
            vla_embeddings: [batch, seq_len, vla_embed_dim]

        Returns:
            loss: scalar reconstruction loss.
        """
        reconstructed, _ = self(vla_embeddings)
        targets = jax.lax.stop_gradient(vla_embeddings)
        loss = jnp.mean(jnp.sum(jnp.square(reconstructed - targets), axis=-1))
        return loss
