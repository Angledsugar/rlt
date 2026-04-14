"""RL Token encoder/decoder configuration."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class RLTokenConfig:
    """Configuration for the RL token encoder-decoder transformer.

    The encoder compresses VLA embeddings into a compact 1 x token_dim vector.
    The decoder autoregressively reconstructs the original embeddings.
    (Paper Section IV-A, Figure 2)
    """

    # RL token embedding dimension (paper: 2048).
    token_dim: int = 2048

    # Encoder transformer config.
    encoder_num_layers: int = 2
    encoder_num_heads: int = 8
    encoder_mlp_dim: int = 2048

    # Decoder transformer config.
    decoder_num_layers: int = 4
    decoder_num_heads: int = 8
    decoder_mlp_dim: int = 2048

    # VLA embedding dimension (PaliGemma Gemma 2B backbone width).
    # embed_prefix() returns tokens from Gemma 2B (width=2048).
    vla_embed_dim: int = 2048

    # Learning rate for RL token training (paper: combined with VLA SFT).
    learning_rate: float = 1e-4

    # Optional VLA SFT coefficient alpha (Eq.2 combined loss).
    # Set > 0 to also fine-tune VLA during RL token training.
    vla_sft_alpha: float = 0.0

    # Number of gradient steps for RL token training.
    num_train_steps: int = 10000

    # Batch size for RL token training.
    batch_size: int = 32
