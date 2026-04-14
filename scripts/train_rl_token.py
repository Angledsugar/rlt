"""Script to train the RL token encoder-decoder (Phase 1).

Usage:
    uv run python scripts/train_rl_token.py --demo-data-path /path/to/demos
    uv run python scripts/train_rl_token.py --demo-data-path /path/to/demos --num-train-steps 5000
"""

import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import logging

import jax
import tyro

from rlt.config.base import RLTConfig
from rlt.training.train_rl_token import train_rl_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


def main(
    demo_data_path: str,
    checkpoint_path: str = "gs://openpi-assets/checkpoints/pi05_base/params",
    num_train_steps: int = 10000,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    seed: int = 42,
) -> None:
    """Train RL token encoder-decoder on demonstration data.

    Args:
        demo_data_path: Path to demonstration dataset.
        checkpoint_path: Path to pi0.5-base checkpoint.
        num_train_steps: Number of training steps.
        learning_rate: Learning rate for RL token training.
        batch_size: Batch size.
        seed: Random seed.
    """
    from rlt.config.rl_token import RLTokenConfig

    config = RLTConfig(
        vla_checkpoint=checkpoint_path,
        rl_token=RLTokenConfig(
            num_train_steps=num_train_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
        ),
    )

    rng = jax.random.PRNGKey(seed)
    train_rl_token(config, demo_data_path, rng)


if __name__ == "__main__":
    tyro.cli(main)
