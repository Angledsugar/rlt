"""Online RL training configuration."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the online RL training loop (Algorithm 1)."""

    # Replay buffer capacity.
    replay_buffer_size: int = 100_000

    # Number of warmup episodes using base VLA policy before RL starts.
    num_warmup_episodes: int = 50

    # Batch size for RL updates.
    batch_size: int = 256

    # BC regularization coefficient beta (Eq.5).
    # Controls how strongly the actor is regularized toward VLA reference actions.
    beta: float = 1.0

    # Reference action dropout probability (paper Section IV-B).
    # Fraction of transitions where reference action is replaced with zeros.
    reference_action_dropout: float = 0.5

    # Action chunk subsampling stride (paper Section V: stride=2).
    action_chunk_stride: int = 2

    # Control frequency in Hz (paper: 50 Hz).
    control_frequency_hz: int = 50

    # Update-to-data ratio (paper: 5).
    utd_ratio: int = 5

    # Maximum number of environment steps for online RL.
    max_env_steps: int = 100_000

    # Logging interval (in environment steps).
    log_interval: int = 100

    # Checkpoint save interval (in environment steps).
    checkpoint_interval: int = 5_000

    # WandB project name.
    wandb_project: str = "rlt"

    # Random seed.
    seed: int = 42
