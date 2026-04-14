"""Top-level RLT configuration."""

import dataclasses

from rlt.config.actor_critic import ActorConfig
from rlt.config.actor_critic import CriticConfig
from rlt.config.rl_token import RLTokenConfig
from rlt.config.training import TrainingConfig


@dataclasses.dataclass(frozen=True)
class RLTConfig:
    """Master config that composes all sub-configs."""

    # Task identifier.
    task_name: str = "default"

    # VLA model checkpoint path (gs:// or local).
    vla_checkpoint: str = "gs://openpi-assets/checkpoints/pi05_base/params"

    # Action space dimensions.
    action_dim: int = 14
    # Action chunk length C (paper default: 10).
    action_chunk_length: int = 10
    # Proprioceptive state dimension.
    proprio_dim: int = 14

    # Sub-configs.
    rl_token: RLTokenConfig = dataclasses.field(default_factory=RLTokenConfig)
    actor: ActorConfig = dataclasses.field(default_factory=ActorConfig)
    critic: CriticConfig = dataclasses.field(default_factory=CriticConfig)
    training: TrainingConfig = dataclasses.field(default_factory=TrainingConfig)

    @property
    def chunked_action_dim(self) -> int:
        """Flattened action chunk dimension: action_dim * action_chunk_length."""
        return self.action_dim * self.action_chunk_length
