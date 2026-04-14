"""Actor and Critic network configurations."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class ActorConfig:
    """Configuration for the RL actor MLP (pi_theta).

    The actor takes RL token + proprioception as input and outputs
    a Gaussian distribution over action chunks.
    (Paper Section IV-B, Eq.4)
    """

    # Hidden layer dimensions.
    # Paper Appendix B: 2-layer MLP (256) for easy tasks, 3-layer (512) for screw.
    hidden_dims: tuple[int, ...] = (256, 256)

    # Learning rate.
    learning_rate: float = 3e-4

    # Fixed standard deviation for Gaussian policy.
    # Paper: "fixed standard deviation" with C'=10.
    init_std: float = 0.1


@dataclasses.dataclass(frozen=True)
class CriticConfig:
    """Configuration for the TD3 critic Q_psi (twin Q-networks).

    The critic takes state (RL token + proprio) and action chunk as input
    and estimates Q-value.
    (Paper Section IV-B, Eq.3)
    """

    # Hidden layer dimensions (same structure as actor).
    hidden_dims: tuple[int, ...] = (256, 256)

    # Learning rate.
    learning_rate: float = 3e-4

    # Discount factor gamma.
    discount: float = 0.99

    # Target network soft update coefficient (tau).
    tau: float = 0.005

    # Number of critic updates per actor update (paper: 2 critic updates per actor).
    critic_updates_per_actor: int = 2
