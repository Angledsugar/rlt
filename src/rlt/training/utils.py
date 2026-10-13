"""Training utilities: logging, checkpointing, metrics."""

import logging
from pathlib import Path

import orbax.checkpoint as ocp

logger = logging.getLogger("rlt")


def create_checkpoint_manager(
    checkpoint_dir: str,
    max_to_keep: int = 5,
) -> ocp.CheckpointManager:
    """Create an Orbax checkpoint manager.

    Args:
        checkpoint_dir: Directory for saving checkpoints.
        max_to_keep: Maximum number of checkpoints to retain.

    Returns:
        CheckpointManager instance.
    """
    path = Path(checkpoint_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return ocp.CheckpointManager(
        path,
        options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep),
    )


def log_metrics(step: int, metrics: dict[str, float]) -> None:
    """Log training metrics to console and optionally to WandB.

    Args:
        step: Current training step.
        metrics: Dict of metric name -> value.
    """
    parts = [f"step={step}"]
    for name, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        parts.append(f"{name}={value:.6f}")
    logger.info(" | ".join(parts))

    # Log to WandB if available.
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass
