"""Download pi0.5-base checkpoint from GCS.

Usage:
    uv run python scripts/download_checkpoint.py
    uv run python scripts/download_checkpoint.py --checkpoint pi05_base
"""

import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import logging

import tyro

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("rlt")

PI05_CHECKPOINTS = {
    "pi05_base": "gs://openpi-assets/checkpoints/pi05_base",
}


def main(
    checkpoint: str = "pi05_base",
    output_dir: str = "~/.cache/openpi",
) -> None:
    """Download pi0.5 checkpoint.

    Args:
        checkpoint: Checkpoint name (e.g. pi05_base).
        output_dir: Local directory to store checkpoints.
    """
    if checkpoint not in PI05_CHECKPOINTS:
        available = ", ".join(PI05_CHECKPOINTS.keys())
        raise ValueError(f"Unknown checkpoint: {checkpoint}. Available: {available}")

    gcs_path = PI05_CHECKPOINTS[checkpoint]
    logger.info("Downloading %s from %s", checkpoint, gcs_path)

    from openpi.shared.download import maybe_download

    local_path = maybe_download(gcs_path)
    logger.info("Checkpoint available at: %s", local_path)


if __name__ == "__main__":
    tyro.cli(main)
