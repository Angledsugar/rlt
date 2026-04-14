"""Phase 0: Pre-extract VLA embeddings from demo data.

Loads the frozen VLA once, extracts embeddings for all demo frames,
and saves them as .npy files. This separates the heavy VLA inference
from the lightweight RL Token training.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/extract_embeddings.py
    CUDA_VISIBLE_DEVICES=0,1 uv run python scripts/extract_embeddings.py --num-episodes 10 --batch-size 2
"""

import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import tyro

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("rlt")


def main(
    output_dir: str = "data/pen_uncap_embeddings",
    num_episodes: int = 20,
    batch_size: int = 2,
    checkpoint_path: str = "gs://openpi-assets/checkpoints/pi05_base/params",
    seed: int = 42,
) -> None:
    """Extract VLA embeddings from pen_uncap demo data.

    Args:
        output_dir: Directory to save extracted embeddings.
        num_episodes: Number of demo episodes to process.
        batch_size: Batch size for VLA forward pass (keep small for memory).
        checkpoint_path: Path to pi0.5-base checkpoint.
        seed: Random seed.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load dataset ---
    logger.info("Step 1: Loading pen_uncap dataset (%d episodes)...", num_episodes)
    sys.stdout.flush()

    from rlt.data.aloha_dataset import AlohaPenUncapDataset

    dataset = AlohaPenUncapDataset(
        repo_id="physical-intelligence/aloha_pen_uncap_diverse",
        prompt="uncap the pen",
        episodes=list(range(num_episodes)),
        download_videos=False,
    )
    dataset.load()
    num_frames = dataset.num_frames
    logger.info("Dataset loaded: %d frames", num_frames)
    sys.stdout.flush()

    # --- Step 2: Load frozen VLA ---
    logger.info("Step 2: Loading VLA backbone...")
    sys.stdout.flush()

    from rlt.vla.backbone import VLABackbone

    vla = VLABackbone(checkpoint_path)
    vla.load_from_openpi_config(
        config_name="pi05_aloha_pen_uncap",
        checkpoint_dir=checkpoint_path,
    )
    logger.info("VLA loaded. Embed dim = %d", vla.get_embed_dim())
    sys.stdout.flush()

    # --- Step 3: Extract embeddings frame by frame ---
    logger.info("Step 3: Extracting embeddings for %d frames (batch=%d)...", num_frames, batch_size)
    sys.stdout.flush()

    all_embeddings = []
    all_states = []
    all_actions = []
    all_episode_indices = []

    start_time = time.time()
    num_batches = (num_frames + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_frames)

        # Collect frames for this batch.
        frames = [dataset.get_frame(i) for i in range(start, end)]
        policy_frames = [dataset.get_policy_frame(i) for i in range(start, end)]

        # Extract embeddings (frozen VLA forward pass).
        embeddings = vla.extract_policy_embeddings(policy_frames)
        embeddings_np = np.asarray(embeddings)

        # Store results.
        all_embeddings.append(embeddings_np)
        all_states.append(np.stack([f["state"] for f in frames]))
        if "action" in frames[0]:
            all_actions.append(np.stack([f["action"] for f in frames]))

        # Get episode indices from raw dataset.
        for i in range(start, end):
            raw = dataset.dataset[i]
            all_episode_indices.append(raw.get("episode_index", -1))

        # Free GPU memory.
        del embeddings, policy_frames

        # Progress log.
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - start_time
            fps = (end) / elapsed if elapsed > 0 else 0
            eta = (num_frames - end) / fps if fps > 0 else 0
            logger.info(
                "  batch %d/%d | frames %d/%d | %.1f fps | ETA %.0fs",
                batch_idx + 1, num_batches, end, num_frames, fps, eta,
            )
            sys.stdout.flush()

    # --- Step 4: Save ---
    logger.info("Step 4: Saving embeddings to %s", output_dir)
    sys.stdout.flush()

    embeddings_all = np.concatenate(all_embeddings, axis=0)
    states_all = np.concatenate(all_states, axis=0)
    episode_indices_all = np.array(all_episode_indices)

    np.save(output_path / "embeddings.npy", embeddings_all)
    np.save(output_path / "states.npy", states_all)
    np.save(output_path / "episode_indices.npy", episode_indices_all)

    if all_actions:
        actions_all = np.concatenate(all_actions, axis=0)
        np.save(output_path / "actions.npy", actions_all)
        logger.info("Saved actions: %s", actions_all.shape)

    elapsed_total = time.time() - start_time
    logger.info("Done! Embeddings: %s, States: %s", embeddings_all.shape, states_all.shape)
    logger.info("Total time: %.1f seconds (%.1f fps)", elapsed_total, num_frames / elapsed_total)
    logger.info("Files saved to: %s", output_path.resolve())
    sys.stdout.flush()


if __name__ == "__main__":
    tyro.cli(main)
