"""Phase 1 smoke test with synthetic embeddings.

Verifies the RL token encoder-decoder training loop works end-to-end
without needing real VLA embeddings or GCS downloads.

Runs at full paper dimensions (token_dim=2048) on GPU 0,1.
"""

import dataclasses
import logging
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("rlt.smoke")


def main():
    from rlt.config.base import RLTConfig
    from rlt.config.rl_token import RLTokenConfig
    from rlt.models.rl_token import RLTokenModule

    EMBED_DIM = 2048
    SEQ_LEN = 8
    NUM_FRAMES = 200
    STEPS = 500

    np.random.seed(42)
    fake_embeddings = np.random.randn(NUM_FRAMES, SEQ_LEN, EMBED_DIM).astype(np.float32)

    from pathlib import Path
    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="rlt_smoke_"))
    np.save(tmpdir / "embeddings.npy", fake_embeddings)
    logger.info("Synthetic embeddings: %s, shape=%s", tmpdir, fake_embeddings.shape)
    logger.info("JAX devices: %s", jax.devices())

    configs = [
        {"name": "lr_5e-5", "learning_rate": 5e-5, "batch_size": 32},
        {"name": "lr_1e-4", "learning_rate": 1e-4, "batch_size": 32},
        {"name": "lr_2e-4", "learning_rate": 2e-4, "batch_size": 32},
        {"name": "lr_1e-4_bs64", "learning_rate": 1e-4, "batch_size": 64},
    ]

    results = []
    for exp in configs:
        name = exp["name"]
        rl_token_cfg = RLTokenConfig(
            token_dim=EMBED_DIM,
            vla_embed_dim=EMBED_DIM,
            learning_rate=exp["learning_rate"],
            num_train_steps=STEPS,
            batch_size=exp["batch_size"],
        )

        rng = jax.random.PRNGKey(42)
        rng, model_rng = jax.random.split(rng)

        rl_token = RLTokenModule(rl_token_cfg, rngs=nnx.Rngs(model_rng))
        optimizer = nnx.Optimizer(rl_token, optax.adam(rl_token_cfg.learning_rate))
        np_rng = np.random.default_rng(42)

        logger.info("=" * 50)
        logger.info("Experiment: %s (lr=%.1e, bs=%d, steps=%d)",
                     name, rl_token_cfg.learning_rate, rl_token_cfg.batch_size, STEPS)

        t0 = time.time()
        losses = []
        for step in range(STEPS):
            indices = np_rng.integers(0, NUM_FRAMES, size=rl_token_cfg.batch_size)
            vla_emb = jnp.asarray(fake_embeddings[indices])

            def loss_fn(model):
                return model.reconstruction_loss(vla_emb)

            loss, grads = nnx.value_and_grad(loss_fn)(rl_token)
            optimizer.update(grads)

            if step % 100 == 0:
                l = float(loss)
                losses.append((step, l))
                logger.info("  step %d: loss=%.4f", step, l)

        wall_time = time.time() - t0

        eval_batch = jnp.asarray(fake_embeddings[:32])
        final_loss = float(rl_token.reconstruction_loss(eval_batch))
        results.append({
            "name": name,
            "lr": rl_token_cfg.learning_rate,
            "batch_size": rl_token_cfg.batch_size,
            "init_loss": losses[0][1] if losses else float("nan"),
            "final_loss": final_loss,
            "wall_time_s": wall_time,
            "losses": losses,
        })
        logger.info("Result: init=%.4f, final=%.4f, wall_time=%.1fs",
                     losses[0][1], final_loss, wall_time)

        # Free GPU memory between experiments.
        del rl_token, optimizer, grads
        jax.clear_caches()

    # Summary table
    print("\n" + "=" * 80)
    print("Phase 1 Smoke Test Results (dim=%d, seq_len=%d, frames=%d, steps=%d)" % (EMBED_DIM, SEQ_LEN, NUM_FRAMES, STEPS))
    print("=" * 80)
    print("%-18s %8s %5s %12s %12s %10s %8s" % ("Name", "LR", "BS", "Init Loss", "Final Loss", "Reduction%", "Time(s)"))
    print("-" * 80)
    for r in results:
        reduction = 100.0 * (1.0 - r["final_loss"] / r["init_loss"]) if r["init_loss"] > 0 else 0
        print("%-18s %8.1e %5d %12.4f %12.4f %9.1f%% %8.1f" % (
            r["name"], r["lr"], r["batch_size"], r["init_loss"], r["final_loss"], reduction, r["wall_time_s"]))

    # Loss curves
    print("\n" + "=" * 80)
    print("Loss Curves (step: loss)")
    print("=" * 80)
    for r in results:
        curve = " | ".join(f"{s}:{l:.2f}" for s, l in r["losses"])
        print(f"  {r['name']}: {curve}")

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
