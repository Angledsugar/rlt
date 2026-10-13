"""Phase 1 extended experiments: find fastest convergence around lr=2e-4.

Previous results showed lr=2e-4 (bs=32) is best. Now test:
- lr=3e-4 (potential further speedup or instability)
- lr=2e-4 + bs=64 (combine best lr with larger batch)
- lr=2e-4 with 1000 steps (longer run for convergence check)
"""

import dataclasses
import logging
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("rlt.smoke_ext")


def main():
    from rlt.config.rl_token import RLTokenConfig
    from rlt.models.rl_token import RLTokenModule

    EMBED_DIM = 2048
    SEQ_LEN = 8
    NUM_FRAMES = 200

    np.random.seed(42)
    fake_embeddings = np.random.randn(NUM_FRAMES, SEQ_LEN, EMBED_DIM).astype(np.float32)

    configs = [
        {"name": "lr_3e-4_bs32", "learning_rate": 3e-4, "batch_size": 32, "steps": 500},
        {"name": "lr_2e-4_bs64", "learning_rate": 2e-4, "batch_size": 64, "steps": 500},
        {"name": "lr_2e-4_1000s", "learning_rate": 2e-4, "batch_size": 32, "steps": 1000},
    ]

    results = []
    for exp in configs:
        name = exp["name"]
        cfg = RLTokenConfig(
            token_dim=EMBED_DIM,
            vla_embed_dim=EMBED_DIM,
            learning_rate=exp["learning_rate"],
            num_train_steps=exp["steps"],
            batch_size=exp["batch_size"],
        )

        rng = jax.random.PRNGKey(42)
        rng, model_rng = jax.random.split(rng)

        model = RLTokenModule(cfg, rngs=nnx.Rngs(model_rng))
        opt = nnx.Optimizer(model, optax.adam(cfg.learning_rate))
        np_rng = np.random.default_rng(42)

        logger.info("=" * 50)
        logger.info("Experiment: %s (lr=%.1e, bs=%d, steps=%d)",
                     name, cfg.learning_rate, cfg.batch_size, exp["steps"])

        t0 = time.time()
        losses = []
        for step in range(exp["steps"]):
            indices = np_rng.integers(0, NUM_FRAMES, size=cfg.batch_size)
            vla_emb = jnp.asarray(fake_embeddings[indices])

            def loss_fn(m):
                return m.reconstruction_loss(vla_emb)

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            opt.update(grads)

            if step % 100 == 0:
                l = float(loss)
                losses.append((step, l))
                logger.info("  step %d: loss=%.4f", step, l)

        wall_time = time.time() - t0
        eval_batch = jnp.asarray(fake_embeddings[:32])
        final_loss = float(model.reconstruction_loss(eval_batch))
        results.append({
            "name": name, "lr": cfg.learning_rate, "bs": cfg.batch_size,
            "steps": exp["steps"],
            "init_loss": losses[0][1], "final_loss": final_loss,
            "wall_time_s": wall_time, "losses": losses,
        })
        logger.info("Result: init=%.4f, final=%.4f, time=%.1fs", losses[0][1], final_loss, wall_time)
        del model, opt
        jax.clear_caches()

    print("\n" + "=" * 90)
    print("Phase 1 Extended Results (dim=%d)" % EMBED_DIM)
    print("=" * 90)
    print("%-20s %8s %5s %6s %12s %12s %10s %8s" % (
        "Name", "LR", "BS", "Steps", "Init Loss", "Final Loss", "Reduction%", "Time(s)"))
    print("-" * 90)
    for r in results:
        red = 100.0 * (1.0 - r["final_loss"] / r["init_loss"])
        print("%-20s %8.1e %5d %6d %12.4f %12.4f %9.1f%% %8.1f" % (
            r["name"], r["lr"], r["bs"], r["steps"], r["init_loss"], r["final_loss"], red, r["wall_time_s"]))
    print("\nLoss Curves:")
    for r in results:
        curve = " | ".join(f"{s}:{l:.1f}" for s, l in r["losses"])
        print(f"  {r['name']}: {curve}")


if __name__ == "__main__":
    main()
