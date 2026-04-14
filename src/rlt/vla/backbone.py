"""Frozen VLA backbone wrapper for pi0.5-base.

Provides an interface to the pretrained pi0.5 VLA model from openpi.
The VLA is kept frozen during online RL — only the RL token encoder/decoder,
actor, and critic are trained.

This module handles:
1. Loading the pi0.5-base checkpoint via openpi's policy system.
2. Running forward passes to extract internal embeddings for RL token.
3. Generating reference action chunks via the VLA's sampling procedure.
4. ALOHA-specific input/output transforms.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger("rlt")


class VLABackbone:
    """Wrapper around the frozen pi0.5 VLA model.

    Uses openpi's Pi0 model with pi05=True configuration.
    All parameters are frozen — no gradients flow through the VLA.
    """

    def __init__(
        self,
        checkpoint_path: str,
        rng: jax.Array | None = None,
    ):
        """Initialize VLA backbone.

        Args:
            checkpoint_path: Path to pi0.5-base checkpoint (gs:// or local path).
            rng: PRNG key for model initialization.
        """
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._config = None
        self._policy = None

        if rng is not None:
            self.load(rng)

    @staticmethod
    def _checkpoint_root(checkpoint_path: str) -> str:
        """Return checkpoint root, accepting either .../pi05_base or .../pi05_base/params."""
        path = str(checkpoint_path).rstrip("/")
        if path.endswith("/params"):
            return path[: -len("/params")]
        return path

    @staticmethod
    def _params_path(checkpoint_path: str) -> str:
        """Return params path, accepting either checkpoint root or direct params path."""
        path = str(checkpoint_path).rstrip("/")
        if path.endswith("/params"):
            return path
        return f"{path}/params"

    def load(self, rng: jax.Array) -> None:
        """Load the pi0.5-base model and checkpoint.

        Uses openpi's native restore_params + config.load() which correctly
        handles the checkpoint tree structure (unstructured params PyTree).
        """
        from openpi.models import model as _model
        from openpi.models.pi0_config import Pi0Config
        from openpi.shared.download import maybe_download

        logger.info("Loading pi0.5-base model from %s", self.checkpoint_path)

        self._config = Pi0Config(pi05=True)

        # Download checkpoint if remote (GCS).
        local_path = maybe_download(self._params_path(self.checkpoint_path))

        # Use openpi's restore_params to load unstructured params,
        # then config.load() to create model with those params.
        logger.info("Restoring params from %s", local_path)
        params = _model.restore_params(local_path, dtype=jnp.bfloat16)
        self._model = self._config.load(params)
        self._model.eval()

        logger.info("VLA backbone loaded successfully.")

    def load_from_openpi_config(
        self,
        config_name: str = "pi05_aloha_pen_uncap",
        checkpoint_dir: str | None = None,
    ) -> None:
        """Load VLA using openpi's training config system.

        This is the recommended way to load for ALOHA tasks, as it
        sets up all the correct data transforms automatically.

        Args:
            config_name: openpi training config name.
            checkpoint_dir: Override checkpoint directory.
        """
        from openpi.training.config import get_config

        train_config = get_config(config_name)
        ckpt_dir = self._checkpoint_root(checkpoint_dir or self.checkpoint_path)

        from openpi.policies.policy_config import create_trained_policy

        self._policy = create_trained_policy(train_config, ckpt_dir)
        self._model = self._policy._model
        self._config = train_config.model
        logger.info("VLA loaded from openpi config: %s", config_name)

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("VLA backbone not loaded. Call load() first.")
        return self._model

    @property
    def config(self):
        if self._config is None:
            raise RuntimeError("VLA backbone not loaded. Call load() first.")
        return self._config

    def extract_embeddings(self, observation: dict) -> jax.Array:
        """Extract final-layer token embeddings from VLA for RL token creation.

        Runs the VLA's prefix embedding (images + language) and returns
        the internal representations z_1:M that will be compressed into
        the RL token.

        Args:
            observation: Dict with keys matching openpi Observation format:
                - images: dict of [batch, H, W, 3] camera images
                - image_masks: dict of [batch] boolean masks
                - state: [batch, 14] proprioceptive state
                - tokenized_prompt: [batch, max_len] token ids
                - tokenized_prompt_mask: [batch, max_len] mask

        Returns:
            embeddings: [batch, seq_len, embed_dim] - VLA internal embeddings.
        """
        obs = self._to_observation(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.model.embed_prefix(obs)
        return prefix_tokens

    def sample_actions(
        self,
        rng: jax.Array,
        observation: dict,
        num_steps: int = 10,
    ) -> jax.Array:
        """Generate reference action chunks using the VLA's flow-matching sampler.

        These reference actions a_hat_{1:C} are used by the RL actor for
        conditioning and BC regularization.

        Args:
            rng: PRNG key for sampling.
            observation: Dict observation (same format as extract_embeddings).
            num_steps: Number of denoising steps (default 10).

        Returns:
            actions: [batch, action_horizon, action_dim] - VLA reference actions.
        """
        obs = self._to_observation(observation)
        return self.model.sample_actions(rng, obs, num_steps=num_steps)

    def _transform_policy_inputs(self, observation: dict) -> dict:
        """Apply openpi policy input transforms to an unbatched raw observation."""
        if self._policy is None:
            raise RuntimeError(
                "Policy transforms not initialized. Call load_from_openpi_config() "
                "or use extract_embeddings()/sample_actions() with pre-tokenized observations."
            )

        # openpi transforms may mutate nested values, so copy leaves before applying them.
        inputs = jax.tree.map(lambda x: x, observation)
        return self._policy._input_transform(inputs)

    @staticmethod
    def _batch_policy_inputs(transformed_inputs: list[dict]) -> dict:
        """Stack unbatched transformed policy inputs into a model batch."""
        return jax.tree.map(
            lambda *xs: jnp.asarray(np.stack(xs, axis=0)),
            *transformed_inputs,
        )

    def _policy_observation_batch(self, observation: dict | list[dict]) -> tuple[object, dict]:
        """Convert raw ALOHA observation(s) to an openpi Observation via policy transforms."""
        from openpi.models.model import Observation

        if isinstance(observation, list):
            if not observation:
                raise ValueError("Cannot create a policy observation batch from an empty list.")
            transformed = [self._transform_policy_inputs(obs) for obs in observation]
            batch = self._batch_policy_inputs(transformed)
        else:
            transformed = self._transform_policy_inputs(observation)
            batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], transformed)

        return Observation.from_dict(batch), batch

    def extract_policy_embeddings(self, observation: dict | list[dict]) -> jax.Array:
        """Extract embeddings after applying the official openpi policy input transforms.

        Args:
            observation: Raw unbatched ALOHA observation dict, or a list of such dicts:
                {"images": {"cam_high": [C,H,W], ...}, "state": [14], "prompt": str}

        Returns:
            embeddings: [batch, seq_len, embed_dim]
        """
        obs, _ = self._policy_observation_batch(observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.model.embed_prefix(obs)
        return prefix_tokens

    def sample_policy_actions(
        self,
        rng: jax.Array,
        observation: dict,
        num_steps: int = 10,
    ) -> jax.Array:
        """Sample actions after applying openpi input and output policy transforms.

        The returned actions are in the environment/output action space (for ALOHA,
        [1, action_horizon, 14]) instead of pi0.5's internal padded 32-dim action space.
        """
        obs, batch = self._policy_observation_batch(observation)
        actions = self.model.sample_actions(rng, obs, num_steps=num_steps)

        # openpi output transforms expect an unbatched dict, matching Policy.infer().
        outputs = {
            "state": np.asarray(batch["state"][0]),
            "actions": np.asarray(actions[0]),
        }
        outputs = self._policy._output_transform(outputs)
        return jnp.asarray(np.asarray(outputs["actions"], dtype=np.float32)[None, ...])

    def infer_with_policy(self, observation: dict) -> dict:
        """Run full inference using openpi's policy pipeline.

        Uses the complete transform chain (including ALOHA-specific
        input/output transforms, normalization, delta actions, etc.).
        Only available when loaded via load_from_openpi_config().

        Args:
            observation: Raw ALOHA observation dict with keys:
                - images: dict of camera images
                - state: [14] proprioceptive state
                - prompt: str language instruction

        Returns:
            Dict with "actions" key containing [action_horizon, 14] actions.
        """
        if self._policy is None:
            raise RuntimeError(
                "Policy not initialized. Use load_from_openpi_config() "
                "to enable infer_with_policy()."
            )
        return self._policy.infer(observation)

    def _to_observation(self, obs_dict: dict):
        """Convert a dict to openpi Observation, adding JAX batch dims."""
        from openpi.models.model import Observation

        # Convert numpy arrays to JAX and ensure batch dims.
        jax_obs = {}
        for key in ("images", "image_masks", "state", "tokenized_prompt", "tokenized_prompt_mask"):
            if key in obs_dict:
                val = obs_dict[key]
                if isinstance(val, dict):
                    jax_obs[key] = {k: jnp.asarray(v) for k, v in val.items()}
                else:
                    jax_obs[key] = jnp.asarray(val)

        return Observation(**jax_obs)

    def get_embed_dim(self) -> int:
        """Return the VLA embedding dimension (Gemma 2B backbone width)."""
        return 2048
