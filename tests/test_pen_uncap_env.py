"""Tests for the pen_uncap simulation environment."""

import numpy as np
import pytest


class TestPenUncapEnv:
    """Test the low-level PenUncapEnv (Gymnasium)."""

    @pytest.fixture()
    def env(self):
        from rlt.envs.pen_uncap_env import PenUncapEnv
        e = PenUncapEnv()
        yield e
        e.close()

    def test_reset(self, env):
        obs, info = env.reset(seed=0)
        assert "pixels" in obs
        assert obs["pixels"]["top"].shape == (480, 640, 3)
        assert obs["agent_pos"].shape == (14,)
        assert info["is_success"] is False

    def test_step_shape(self, env):
        env.reset(seed=0)
        action = np.zeros(14, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs["pixels"]["top"].shape == (480, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_pen_cap_initial_positions(self, env):
        env.reset(seed=42)
        physics = env._env.physics
        pen_pos = physics.named.data.xpos["pen_body"]
        cap_pos = physics.named.data.xpos["pen_cap"]
        # Cap should be at the +Y end of the pen barrel (pen lies along Y).
        assert cap_pos[1] > pen_pos[1]
        # Same height (both lying flat on table).
        assert abs(cap_pos[2] - pen_pos[2]) < 0.02

    def test_reward_range(self, env):
        env.reset(seed=0)
        for _ in range(10):
            action = np.zeros(14, dtype=np.float32)
            _, reward, terminated, _, _ = env.step(action)
            assert 0 <= reward <= 4
            if terminated:
                break

    def test_randomized_pen_pose(self, env):
        positions = []
        for seed in range(5):
            env.reset(seed=seed)
            pen_pos = env._env.physics.named.data.xpos["pen_body"].copy()
            positions.append(pen_pos)
        # Positions should vary across seeds.
        positions = np.array(positions)
        assert np.std(positions[:, 0]) > 0.0  # x varies


class TestAlohaSimEnvPenUncap:
    """Test PenUncapEnv through the AlohaSimEnv wrapper."""

    @pytest.fixture()
    def env(self):
        from rlt.envs.aloha_env import AlohaSimEnv
        return AlohaSimEnv(
            task="gym_aloha/AlohaPenUncap-v0",
            prompt="uncap the pen",
            max_episode_steps=300,
        )

    def test_reset_obs_format(self, env):
        obs = env.reset()
        assert "images" in obs
        assert "state" in obs
        assert "policy_observation" in obs
        assert obs["policy_observation"]["prompt"] == "uncap the pen"

    def test_action_chunk_step(self, env):
        env.reset()
        chunk = np.zeros((10, 14), dtype=np.float32)
        obs, reward, done, info = env.step(chunk)
        assert "images" in obs
        assert isinstance(reward, float)

    def test_proprio_extraction(self, env):
        obs = env.reset()
        proprio = env.get_proprio(obs)
        assert proprio.shape == (14,)
        assert proprio.dtype == np.float32
