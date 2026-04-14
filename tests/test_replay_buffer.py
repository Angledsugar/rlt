"""Tests for the replay buffer."""

import numpy as np
import pytest

from rlt.rl.replay_buffer import ReplayBuffer

RL_TOKEN_DIM = 64
PROPRIO_DIM = 14
ACTION_CHUNK_DIM = 140


@pytest.fixture
def buffer():
    return ReplayBuffer(
        capacity=100,
        rl_token_dim=RL_TOKEN_DIM,
        proprio_dim=PROPRIO_DIM,
        action_chunk_dim=ACTION_CHUNK_DIM,
    )


class TestReplayBuffer:
    def test_add_and_size(self, buffer):
        assert len(buffer) == 0
        buffer.add(
            rl_token=np.zeros(RL_TOKEN_DIM),
            proprio=np.zeros(PROPRIO_DIM),
            action=np.zeros(ACTION_CHUNK_DIM),
            reference_action=np.zeros(ACTION_CHUNK_DIM),
            reward=0.0,
            next_rl_token=np.zeros(RL_TOKEN_DIM),
            next_proprio=np.zeros(PROPRIO_DIM),
            done=False,
        )
        assert len(buffer) == 1

    def test_capacity_overflow(self, buffer):
        for i in range(150):
            buffer.add(
                rl_token=np.full(RL_TOKEN_DIM, i, dtype=np.float32),
                proprio=np.zeros(PROPRIO_DIM),
                action=np.zeros(ACTION_CHUNK_DIM),
                reference_action=np.zeros(ACTION_CHUNK_DIM),
                reward=0.0,
                next_rl_token=np.zeros(RL_TOKEN_DIM),
                next_proprio=np.zeros(PROPRIO_DIM),
                done=False,
            )
        assert len(buffer) == 100

    def test_sample(self, buffer):
        for i in range(20):
            buffer.add(
                rl_token=np.random.randn(RL_TOKEN_DIM).astype(np.float32),
                proprio=np.random.randn(PROPRIO_DIM).astype(np.float32),
                action=np.random.randn(ACTION_CHUNK_DIM).astype(np.float32),
                reference_action=np.random.randn(ACTION_CHUNK_DIM).astype(np.float32),
                reward=1.0,
                next_rl_token=np.random.randn(RL_TOKEN_DIM).astype(np.float32),
                next_proprio=np.random.randn(PROPRIO_DIM).astype(np.float32),
                done=False,
            )
        batch = buffer.sample(8)
        assert batch["rl_token"].shape == (8, RL_TOKEN_DIM)
        assert batch["proprio"].shape == (8, PROPRIO_DIM)
        assert batch["action"].shape == (8, ACTION_CHUNK_DIM)
        assert batch["reference_action"].shape == (8, ACTION_CHUNK_DIM)
        assert batch["reward"].shape == (8, 1)
        assert batch["next_rl_token"].shape == (8, RL_TOKEN_DIM)
        assert batch["next_proprio"].shape == (8, PROPRIO_DIM)
        assert batch["done"].shape == (8, 1)

    def test_add_episode(self, buffer):
        T = 20
        buffer.add_episode(
            rl_tokens=np.random.randn(T, RL_TOKEN_DIM).astype(np.float32),
            proprios=np.random.randn(T, PROPRIO_DIM).astype(np.float32),
            actions=np.random.randn(T, ACTION_CHUNK_DIM).astype(np.float32),
            reference_actions=np.random.randn(T, ACTION_CHUNK_DIM).astype(np.float32),
            rewards=np.zeros(T, dtype=np.float32),
            dones=np.zeros(T, dtype=np.float32),
            stride=2,
        )
        # With stride=2 and T=20: indices 0, 2, 4, ..., 18 = 10 transitions.
        assert len(buffer) == 10
