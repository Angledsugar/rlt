"""Tests for ALOHA dataset loader and transforms."""

import numpy as np
import pytest

from rlt.data.aloha_dataset import (
    AlohaPenUncapDataset,
    _gripper_from_angular,
    _gripper_from_angular_inv,
    _gripper_to_angular,
    _joint_flip_mask,
)


class TestAlohaTransforms:
    """Test ALOHA pi-space conversion functions."""

    def test_joint_flip_mask_shape(self):
        mask = _joint_flip_mask()
        assert mask.shape == (14,)

    def test_joint_flip_mask_involutory(self):
        """Applying flip mask twice should return original."""
        mask = _joint_flip_mask()
        state = np.random.randn(14)
        flipped = mask * state
        recovered = mask * flipped
        np.testing.assert_allclose(recovered, state)

    def test_gripper_roundtrip(self):
        """gripper_from_angular_inv should invert gripper_from_angular."""
        # Test with values in expected pi-space range.
        values = np.array([0.0, 0.3, 0.5, 0.8, 1.0])
        # Forward: pi-space -> aloha angular.
        aloha_vals = _gripper_from_angular(values)
        # Inverse: aloha angular -> pi-space.
        recovered = _gripper_from_angular_inv(aloha_vals)
        np.testing.assert_allclose(recovered, values, atol=1e-6)

    def test_state_transform_dimensions(self):
        """Verify state transforms preserve dimensions."""
        state = np.random.rand(14).astype(np.float32)
        # Simulate decode_state: flip + gripper transform.
        transformed = _joint_flip_mask() * state.copy()
        transformed[[6, 13]] = _gripper_to_angular(transformed[[6, 13]])
        assert transformed.shape == (14,)
        assert np.isfinite(transformed).all()


class TestAlohaPenUncapDataset:
    """Test dataset initialization (without actual download)."""

    def test_init(self):
        ds = AlohaPenUncapDataset(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            prompt="uncap the pen",
        )
        assert ds.repo_id == "physical-intelligence/aloha_pen_uncap_diverse"
        assert ds.prompt == "uncap the pen"
        assert ds.image_size == (224, 224)
        assert ds.adapt_to_pi is True

    def test_collate(self):
        """Test batch collation with synthetic frames."""
        ds = AlohaPenUncapDataset()

        frames = []
        for _ in range(4):
            frames.append({
                "images": {
                    "base_0_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                    "left_wrist_0_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                    "right_wrist_0_rgb": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                },
                "image_masks": {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.True_,
                },
                "state": np.random.randn(14).astype(np.float32),
                "action": np.random.randn(14).astype(np.float32),
                "prompt": "uncap the pen",
            })

        batch = ds._collate(frames)

        assert batch["images"]["base_0_rgb"].shape == (4, 224, 224, 3)
        assert batch["images"]["left_wrist_0_rgb"].shape == (4, 224, 224, 3)
        assert batch["images"]["right_wrist_0_rgb"].shape == (4, 224, 224, 3)
        assert batch["state"].shape == (4, 14)
        assert batch["action"].shape == (4, 14)
        assert batch["prompt"] == "uncap the pen"
