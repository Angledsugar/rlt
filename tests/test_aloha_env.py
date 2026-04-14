"""Tests for ALOHA environment wrapper."""

import numpy as np
import pytest

from rlt.envs.aloha_env import (
    ALOHA_ACTION_DIM,
    ALOHA_IMAGE_SIZE,
    ALOHA_PROPRIO_DIM,
    CAMERA_MAPPING,
)


class TestAlohaEnvConstants:
    """Test ALOHA environment constants."""

    def test_action_dim(self):
        assert ALOHA_ACTION_DIM == 14

    def test_proprio_dim(self):
        assert ALOHA_PROPRIO_DIM == 14

    def test_image_size(self):
        assert ALOHA_IMAGE_SIZE == (224, 224)

    def test_camera_mapping(self):
        assert "cam_high" in CAMERA_MAPPING
        assert "cam_left_wrist" in CAMERA_MAPPING
        assert "cam_right_wrist" in CAMERA_MAPPING
        assert CAMERA_MAPPING["cam_high"] == "base_0_rgb"
        assert CAMERA_MAPPING["cam_left_wrist"] == "left_wrist_0_rgb"
        assert CAMERA_MAPPING["cam_right_wrist"] == "right_wrist_0_rgb"
