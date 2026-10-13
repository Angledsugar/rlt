"""MuJoCo task definition for bimanual pen uncapping.

The task requires the ALOHA bimanual robot to:
1. Grasp the pen body with one gripper.
2. Grasp the cap with the other gripper.
3. Pull the cap off the pen.

The cap is a child body of the pen, connected by a slide joint along the
barrel axis.  Damping and friction on the joint simulate the snap-fit.
When the slide displacement exceeds a threshold, the cap is "off".

Reward is staged (0-4) following the same convention as gym-aloha:
  0: nothing
  1: one gripper touching the pen body
  2: one gripper touching the cap
  3: both grippers touching their respective targets (pen + cap)
  4: cap separated from pen (success)
"""

import collections

import numpy as np
from dm_control.suite import base

from gym_aloha.constants import (
    START_ARM_POSE,
    normalize_puppet_gripper_position,
    normalize_puppet_gripper_velocity,
    unnormalize_puppet_gripper_position,
)

PEN_POSE = [None]

# Slide displacement threshold to count as "cap removed".
_CAP_SLIDE_THRESHOLD = 0.03


def sample_pen_pose(seed=None):
    """Sample a random pen position on the table (lying flat along Y)."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(-0.05, 0.15)
    y = rng.uniform(0.42, 0.52)
    z = 0.008  # table surface at z=0, pen half-height 0.007
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[x, y, z], quat])


class PenUncapTask(base.Task):
    """Bimanual pen uncapping task for ALOHA."""

    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    # -- action / observation interface (same as BimanualViperXTask) ----------

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7:13]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[13]

        left_gripper_action = unnormalize_puppet_gripper_position(normalized_left_gripper_action)
        right_gripper_action = unnormalize_puppet_gripper_position(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate(
            [left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action]
        )
        super().before_step(env_action, physics)

    def initialize_episode(self, physics):
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert PEN_POSE[0] is not None
            # qpos layout: [robot(16), pen_free(7), cap_slide(1)]
            physics.named.data.qpos[16:23] = PEN_POSE[0]
            physics.named.data.qpos[23] = 0.0  # cap on pen (slide=0)
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [normalize_puppet_gripper_position(left_qpos_raw[6])]
        right_gripper_qpos = [normalize_puppet_gripper_position(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [normalize_puppet_gripper_velocity(left_qvel_raw[6])]
        right_gripper_qvel = [normalize_puppet_gripper_velocity(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        # pen_free(7) + cap_slide(1)
        return physics.data.qpos.copy()[16:]

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = {}
        obs["images"]["top"] = physics.render(height=480, width=640, camera_id="top")
        obs["images"]["angle"] = physics.render(height=480, width=640, camera_id="angle")
        obs["images"]["vis"] = physics.render(height=480, width=640, camera_id="front_close")
        return obs

    # -- reward ---------------------------------------------------------------

    def get_reward(self, physics):
        contacts = self._get_contact_pairs(physics)

        touch_pen_left = self._touches_pen(contacts, "left")
        touch_pen_right = self._touches_pen(contacts, "right")
        touch_cap_left = self._touches_cap(contacts, "left")
        touch_cap_right = self._touches_cap(contacts, "right")

        pen_grasped = touch_pen_left or touch_pen_right
        cap_grasped = touch_cap_left or touch_cap_right

        cap_separated = self._cap_separated(physics)

        reward = 0
        if pen_grasped:
            reward = 1
        if cap_grasped:
            reward = 2
        if pen_grasped and cap_grasped:
            reward = 3
        if cap_separated and (pen_grasped or cap_grasped):
            reward = 4
        return reward

    @staticmethod
    def _get_contact_pairs(physics):
        pairs = set()
        for i in range(physics.data.ncon):
            g1 = physics.model.id2name(physics.data.contact[i].geom1, "geom")
            g2 = physics.model.id2name(physics.data.contact[i].geom2, "geom")
            pairs.add((g1, g2))
            pairs.add((g2, g1))
        return pairs

    @staticmethod
    def _touches_pen(contacts, side):
        finger = f"vx300s_{side}/10_{side}_gripper_finger"
        return (
            ("pen_barrel", finger) in contacts
            or ("pen_tip", finger) in contacts
        )

    @staticmethod
    def _touches_cap(contacts, side):
        finger = f"vx300s_{side}/10_{side}_gripper_finger"
        return (
            ("cap_outer", finger) in contacts
            or ("cap_top", finger) in contacts
        )

    @staticmethod
    def _cap_separated(physics):
        # Cap slide joint displacement: 0 = on pen, >threshold = off.
        cap_slide = physics.data.qpos[23]
        return cap_slide > _CAP_SLIDE_THRESHOLD
