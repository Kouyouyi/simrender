"""Galbot/Galaxea robot spec for MuJoCo visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from retarget_galbot.robots import RobotSpec, register
from retarget_galbot.robots.mjcf_patch import patch_mjcf_local

_ROOT = Path(__file__).resolve().parents[2]
GALBOT_ASSET_DIR = _ROOT / "assets" / "robots" / "galbot"
GALBOT_MJCF_PATH = GALBOT_ASSET_DIR / "galbot_one_golf_with_sites.xml"
GALBOT_URDF_PATH = GALBOT_ASSET_DIR / "galbot_one_golf.urdf"

GALBOT_ACTION_JOINT_NAMES = [
    "leg_joint1",
    "leg_joint2",
    "leg_joint3",
    "leg_joint4",
    "leg_joint5",
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_arm_joint7",
    "left_gripper_joint",
    "left_gripper_r_inner_knuckle_joint",
    "left_gripper_r_finger_joint",
    "left_gripper_l_knuckle_joint",
    "left_gripper_l_inner_knuckle_joint",
    "left_gripper_l_finger_joint",
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_joint7",
    "right_gripper_joint",
    "right_gripper_r_inner_knuckle_joint",
    "right_gripper_r_finger_joint",
    "right_gripper_l_knuckle_joint",
    "right_gripper_l_inner_knuckle_joint",
    "right_gripper_l_finger_joint",
    "head_joint1",
    "head_joint2",
]

GALBOT_ACTION_DIM = len(GALBOT_ACTION_JOINT_NAMES)

GALBOT_MODALITY = {
    "action": {
        "qpos": {
            "start": 0,
            "end": GALBOT_ACTION_DIM,
        }
    }
}


def _galbot_body_predicate(body_name: str) -> bool:
    return "gripper" in body_name or "arm_link" in body_name


def _galbot_qpos_writer_factory(model):
    """Action layout follows Pinocchio movable-joint qpos order."""
    import mujoco

    qpos_idx = []
    action_idx = []
    for idx, name in enumerate(GALBOT_ACTION_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        qpos_idx.append(int(model.jnt_qposadr[jid]))
        action_idx.append(idx)

    qpos_idx_arr = np.asarray(qpos_idx, dtype=int)
    action_idx_arr = np.asarray(action_idx, dtype=int)

    def write(qpos_full: np.ndarray, action: np.ndarray) -> None:
        qpos_full[qpos_idx_arr] = np.asarray(action)[action_idx_arr]

    return write


GALBOT_SPEC = register(RobotSpec(
    name="galbot",
    display_name="Galbot",
    mjcf_path=GALBOT_MJCF_PATH,
    action_dim=GALBOT_ACTION_DIM,
    action_joint_names=list(GALBOT_ACTION_JOINT_NAMES),
    hand_body_predicate=_galbot_body_predicate,
    patch_mjcf=patch_mjcf_local,
    standing_height=0.0,
    head_mesh_names=("head_link2_collision_01",),
    left_shoulder_body="left_arm_link1",
    right_shoulder_body="right_arm_link1",
    left_wrist_body="left_arm_link7",
    right_wrist_body="right_arm_link7",
    has_floating_base=False,
    robot_type="galbot_one_golf",
    modality=GALBOT_MODALITY,
    qpos_writer_factory=_galbot_qpos_writer_factory,
))
