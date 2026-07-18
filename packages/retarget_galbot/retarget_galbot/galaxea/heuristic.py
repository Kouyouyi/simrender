from __future__ import annotations

import numpy as np

from .annotations import ActionSegment
from .config import RetargetConfig
from .features import HandFeatureSequence, HandFeatures
from .urdf_utils import JointInfo, joint_limit_map


class HeuristicGalbotRetargeter:
    """Gripper mapping helpers shared by the bimanual Pinocchio retargeter."""

    def __init__(self, config: RetargetConfig, joints: list[JointInfo]):
        self.config = config
        self.joints = joints
        self.joint_names = tuple(joint.name for joint in joints)
        self.limits = joint_limit_map(joints)
        self._validate_joint_names(config.arm_joint_names + config.gripper_joint_names)

    def _validate_joint_names(self, names: tuple[str, ...]) -> None:
        missing = [name for name in names if name not in self.limits]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Configured joints are not present in URDF: {joined}")

    def _gripper_value(self, features: HandFeatures, segment: ActionSegment | None) -> float:
        del segment
        if features.gripper_closed:
            return self.config.gripper.close_value
        return self.config.gripper.open_value

    def _mimic_adjusted_value(self, name: str, master_value: float) -> float:
        joint = next(joint for joint in self.joints if joint.name == name)
        if joint.mimic_joint:
            return joint.mimic_multiplier * master_value + joint.mimic_offset
        return master_value

    def _clip(self, name: str, value: float) -> float:
        lower, upper = self.limits[name]
        return float(np.clip(value, lower, upper))

    @staticmethod
    def _first_valid_palm(sequence: HandFeatureSequence) -> np.ndarray:
        valid_indices = np.flatnonzero(sequence.valid)
        index = int(valid_indices[0]) if len(valid_indices) else 0
        return sequence.features[index].palm_position.copy()
