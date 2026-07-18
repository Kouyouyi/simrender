"""Named-joint kinematic controller for AgiBot G1 trajectory replay."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .trajectory import AgibotTrajectory


GRIPPER_JOINTS = {
    "left": {
        "inner1": "idx31_gripper_l_inner_joint1",
        "inner2": "idx39_gripper_l_inner_joint2",
        "inner3": "idx32_gripper_l_inner_joint3",
        "outer1": "idx41_gripper_l_outer_joint1",
        "outer2": "idx49_gripper_l_outer_joint2",
        "outer3": "idx42_gripper_l_outer_joint3",
        "inner4": "idx33_gripper_l_inner_joint4",
        "outer4": "idx43_gripper_l_outer_joint4",
    },
    "right": {
        "inner1": "idx71_gripper_r_inner_joint1",
        "inner2": "idx79_gripper_r_inner_joint2",
        "inner3": "idx72_gripper_r_inner_joint3",
        "outer1": "idx81_gripper_r_outer_joint1",
        "outer2": "idx89_gripper_r_outer_joint2",
        "outer3": "idx82_gripper_r_outer_joint3",
        "inner4": "idx73_gripper_r_inner_joint4",
        "outer4": "idx83_gripper_r_outer_joint4",
    },
}


@dataclass
class AgibotG1Controller:
    """Set measured G1 poses by name while enforcing the loaded model limits."""

    model: mujoco.MjModel
    data: mujoco.MjData = field(init=False)
    clip_counts: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.data = mujoco.MjData(self.model)
        self._base_joint_id = self._joint_id("base_free_joint")
        self._base_qpos_address = int(self.model.jnt_qposadr[self._base_joint_id])
        self._is_120s = all(
            self._has_joint(GRIPPER_JOINTS[side][key])
            for side in ("left", "right")
            for key in ("inner4", "outer4")
        )

    def _has_joint(self, name: str) -> bool:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0

    def _joint_id(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"MuJoCo model has no joint {name!r}")
        return int(joint_id)

    def _set_scalar_joint(self, name: str, value: float) -> None:
        joint_id = self._joint_id(name)
        value = float(value)
        if self.model.jnt_limited[joint_id]:
            lower, upper = self.model.jnt_range[joint_id]
            clipped = float(np.clip(value, lower, upper))
            if clipped != value:
                self.clip_counts[name] = self.clip_counts.get(name, 0) + 1
            value = clipped
        address = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[address] = value

    def _set_base_pose(
        self,
        position: np.ndarray,
        orientation_xyzw: np.ndarray,
    ) -> None:
        orientation_xyzw = np.asarray(orientation_xyzw, dtype=np.float64)
        orientation_xyzw /= np.linalg.norm(orientation_xyzw)
        start = self._base_qpos_address
        self.data.qpos[start : start + 3] = position
        self.data.qpos[start + 3 : start + 7] = orientation_xyzw[[3, 0, 1, 2]]

    def _set_gripper(self, side: str, closed_fraction: float) -> None:
        joints = GRIPPER_JOINTS[side]
        if self._is_120s:
            # The trajectory stores 0=open and 1=closed, while CRT-120S joint
            # coordinates use 1 rad=open and 0 rad=closed. Joint 2/4 values
            # follow the URDF mimic tags. Joint 3 is the passive link of the
            # four-bar loop and remains at zero in this tree-only model.
            q = 1.0 - float(np.clip(closed_fraction, 0.0, 1.0))
            self._set_scalar_joint(joints["outer1"], q)
            self._set_scalar_joint(joints["inner1"], q)
            self._set_scalar_joint(joints["outer2"], q)
            self._set_scalar_joint(joints["inner2"], -q)
            self._set_scalar_joint(joints["outer3"], 0.0)
            self._set_scalar_joint(joints["inner3"], 0.0)
            self._set_scalar_joint(joints["outer4"], q)
            self._set_scalar_joint(joints["inner4"], q)
            return

        q = float(np.clip(closed_fraction, 0.0, 1.0)) * (np.pi / 4.0)
        # These values approximate the Omnipicker four-bar linkage. Joint 1 is
        # the URDF master/mimic pair; joints 2/3 keep the proxy fingers parallel.
        self._set_scalar_joint(joints["outer1"], q)
        self._set_scalar_joint(joints["inner1"], -q)
        self._set_scalar_joint(joints["outer2"], q)
        self._set_scalar_joint(joints["inner2"], -q)
        self._set_scalar_joint(joints["outer3"], -q)
        self._set_scalar_joint(joints["inner3"], q)

    def set_frame(
        self,
        trajectory: AgibotTrajectory,
        frame_index: int,
        *,
        joint_source: str = "state",
        base_mode: str = "world",
    ) -> mujoco.MjData:
        """Apply one trajectory frame and run forward kinematics."""
        if base_mode not in {"world", "relative", "fixed"}:
            raise ValueError(f"Unsupported base_mode {base_mode!r}")

        if base_mode == "fixed":
            position = np.zeros(3)
            orientation = np.asarray([0.0, 0.0, 0.0, 1.0])
        elif base_mode == "relative":
            initial = Rotation.from_quat(trajectory.base_orientation_xyzw[0])
            current = Rotation.from_quat(trajectory.base_orientation_xyzw[frame_index])
            position = initial.inv().apply(
                trajectory.base_position[frame_index] - trajectory.base_position[0]
            )
            orientation = (initial.inv() * current).as_quat()
        else:
            position = trajectory.base_position[frame_index]
            orientation = trajectory.base_orientation_xyzw[frame_index]
        self._set_base_pose(position, orientation)

        for name, value in trajectory.joint_values(frame_index, joint_source).items():
            self._set_scalar_joint(name, value)
        gripper = trajectory.effector_closed_fraction(frame_index, joint_source)
        self._set_gripper("left", gripper[0])
        self._set_gripper("right", gripper[1])

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.data
