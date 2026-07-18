from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IKSolveResult:
    qpos: np.ndarray
    position_error: float
    orientation_error: float
    iterations: int
    converged: bool


class PinocchioIKSolver:
    def __init__(
        self,
        urdf_path: str,
        controlled_joint_names: tuple[str, ...],
        tcp_link_name: str,
        damping: float = 1e-3,
        step_size: float = 0.6,
        max_iterations: int = 40,
        position_tolerance: float = 2e-3,
        orientation_tolerance: float = 5e-2,
        orientation_weight: float = 0.15,
    ):
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.q0 = pin.neutral(self.model)
        self.dof_joint_names = tuple(
            name for i, name in enumerate(self.model.names) if self.model.nqs[i] > 0
        )
        self.name_to_q_index = {name: i for i, name in enumerate(self.dof_joint_names)}
        missing = [name for name in controlled_joint_names if name not in self.name_to_q_index]
        if missing:
            raise ValueError(f"Controlled joints are not in Pinocchio model: {missing}")
        self.controlled_indices = np.asarray(
            [self.name_to_q_index[name] for name in controlled_joint_names], dtype=int
        )
        self.tcp_frame_id = self.model.getFrameId(tcp_link_name)
        if self.tcp_frame_id >= len(self.model.frames):
            raise ValueError(f"TCP link not found in model frames: {tcp_link_name}")
        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()
        self.damping = damping
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.position_tolerance = position_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.orientation_weight = orientation_weight

    def neutral_qpos(self) -> np.ndarray:
        return self.q0.copy()

    def frame_pose(self, qpos: np.ndarray, frame_name: str | None = None):
        frame_id = self.tcp_frame_id if frame_name is None else self.model.getFrameId(frame_name)
        self.pin.forwardKinematics(self.model, self.data, qpos)
        self.pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[frame_id].copy()

    def solve(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        initial_qpos: np.ndarray,
    ) -> IKSolveResult:
        pin = self.pin
        qpos = initial_qpos.copy()
        target_position = np.asarray(target_position, dtype=np.float64)
        target_rotation = np.asarray(target_rotation, dtype=np.float64)
        converged = False
        pos_err_norm = float("inf")
        ori_err_norm = float("inf")

        for iteration in range(1, self.max_iterations + 1):
            pin.forwardKinematics(self.model, self.data, qpos)
            pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[self.tcp_frame_id]
            pos_err = target_position - current.translation
            ori_err = pin.log3(current.rotation.T @ target_rotation)
            pos_err_norm = float(np.linalg.norm(pos_err))
            ori_err_norm = float(np.linalg.norm(ori_err))
            orientation_ok = (
                self.orientation_weight <= 0.0
                or ori_err_norm <= self.orientation_tolerance
            )
            if pos_err_norm <= self.position_tolerance and orientation_ok:
                converged = True
                break

            jac = pin.computeFrameJacobian(
                self.model,
                self.data,
                qpos,
                self.tcp_frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            if self.orientation_weight > 0.0:
                task_jac = np.vstack(
                    [
                        jac[:3, self.controlled_indices],
                        self.orientation_weight * jac[3:, self.controlled_indices],
                    ]
                )
                task_err = np.concatenate([pos_err, self.orientation_weight * ori_err])
            else:
                task_jac = jac[:3, self.controlled_indices]
                task_err = pos_err
            lhs = task_jac @ task_jac.T + self.damping * np.eye(task_jac.shape[0])
            delta = task_jac.T @ np.linalg.solve(lhs, task_err)
            qpos[self.controlled_indices] += self.step_size * delta
            qpos = np.clip(qpos, self.lower, self.upper)

        return IKSolveResult(
            qpos=qpos,
            position_error=pos_err_norm,
            orientation_error=ori_err_norm,
            iterations=iteration,
            converged=converged,
        )
