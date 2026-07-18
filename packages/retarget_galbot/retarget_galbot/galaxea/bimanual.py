from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .annotations import ActionSegment
from .config import RetargetConfig
from .features import HandFeatureSequence
from .heuristic import HeuristicGalbotRetargeter
from .pinocchio_ik import PinocchioIKSolver
from .urdf_utils import JointInfo


SOURCE_SHOULDER_OFFSET_HEAD = np.array([0.0, -0.05, 0.0], dtype=np.float64)
SOURCE_SHOULDER_HALF_WIDTH = 0.18
MIN_ARM_LENGTH_SAMPLE = 0.12
ARM_LENGTH_PERCENTILE = 95.0


@dataclass
class BimanualRetargetFrame:
    frame: int
    timestamp: float
    qpos: list[float]
    valid: bool
    left: dict[str, Any]
    right: dict[str, Any]
    action_id: int | None
    verbs: list[str]


class BimanualDexRetargeter:
    """Bimanual Galbot retargeter backed by Pinocchio TCP IK."""

    def __init__(self, config: RetargetConfig, joints: list[JointInfo]):
        self.config = config
        self.joints = joints
        self.left_config = self._side_config("left")
        self.right_config = self._side_config("right")
        self.left_retargeter = HeuristicGalbotRetargeter(self.left_config, joints)
        self.right_retargeter = HeuristicGalbotRetargeter(self.right_config, joints)
        self.left_solver = self._make_solver(self.left_config)
        self.right_solver = self._make_solver(self.right_config)
        if self.left_solver.dof_joint_names != self.right_solver.dof_joint_names:
            raise ValueError("Left/right IK solvers produced inconsistent joint order")
        self.joint_names = self.left_solver.dof_joint_names
        self.name_to_q_index = {
            name: index for index, name in enumerate(self.joint_names)
        }
        self.initial_qpos = self._initial_qpos()
        head_link_name = str(self.config.raw.get("head_link_name", "head_end_effector_mount_link"))
        head_pose0 = self.left_solver.frame_pose(self.initial_qpos, head_link_name)
        self.head_pose0 = head_pose0
        self.head_link_name = head_link_name
        left_tcp0 = self.left_solver.frame_pose(self.initial_qpos)
        right_tcp0 = self.right_solver.frame_pose(self.initial_qpos)
        self.left_tcp_offset_in_head = head_pose0.rotation.T @ (
            left_tcp0.translation - head_pose0.translation
        )
        self.right_tcp_offset_in_head = head_pose0.rotation.T @ (
            right_tcp0.translation - head_pose0.translation
        )
        self.left_tcp0 = left_tcp0
        self.right_tcp0 = right_tcp0
        self.position_mapping = str(
            self.config.raw.get("position_mapping", "shoulder_scaled")
        ).lower()
        if self.position_mapping not in {"shoulder_scaled", "absolute", "delta"}:
            raise ValueError(
                "position_mapping must be 'shoulder_scaled', 'absolute' or 'delta', "
                f"got {self.position_mapping!r}"
            )
        self.left_shoulder_pose0 = self.left_solver.frame_pose(
            self.initial_qpos,
            str(self.config.raw.get("left_shoulder_link_name", "left_arm_link1")),
        )
        self.right_shoulder_pose0 = self.left_solver.frame_pose(
            self.initial_qpos,
            str(self.config.raw.get("right_shoulder_link_name", "right_arm_link1")),
        )
        self.robot_shoulder_mid = 0.5 * (
            self.left_shoulder_pose0.translation + self.right_shoulder_pose0.translation
        )
        self.source_shoulder_offset = np.asarray(
            self.config.raw.get("source_shoulder_offset_head", SOURCE_SHOULDER_OFFSET_HEAD),
            dtype=np.float64,
        )
        self.source_shoulder_half_width = float(
            self.config.raw.get("source_shoulder_half_width", SOURCE_SHOULDER_HALF_WIDTH)
        )
        self.robot_arm_length = float(
            self.config.raw.get("robot_arm_length", self._estimate_robot_arm_length())
        )

    def retarget(
        self,
        left_sequence: HandFeatureSequence,
        right_sequence: HandFeatureSequence,
        segments: list[ActionSegment],
        max_frames: int | None = None,
        stride: int = 1,
    ) -> list[BimanualRetargetFrame]:
        source_frame_count = min(left_sequence.frame_count, right_sequence.frame_count)
        frame_count = (
            source_frame_count
            if max_frames is None
            else min(max_frames, source_frame_count)
        )
        left_reference = self.left_retargeter._first_valid_palm(left_sequence)
        right_reference = self.right_retargeter._first_valid_palm(right_sequence)
        left_rotation_reference = left_sequence.features[0].palm_rotation.copy()
        right_rotation_reference = right_sequence.features[0].palm_rotation.copy()
        source_scale = self._episode_source_to_robot_scale(
            left_sequence,
            right_sequence,
            frame_count,
        )
        self.last_source_scale = source_scale
        left_align = self._episode_tcp_align(
            self.left_tcp0.rotation,
            left_rotation_reference,
        )
        right_align = self._episode_tcp_align(
            self.right_tcp0.rotation,
            right_rotation_reference,
        )
        qpos = self.initial_qpos.copy()
        previous = qpos.copy()
        alpha = float(np.clip(self.config.low_pass_alpha, 0.0, 1.0))

        frames: list[BimanualRetargetFrame] = []
        for frame in range(0, frame_count, stride):
            left_features = left_sequence.features[frame]
            right_features = right_sequence.features[frame]
            left_position, left_rotation, left_mapped_wrist = self._target_pose(
                self.left_config,
                self.left_tcp0,
                self.left_tcp_offset_in_head,
                self.head_pose0,
                left_reference,
                left_rotation_reference,
                left_features,
                source_scale,
                left_align,
            )
            left_ik = self.left_solver.solve(
                left_position,
                left_rotation,
                previous,
            )
            right_position, right_rotation, right_mapped_wrist = self._target_pose(
                self.right_config,
                self.right_tcp0,
                self.right_tcp_offset_in_head,
                self.head_pose0,
                right_reference,
                right_rotation_reference,
                right_features,
                source_scale,
                right_align,
            )
            right_ik = self.right_solver.solve(
                right_position,
                right_rotation,
                left_ik.qpos,
            )
            raw_qpos = right_ik.qpos.copy()
            self._write_gripper_qpos(
                raw_qpos,
                self.left_retargeter,
                left_features,
                self._segment_for_frame(segments, frame),
            )
            self._write_gripper_qpos(
                raw_qpos,
                self.right_retargeter,
                right_features,
                self._segment_for_frame(segments, frame),
            )
            qpos = alpha * raw_qpos + (1.0 - alpha) * previous
            qpos = np.clip(qpos, self.left_solver.lower, self.left_solver.upper)
            previous = qpos

            segment = self._segment_for_frame(segments, frame)
            frames.append(
                BimanualRetargetFrame(
                    frame=frame,
                    timestamp=frame / self.config.fps,
                    qpos=qpos.astype(float).tolist(),
                    valid=bool(left_sequence.valid[frame] and right_sequence.valid[frame]),
                    left=self._feature_payload(
                        left_features,
                        bool(left_sequence.valid[frame]),
                        left_position,
                        left_mapped_wrist,
                        left_ik,
                    ),
                    right=self._feature_payload(
                        right_features,
                        bool(right_sequence.valid[frame]),
                        right_position,
                        right_mapped_wrist,
                        right_ik,
                    ),
                    action_id=segment.id if segment else None,
                    verbs=list(segment.verbs) if segment else [],
                )
            )
        return frames

    def _side_config(self, side: str) -> RetargetConfig:
        raw_joints = tuple(str(name) for name in self.config.raw.get("target_joint_names", []))
        arm_joint_names = tuple(
            name for name in raw_joints if name.startswith(f"{side}_arm_joint")
        )
        gripper_joint_names = tuple(
            name for name in raw_joints if name.startswith(f"{side}_gripper_")
        )
        if len(arm_joint_names) != 7:
            raise ValueError(f"Expected 7 {side} arm joints, got {len(arm_joint_names)}")
        if not gripper_joint_names:
            raise ValueError(f"Missing {side} gripper joints in target_joint_names")

        workspace = (self.config.raw.get("workspace") or {}).get(side, {})
        workspace_scale = tuple(float(v) for v in workspace.get("scale", self.config.workspace_scale))
        workspace_min = tuple(float(v) for v in workspace.get("min", self.config.workspace_min))
        workspace_max = tuple(float(v) for v in workspace.get("max", self.config.workspace_max))
        return replace(
            self.config,
            hand=side,
            arm_joint_names=arm_joint_names,
            gripper_joint_names=gripper_joint_names,
            tcp_link_name=f"{side}_gripper_tcp_link",
            workspace_scale=workspace_scale,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
        )

    def _initial_qpos(self) -> np.ndarray:
        qpos = self.left_solver.neutral_qpos()
        seed = self.config.raw.get("ik", {}).get("seed_qpos", {}) or {}
        for joint_name, value in seed.items():
            if joint_name not in self.name_to_q_index:
                raise ValueError(f"Seed joint is not in Pinocchio model: {joint_name}")
            qpos[self.name_to_q_index[joint_name]] = float(value)
        qpos = np.clip(qpos, self.left_solver.lower, self.left_solver.upper)
        return qpos

    def _make_solver(self, config: RetargetConfig) -> PinocchioIKSolver:
        ik_cfg = self.config.raw.get("ik", {}) or {}
        return PinocchioIKSolver(
            urdf_path=str(self.config.urdf_path),
            controlled_joint_names=config.arm_joint_names,
            tcp_link_name=config.tcp_link_name or f"{config.hand}_gripper_tcp_link",
            damping=float(ik_cfg.get("damping", 1e-3)),
            step_size=float(ik_cfg.get("step_size", 0.6)),
            max_iterations=int(ik_cfg.get("max_iterations", 40)),
            position_tolerance=float(ik_cfg.get("position_tolerance", 2e-3)),
            orientation_tolerance=float(ik_cfg.get("orientation_tolerance", 5e-2)),
            orientation_weight=float(ik_cfg.get("orientation_weight", 0.15)),
        )

    def _target_pose(
        self,
        config: RetargetConfig,
        tcp0,
        tcp_offset_in_head: np.ndarray,
        head_pose0,
        palm_reference: np.ndarray,
        palm_rotation_reference: np.ndarray,
        features,
        source_scale: float,
        tcp_align: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Position target is human palm center; mapped_* key kept for downstream compat.
        palm = features.palm_position
        if self.position_mapping == "shoulder_scaled":
            mapped_palm_in_head = self._target_in_head_shoulder_scaled(
                palm,
                source_scale,
            )
            target_in_head = mapped_palm_in_head
            target_rotation = head_pose0.rotation @ features.palm_rotation @ tcp_align
        else:
            workspace_min = np.asarray(config.workspace_min, dtype=np.float64)
            workspace_max = np.asarray(config.workspace_max, dtype=np.float64)
            if self.position_mapping == "absolute":
                mapped_palm_in_head = np.clip(
                    palm * np.asarray(config.workspace_scale),
                    workspace_min,
                    workspace_max,
                )
                target_in_head = mapped_palm_in_head
                target_rotation = head_pose0.rotation @ features.palm_rotation
            else:
                clipped_delta = np.clip(
                    (palm - palm_reference)
                    * np.asarray(config.workspace_scale),
                    workspace_min,
                    workspace_max,
                )
                mapped_palm_in_head = tcp_offset_in_head + clipped_delta
                target_in_head = mapped_palm_in_head
                relative_palm_rotation = palm_rotation_reference.T @ features.palm_rotation
                target_rotation = tcp0.rotation @ relative_palm_rotation

        target_position = head_pose0.translation + head_pose0.rotation @ target_in_head
        return target_position, target_rotation, mapped_palm_in_head

    def _source_shoulders_head(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mid = self.source_shoulder_offset
        left = mid + np.array([0.0, 0.0, self.source_shoulder_half_width])
        right = mid + np.array([0.0, 0.0, -self.source_shoulder_half_width])
        return left, right, mid

    def _target_in_head_shoulder_scaled(
        self,
        palm_head: np.ndarray,
        source_scale: float,
    ) -> np.ndarray:
        _, _, source_mid = self._source_shoulders_head()
        rel_head = np.asarray(palm_head, dtype=np.float64) - source_mid
        target_base = self.robot_shoulder_mid + self.head_pose0.rotation @ (
            rel_head * source_scale
        )
        target_in_head = self.head_pose0.rotation.T @ (
            target_base - self.head_pose0.translation
        )
        return target_in_head

    def _episode_source_to_robot_scale(
        self,
        left_sequence: HandFeatureSequence,
        right_sequence: HandFeatureSequence,
        frame_count: int,
    ) -> float:
        scale_override = self.config.raw.get("source_scale")
        if scale_override is not None:
            scale = float(scale_override)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"source_scale must be finite and positive, got {scale!r}")
            return scale

        left_sh, right_sh, _ = self._source_shoulders_head()
        left_dist = []
        right_dist = []
        for frame in range(frame_count):
            if left_sequence.valid[frame]:
                left_dist.append(
                    np.linalg.norm(left_sequence.features[frame].palm_position - left_sh)
                )
            if right_sequence.valid[frame]:
                right_dist.append(
                    np.linalg.norm(right_sequence.features[frame].palm_position - right_sh)
                )
        all_dists = np.asarray(left_dist + right_dist, dtype=np.float64)
        all_dists = all_dists[all_dists > MIN_ARM_LENGTH_SAMPLE]
        if all_dists.size == 0:
            return 1.0
        p95 = float(np.percentile(all_dists, ARM_LENGTH_PERCENTILE))
        return self.robot_arm_length / p95 if p95 > 1e-9 else 1.0

    def _estimate_robot_arm_length(self) -> float:
        left_len = np.linalg.norm(
            self.left_tcp0.translation - self.left_shoulder_pose0.translation
        )
        right_len = np.linalg.norm(
            self.right_tcp0.translation - self.right_shoulder_pose0.translation
        )
        return float(max(left_len, right_len))

    def _episode_tcp_align(
        self,
        tcp_rotation0: np.ndarray,
        palm_rotation_reference: np.ndarray,
    ) -> np.ndarray:
        return palm_rotation_reference.T @ self.head_pose0.rotation.T @ tcp_rotation0

    def _write_gripper_qpos(
        self,
        qpos: np.ndarray,
        side_retargeter: HeuristicGalbotRetargeter,
        features,
        segment: ActionSegment | None,
    ) -> None:
        master = side_retargeter._gripper_value(features, segment)
        for joint_name in side_retargeter.config.gripper_joint_names:
            if joint_name not in self.name_to_q_index:
                continue
            value = side_retargeter._mimic_adjusted_value(joint_name, master)
            value = side_retargeter._clip(joint_name, value)
            qpos[self.name_to_q_index[joint_name]] = value

    @staticmethod
    def _feature_payload(
        features,
        valid: bool,
        target_position,
        mapped_wrist_in_head,
        ik_result,
    ) -> dict[str, Any]:
        return {
            "valid": valid,
            "wrist_position": features.wrist_position.astype(float).tolist(),
            # Mapped palm center in head frame (key name kept for downstream compat).
            "mapped_wrist_in_head": np.asarray(mapped_wrist_in_head, dtype=float).tolist(),
            "palm_position": features.palm_position.astype(float).tolist(),
            "target_tcp_position": np.asarray(target_position, dtype=float).tolist(),
            "ik_position_error": float(ik_result.position_error),
            "ik_orientation_error": float(ik_result.orientation_error),
            "ik_iterations": int(ik_result.iterations),
            "ik_converged": bool(ik_result.converged),
            "pinch_score": float(features.pinch_score),
            "hand_openness": float(features.hand_openness),
            "gripper_closed": bool(features.gripper_closed),
            "gripper_command": int(features.gripper_closed),
            "finger_curl": {k: float(v) for k, v in features.finger_curl.items()},
            "dexterous_hand_pose": {
                f"{name}_curl": float(value)
                for name, value in features.finger_curl.items()
            },
        }

    @staticmethod
    def _segment_for_frame(
        segments: list[ActionSegment], frame: int
    ) -> ActionSegment | None:
        for segment in segments:
            if segment.contains(frame):
                return segment
        return None
