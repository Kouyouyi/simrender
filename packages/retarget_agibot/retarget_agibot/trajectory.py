"""Load and validate frame-aligned AgiBot proprioception trajectories."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


LEFT_ARM_JOINTS = tuple(f"idx{20 + i}_arm_l_joint{i}" for i in range(1, 8))
RIGHT_ARM_JOINTS = tuple(f"idx{60 + i}_arm_r_joint{i}" for i in range(1, 8))
WAIST_JOINTS = ("idx01_body_joint1", "idx02_body_joint2")
HEAD_JOINTS = ("idx11_head_joint1", "idx12_head_joint2")
CORE_JOINTS = WAIST_JOINTS + HEAD_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS


@dataclass(frozen=True)
class AgibotTrajectory:
    """The state and action signals needed for kinematic G1 playback."""

    timestamps_ns: np.ndarray
    state_arm: np.ndarray
    action_arm: np.ndarray
    state_waist: np.ndarray
    action_waist: np.ndarray
    state_head: np.ndarray
    action_head: np.ndarray
    state_gripper_closed_fraction: np.ndarray
    action_gripper_closed_fraction: np.ndarray
    base_position: np.ndarray
    base_orientation_xyzw: np.ndarray
    end_position: np.ndarray
    end_orientation_xyzw: np.ndarray
    source_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "AgibotTrajectory":
        path = Path(path).expanduser().resolve()
        required = (
            "timestamp",
            "state/joint/position",
            "action/joint/position",
            "state/waist/position",
            "action/waist/position",
            "state/head/position",
            "action/head/position",
            "state/effector/position",
            "action/effector/position",
            "state/robot/position",
            "state/robot/orientation",
            "state/end/position",
            "state/end/orientation",
        )
        with h5py.File(path, "r") as handle:
            missing = [key for key in required if key not in handle]
            if missing:
                raise KeyError(f"Missing trajectory datasets in {path}: {missing}")
            arrays = {key: np.asarray(handle[key]) for key in required}

        timestamps = arrays["timestamp"].astype(np.int64, copy=False)
        n = int(timestamps.shape[0])
        expected_shapes = {
            "state/joint/position": (n, 14),
            "action/joint/position": (n, 14),
            "state/waist/position": (n, 2),
            "action/waist/position": (n, 2),
            "state/head/position": (n, 2),
            "action/head/position": (n, 2),
            "state/effector/position": (n, 2),
            "action/effector/position": (n, 2),
            "state/robot/position": (n, 3),
            "state/robot/orientation": (n, 4),
            "state/end/position": (n, 2, 3),
            "state/end/orientation": (n, 2, 4),
        }
        for key, shape in expected_shapes.items():
            if arrays[key].shape != shape:
                raise ValueError(f"{key} has shape {arrays[key].shape}, expected {shape}")

        quaternion_norms = np.linalg.norm(arrays["state/robot/orientation"], axis=1)
        if np.any(quaternion_norms < 1e-8):
            bad = np.flatnonzero(quaternion_norms < 1e-8)[:5].tolist()
            raise ValueError(f"Zero-length base quaternion at frames {bad}")

        # State values are measured angles (roughly 35=open and 112=closed),
        # while actions are already normalized as 0=open and 1=closed.
        state_effector = arrays["state/effector/position"].astype(np.float64)
        state_open = np.percentile(state_effector, 1.0, axis=0)
        state_closed = np.percentile(state_effector, 99.0, axis=0)
        state_span = state_closed - state_open
        if np.any(state_span < 1e-6):
            raise ValueError(
                "state/effector/position does not span distinct open/closed values"
            )
        state_gripper = np.clip(
            (state_effector - state_open) / state_span,
            0.0,
            1.0,
        )
        action_gripper = np.clip(arrays["action/effector/position"], 0.0, 1.0)
        return cls(
            timestamps_ns=timestamps,
            state_arm=arrays["state/joint/position"].astype(np.float64),
            action_arm=arrays["action/joint/position"].astype(np.float64),
            state_waist=arrays["state/waist/position"].astype(np.float64),
            action_waist=arrays["action/waist/position"].astype(np.float64),
            state_head=arrays["state/head/position"].astype(np.float64),
            action_head=arrays["action/head/position"].astype(np.float64),
            state_gripper_closed_fraction=state_gripper.astype(np.float64),
            action_gripper_closed_fraction=action_gripper.astype(np.float64),
            base_position=arrays["state/robot/position"].astype(np.float64),
            base_orientation_xyzw=arrays["state/robot/orientation"].astype(np.float64),
            end_position=arrays["state/end/position"].astype(np.float64),
            end_orientation_xyzw=arrays["state/end/orientation"].astype(np.float64),
            source_path=path,
        )

    @property
    def frame_count(self) -> int:
        return int(self.timestamps_ns.shape[0])

    @property
    def fps(self) -> float:
        if self.frame_count < 2:
            return 30.0
        delta_ns = np.diff(self.timestamps_ns)
        positive = delta_ns[delta_ns > 0]
        if positive.size == 0:
            return 30.0
        return float(1e9 / np.median(positive))

    @property
    def duration_seconds(self) -> float:
        if self.frame_count < 2:
            return 0.0
        return float((self.timestamps_ns[-1] - self.timestamps_ns[0]) / 1e9)

    @property
    def gripper_closed_fraction(self) -> np.ndarray:
        """Backward-compatible action gripper values."""
        return self.action_gripper_closed_fraction

    def effector_closed_fraction(self, frame_index: int, source: str) -> np.ndarray:
        if source == "state":
            return self.state_gripper_closed_fraction[frame_index]
        if source == "action":
            return self.action_gripper_closed_fraction[frame_index]
        raise ValueError(f"source must be 'state' or 'action', got {source!r}")

    def joint_values(self, frame_index: int, source: str = "state") -> dict[str, float]:
        if not 0 <= frame_index < self.frame_count:
            raise IndexError(frame_index)
        if source not in {"state", "action"}:
            raise ValueError(f"source must be 'state' or 'action', got {source!r}")
        arm = self.state_arm if source == "state" else self.action_arm
        waist = self.state_waist if source == "state" else self.action_waist
        head = self.state_head if source == "state" else self.action_head
        names = WAIST_JOINTS + HEAD_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
        values = np.concatenate((waist[frame_index], head[frame_index], arm[frame_index]))
        return dict(zip(names, values.astype(float), strict=True))

    def limit_report(self, urdf_path: str | Path, source: str = "state") -> dict[str, Any]:
        """Compare every recorded core joint value against its URDF limits."""
        root = ET.parse(Path(urdf_path)).getroot()
        limits: dict[str, tuple[float, float]] = {}
        for joint in root.findall("joint"):
            limit = joint.find("limit")
            if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
                continue
            limits[joint.attrib["name"]] = (
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )

        rows: dict[str, dict[str, float | int]] = {}
        value_rows = [self.joint_values(i, source) for i in range(self.frame_count)]
        for name in CORE_JOINTS:
            if name not in limits:
                raise KeyError(f"URDF has no finite limit for controlled joint {name}")
            values = np.asarray([row[name] for row in value_rows])
            lower, upper = limits[name]
            under = np.maximum(lower - values, 0.0)
            over = np.maximum(values - upper, 0.0)
            violation = np.maximum(under, over)
            rows[name] = {
                "lower": lower,
                "upper": upper,
                "observed_min": float(values.min()),
                "observed_max": float(values.max()),
                "violation_count": int(np.count_nonzero(violation > 0.0)),
                "max_violation": float(violation.max()),
            }
        return {
            "trajectory": str(self.source_path),
            "source": source,
            "frame_count": self.frame_count,
            "fps_from_timestamps": self.fps,
            "duration_seconds": self.duration_seconds,
            "total_violation_count": int(
                sum(int(row["violation_count"]) for row in rows.values())
            ),
            "joints": rows,
        }

    def write_limit_report(
        self, output_path: str | Path, urdf_path: str | Path, source: str = "state"
    ) -> dict[str, Any]:
        report = self.limit_report(urdf_path, source)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
