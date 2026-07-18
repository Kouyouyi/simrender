"""Replay RoboCOIN Galbot G1 states with a head-mounted MuJoCo camera."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from retarget_galbot.egoview.overlay import feather_alpha
from retarget_galbot.egoview.render import _set_action
from retarget_galbot.robots import RobotSpec, get_spec
from retarget_galbot.robots.mjcf_patch import EGO_CAMERA_NAME, patch_mjcf_local


ROBOCOIN_STATE_NAMES = (
    "torso_joint_1_rad",
    "torso_joint_2_rad",
    "torso_joint_3_rad",
    "head_joint_1_rad",
    "head_joint_2_rad",
    *(f"left_arm_joint_{index}_rad" for index in range(1, 8)),
    "left_gripper_open",
    *(f"right_arm_joint_{index}_rad" for index in range(1, 8)),
    "right_gripper_open",
)


@dataclass(frozen=True)
class RoboCoinEpisode:
    root: Path
    episode_index: int
    states: np.ndarray
    eef_sim_pose: np.ndarray
    timestamps: np.ndarray
    video_path: Path
    fps: float

    @classmethod
    def load(cls, root: str | Path, episode_index: int = 0) -> "RoboCoinEpisode":
        root = Path(root).expanduser().resolve()
        chunk = episode_index // 1000
        parquet_path = root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        video_path = (
            root
            / f"videos/chunk-{chunk:03d}/observation.images.cam_front_head_rgb"
            / f"episode_{episode_index:06d}.mp4"
        )
        info_path = root / "meta/info.json"
        if not parquet_path.exists():
            raise FileNotFoundError(parquet_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        if not info_path.exists():
            raise FileNotFoundError(info_path)

        info = json.loads(info_path.read_text(encoding="utf-8"))
        published_names = tuple(info["features"]["observation.state"]["names"])
        if published_names != ROBOCOIN_STATE_NAMES:
            raise ValueError(
                "RoboCOIN observation.state layout changed: "
                f"expected {ROBOCOIN_STATE_NAMES}, got {published_names}"
            )

        table = pd.read_parquet(parquet_path)
        states = np.stack(table["observation.state"].to_numpy()).astype(np.float64)
        eef = np.stack(table["eef_sim_pose_state"].to_numpy()).astype(np.float64)
        timestamps = table["timestamp"].to_numpy(dtype=np.float64)
        if states.shape[1:] != (21,):
            raise ValueError(f"Expected observation.state (T, 21), got {states.shape}")
        if eef.shape != (states.shape[0], 12):
            raise ValueError(f"Expected eef_sim_pose_state (T, 12), got {eef.shape}")

        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open {video_path}")
            video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if video_frames != states.shape[0]:
            raise ValueError(
                f"Video has {video_frames} frames but parquet has {states.shape[0]} rows"
            )
        return cls(
            root=root,
            episode_index=episode_index,
            states=states,
            eef_sim_pose=eef,
            timestamps=timestamps,
            video_path=video_path,
            fps=float(info["fps"]),
        )

    @property
    def frame_count(self) -> int:
        return int(self.states.shape[0])


def _gripper_mimic_values(master: np.ndarray) -> np.ndarray:
    """Expand the published master angle to the six URDF gripper joints."""
    master = np.asarray(master, dtype=np.float64)
    # Joint order: master, r_inner, r_finger, l_knuckle, l_inner, l_finger.
    return np.stack((master, -master, master, master, master, -master), axis=-1)


def expand_observation_states(states: np.ndarray) -> np.ndarray:
    """Map RoboCOIN's 21-D state to this project's 33-D Galbot qpos layout."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 21:
        raise ValueError(f"Expected states (T, 21), got {states.shape}")

    actions = np.zeros((states.shape[0], 33), dtype=np.float64)
    # RoboCOIN's eef_sim_pose was generated with the Galbot leg chain at zero.
    # The three published torso SDK readings are not the five URDF leg angles.
    actions[:, 5:12] = states[:, 5:12]
    actions[:, 12:18] = _gripper_mimic_values(states[:, 12])
    actions[:, 18:25] = states[:, 13:20]
    actions[:, 25:31] = _gripper_mimic_values(states[:, 20])
    actions[:, 31:33] = states[:, 3:5]
    return actions


@dataclass(frozen=True)
class RoboCoinHeadCamera:
    """Fixed camera transform calibrated against episode-0 arm joint centres."""

    head_body: str
    offset_in_head: np.ndarray
    delta_rotvec_opencv: np.ndarray
    vertical_fov_degrees: float
    calibration_width: int = 640
    calibration_height: int = 480
    metadata: dict[str, Any] | None = None

    @classmethod
    def load(cls, path: str | Path) -> "RoboCoinHeadCamera":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            head_body=str(raw["head_body"]),
            offset_in_head=np.asarray(raw["offset_in_head_m"], dtype=np.float64),
            delta_rotvec_opencv=np.asarray(
                raw["delta_rotvec_opencv_rad"], dtype=np.float64
            ),
            vertical_fov_degrees=float(raw["vertical_fov_degrees"]),
            calibration_width=int(raw.get("calibration_width", 640)),
            calibration_height=int(raw.get("calibration_height", 480)),
            metadata=raw,
        )

    def mujoco_world_pose(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> tuple[np.ndarray, np.ndarray]:
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.head_body
        )
        if body_id < 0:
            raise KeyError(f"Model has no camera parent body {self.head_body!r}")
        head_position = data.xpos[body_id]
        head_rotation = data.xmat[body_id].reshape(3, 3)

        # OpenCV axes expressed in head_link2: right=-z, down=+y, forward=+x.
        head_from_opencv = np.asarray(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        world_from_opencv = (
            head_rotation
            @ head_from_opencv
            @ Rotation.from_rotvec(self.delta_rotvec_opencv).as_matrix()
        )
        # MuJoCo camera axes are right, up, back instead of right, down, forward.
        world_from_mujoco = world_from_opencv @ np.diag([1.0, -1.0, -1.0])
        camera_position = head_position + head_rotation @ self.offset_in_head
        quat_xyzw = Rotation.from_matrix(world_from_mujoco).as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        return camera_position, quat_wxyz


def validate_eef_forward_kinematics(
    actions: np.ndarray,
    expected_eef_pose: np.ndarray,
    *,
    spec: RobotSpec | None = None,
) -> dict[str, float]:
    """Prove that the expanded state reproduces RoboCOIN's simulated EEF pose."""
    if spec is None:
        spec = get_spec("galbot")
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    data = mujoco.MjData(model)
    writer = spec.qpos_writer_factory(model)
    setattr(writer, "_spec", spec)
    body_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_arm_link7"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_arm_link7"),
    ]

    position_errors: list[float] = []
    rotation_errors: list[float] = []
    for frame_index, action in enumerate(np.asarray(actions)):
        _set_action(data, writer, action)
        for side, body_id in enumerate(body_ids):
            expected_offset = 0 if side == 0 else 6
            actual_position = data.xpos[body_id]
            expected_position = expected_eef_pose[
                frame_index, expected_offset : expected_offset + 3
            ]
            position_errors.append(float(np.linalg.norm(actual_position - expected_position)))

            actual_rotation = data.xmat[body_id].reshape(3, 3)
            expected_rotation = Rotation.from_euler(
                "xyz",
                expected_eef_pose[
                    frame_index, expected_offset + 3 : expected_offset + 6
                ],
            ).as_matrix()
            delta = expected_rotation.T @ actual_rotation
            rotation_errors.append(float(Rotation.from_matrix(delta).magnitude()))

    position = np.asarray(position_errors)
    rotation_deg = np.degrees(np.asarray(rotation_errors))
    return {
        "position_mean_m": float(position.mean()),
        "position_max_m": float(position.max()),
        "rotation_mean_deg": float(rotation_deg.mean()),
        "rotation_max_deg": float(rotation_deg.max()),
    }


class RoboCoinOverlayRenderer:
    """Render only Galbot arms/grippers from the calibrated head camera."""

    def __init__(
        self,
        camera: RoboCoinHeadCamera,
        *,
        width: int = 640,
        height: int = 480,
        style: str = "realistic",
        spec: RobotSpec | None = None,
    ) -> None:
        if style not in {"realistic", "debug"}:
            raise ValueError(f"Unsupported render style {style!r}")
        self.camera = camera
        self.width = int(width)
        self.height = int(height)
        self.style = style
        self.spec = spec or get_spec("galbot")
        patcher = self.spec.patch_mjcf or patch_mjcf_local
        self._patched_xml = Path(
            patcher(self.spec.mjcf_path, camera.vertical_fov_degrees)
        )
        self.model = mujoco.MjModel.from_xml_path(str(self._patched_xml))
        self.data = mujoco.MjData(self.model)
        self.writer = self.spec.qpos_writer_factory(self.model)
        setattr(self.writer, "_spec", self.spec)
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, EGO_CAMERA_NAME
        )

        self.render_geom_ids: list[int] = []
        hidden_geom_ids: list[int] = []
        for geom_id in range(self.model.ngeom):
            body_name = (
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(self.model.geom_bodyid[geom_id]),
                )
                or ""
            )
            if "arm_link" in body_name or "gripper" in body_name:
                self.render_geom_ids.append(geom_id)
                self._set_geom_color(geom_id, body_name)
            else:
                hidden_geom_ids.append(geom_id)
        self.model.geom_rgba[hidden_geom_ids, 3] = 0.0
        self._render_geom_ids_array = np.asarray(self.render_geom_ids, dtype=np.int32)

        self.renderer_rgb = mujoco.Renderer(
            self.model, height=self.height, width=self.width
        )
        self.renderer_seg = mujoco.Renderer(
            self.model, height=self.height, width=self.width
        )
        self.renderer_seg.enable_segmentation_rendering()

    def _set_geom_color(self, geom_id: int, body_name: str) -> None:
        if self.style == "debug":
            self.model.geom_rgba[geom_id] = (
                [0.05, 0.85, 1.0, 1.0]
                if body_name.startswith("left")
                else [1.0, 0.35, 0.05, 1.0]
            )
            return

        is_dark = "gripper" in body_name
        mesh_id = int(self.model.geom_dataid[geom_id])
        if mesh_id >= 0:
            mesh_name = (
                mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_MESH, mesh_id
                )
                or ""
            )
            is_dark = is_dark or "flange" in mesh_name or "base_link" in mesh_name
        self.model.geom_rgba[geom_id] = (
            [0.055, 0.065, 0.075, 1.0]
            if is_dark
            else [0.78, 0.82, 0.86, 1.0]
        )

    def render(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _set_action(self.data, self.writer, action)
        position, quaternion = self.camera.mujoco_world_pose(self.model, self.data)
        self.model.cam_pos[self.camera_id] = position
        self.model.cam_quat[self.camera_id] = quaternion
        self.model.cam_fovy[self.camera_id] = self.camera.vertical_fov_degrees
        mujoco.mj_forward(self.model, self.data)

        self.renderer_rgb.update_scene(self.data, camera=EGO_CAMERA_NAME)
        robot_rgb = self.renderer_rgb.render()
        self.renderer_seg.update_scene(self.data, camera=EGO_CAMERA_NAME)
        segmentation = self.renderer_seg.render()
        mask = np.isin(segmentation[..., 0], self._render_geom_ids_array)
        return robot_rgb, mask

    def close(self) -> None:
        self.renderer_rgb.close()
        self.renderer_seg.close()
        for candidate in (self._patched_xml.parent, self._patched_xml.parent.parent):
            if str(candidate).startswith(tempfile.gettempdir()):
                shutil.rmtree(candidate, ignore_errors=True)
                break

    def __enter__(self) -> "RoboCoinOverlayRenderer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def composite_direct_overlay(
    scene_rgb: np.ndarray,
    robot_rgb: np.ndarray,
    robot_mask: np.ndarray,
    *,
    opacity: float = 0.82,
    outline: bool = False,
) -> np.ndarray:
    opacity = float(np.clip(opacity, 0.0, 1.0))
    alpha = feather_alpha(robot_mask, sigma=1.2) * opacity
    output = (
        scene_rgb.astype(np.float32) * (1.0 - alpha[..., None])
        + robot_rgb.astype(np.float32) * alpha[..., None]
    )
    output = np.clip(output, 0, 255).astype(np.uint8)
    if outline:
        contours, _ = cv2.findContours(
            robot_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    return output


class H264StreamWriter:
    """Stream RGB frames to the project's compatible H.264 MP4 format."""

    def __init__(self, path: str | Path, fps: float) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            str(self.path),
            format="FFMPEG",
            mode="I",
            fps=float(fps),
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
            output_params=[
                "-preset",
                "medium",
                "-crf",
                "18",
                "-profile:v",
                "high",
                "-level:v",
                "4.0",
                "-tag:v",
                "avc1",
                "-movflags",
                "+faststart",
            ],
        )

    def append(self, frame_rgb: np.ndarray) -> None:
        self.writer.append_data(np.asarray(frame_rgb, dtype=np.uint8))

    def close(self) -> None:
        self.writer.close()

    def __enter__(self) -> "H264StreamWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
