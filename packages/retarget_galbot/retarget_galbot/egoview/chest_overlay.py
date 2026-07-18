"""Stream Galbot arm overlays from a chest-mounted camera into an ego video."""

from __future__ import annotations

import json
import logging
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from retarget_galbot.egoview.overlay import overlay_robot
from retarget_galbot.galaxea.coordinates import CAMERA_TO_HEAD_LINK
from retarget_galbot.robots import RobotSpec
from retarget_galbot.robots.mjcf_patch import EGO_CAMERA_NAME, patch_mjcf_local

logger = logging.getLogger(__name__)

# OpenCV camera: +x right, +y down, +z forward.
# MuJoCo camera: +x right, +y up, and looks along -z.
_OPENCV_FROM_MUJOCO_CAMERA = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class OverlayVideoStats:
    frame_count: int
    width: int
    height: int
    fps: float
    fovy_degrees: float
    first_camera_position: tuple[float, float, float]


def chest_camera_pose_from_shoulders(
    shoulder_mid_base: np.ndarray,
    source_shoulder_mid_head: np.ndarray,
    head_rotation_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return MuJoCo camera position/rotation using the retarget shoulder prior.

    ``source_shoulder_mid_head`` stores the human shoulder midpoint relative to
    the chest camera. Inverting that relation at the robot shoulder midpoint
    places the simulated camera with the same geometric definition.
    """
    shoulder_mid = np.asarray(shoulder_mid_base, dtype=np.float64).reshape(3)
    source_offset = np.asarray(source_shoulder_mid_head, dtype=np.float64).reshape(3)
    rotation_base_head = np.asarray(head_rotation_base, dtype=np.float64).reshape(3, 3)

    camera_position = shoulder_mid - rotation_base_head @ source_offset
    rotation_base_opencv = rotation_base_head @ CAMERA_TO_HEAD_LINK
    rotation_base_mujoco = rotation_base_opencv @ _OPENCV_FROM_MUJOCO_CAMERA
    return camera_position, rotation_base_mujoco


def calibrated_vertical_fov(
    episode_dir: str | Path,
    target_width: int,
    target_height: int,
) -> tuple[float, float, float]:
    """Return (fovy_degrees, cx_scaled, cy_scaled) from video calibration."""
    info_path = (
        Path(episode_dir)
        / "ego_process"
        / "ego_undistorted_video"
        / "undistorted_video_info.json"
    )
    if not info_path.exists():
        raise FileNotFoundError(f"Missing camera calibration: {info_path}")

    info = json.loads(info_path.read_text())
    params = info.get("cameraParams", {})
    resolution = str(params.get("resolution", ""))
    if "x" not in resolution:
        raise ValueError(f"Invalid camera resolution in {info_path}: {resolution!r}")
    original_width, original_height = (int(value) for value in resolution.split("x", 1))
    fy = float(params["fy_pixels"])
    cx = float(params["cx_pixels"])
    cy = float(params["cy_pixels"])

    calibration_width = float(original_width)
    calibration_height = float(original_height)
    principal_point_mismatch = (
        abs(cx - original_width / 2.0) > original_width * 0.1
        or abs(cy - original_height / 2.0) > original_height * 0.1
    )
    inferred_width = 2.0 * cx
    inferred_height = 2.0 * cy
    target_aspect = float(target_width) / float(target_height)
    inferred_aspect = inferred_width / inferred_height
    if principal_point_mismatch and abs(inferred_aspect / target_aspect - 1.0) < 0.02:
        calibration_width = inferred_width
        calibration_height = inferred_height
        logger.warning(
            "Camera K appears calibrated at %.1fx%.1f while metadata says %dx%d; "
            "rescaling K to %dx%d",
            calibration_width,
            calibration_height,
            original_width,
            original_height,
            target_width,
            target_height,
        )

    scale_x = float(target_width) / calibration_width
    scale_y = float(target_height) / calibration_height
    fy_scaled = fy * scale_y
    fovy = math.degrees(2.0 * math.atan(float(target_height) / (2.0 * fy_scaled)))
    return fovy, cx * scale_x, cy * scale_y


def legacy_camera_pose(
    camera_position: np.ndarray,
    wrist_midpoint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the original ego renderer camera that looks at both wrists."""
    position = np.asarray(camera_position, dtype=np.float64).reshape(3)
    forward = np.asarray(wrist_midpoint, dtype=np.float64).reshape(3) - position
    forward_norm = np.linalg.norm(forward)
    forward = forward / forward_norm if forward_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right_norm = np.linalg.norm(right)
    right = right / right_norm if right_norm > 1e-6 else np.array([0.0, 1.0, 0.0])
    up = np.cross(right, forward)
    return position, np.column_stack([right, up, -forward])


def _arm_geom_ids(model: mujoco.MjModel, spec: RobotSpec) -> np.ndarray:
    body_ids = set()
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if spec.hand_body_predicate(name):
            body_ids.add(body_id)
    geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in body_ids
    ]
    if not geom_ids:
        raise RuntimeError("No arm/gripper geoms matched the robot body predicate")
    return np.asarray(geom_ids, dtype=np.int32)


def _set_world_camera_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_id: int,
    position: np.ndarray,
    rotation: np.ndarray,
) -> None:
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )
    model.cam_pos[camera_id] = position
    model.cam_quat[camera_id] = quat_wxyz
    data.cam_xpos[camera_id] = position
    data.cam_xmat[camera_id] = rotation.reshape(-1)


def _read_source_frame(
    capture: cv2.VideoCapture,
    source_frame: int,
    previous_source_frame: int | None,
) -> np.ndarray:
    if previous_source_frame is None or source_frame != previous_source_frame + 1:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_frame))
    ok, frame_bgr = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read source video frame {source_frame}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def render_chest_overlay_video(
    actions: np.ndarray,
    source_frame_indices: Iterable[int],
    ego_video_path: str | Path,
    output_path: str | Path,
    *,
    spec: RobotSpec,
    source_shoulder_mid_head: np.ndarray,
    head_rotation_base: np.ndarray,
    fps: float,
    feather_sigma: float = 1.0,
    camera_mode: Literal["chest", "legacy"] = "chest",
) -> OverlayVideoStats:
    """Render Galbot arms and alpha-blend them directly over original frames."""
    actions = np.asarray(actions, dtype=np.float64)
    source_frames = np.asarray(list(source_frame_indices), dtype=np.int64)
    if actions.ndim != 2 or actions.shape[1] != spec.action_dim:
        raise ValueError(f"Expected actions (T,{spec.action_dim}), got {actions.shape}")
    if source_frames.shape != (actions.shape[0],):
        raise ValueError("source_frame_indices must have one entry per action")
    if actions.shape[0] == 0:
        raise ValueError("No actions to render")

    video_path = Path(ego_video_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open ego video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_frames.min() < 0 or source_frames.max() >= frame_count:
        capture.release()
        raise ValueError(
            f"Source frame range [{source_frames.min()}, {source_frames.max()}] "
            f"is outside video frame count {frame_count}"
        )

    if camera_mode == "legacy":
        fovy, cx, cy = 70.0, width / 2.0, height / 2.0
    elif camera_mode == "chest":
        fovy, cx, cy = calibrated_vertical_fov(video_path.parents[2], width, height)
    else:
        raise ValueError(f"Unknown camera_mode: {camera_mode!r}")
    if abs(cx - width / 2.0) > 1.5 or abs(cy - height / 2.0) > 1.5:
        logger.warning(
            "MuJoCo uses a centered principal point, but calibration is (%.2f, %.2f) "
            "for %dx%d",
            cx,
            cy,
            width,
            height,
        )

    patcher = spec.patch_mjcf or patch_mjcf_local
    patched_xml = Path(patcher(spec.mjcf_path, fovy))
    model = mujoco.MjModel.from_xml_path(str(patched_xml))
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    qpos_writer = spec.qpos_writer_factory(model)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, EGO_CAMERA_NAME)
    left_shoulder_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, spec.left_shoulder_body
    )
    right_shoulder_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, spec.right_shoulder_body
    )
    left_wrist_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, spec.left_wrist_body
    )
    right_wrist_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, spec.right_wrist_body
    )
    arm_geom_ids = _arm_geom_ids(model, spec)
    arm_geom_mask = np.zeros(model.ngeom, dtype=bool)
    arm_geom_mask[arm_geom_ids] = True
    model.geom_rgba[~arm_geom_mask, 3] = 0.0

    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    legacy_camera_position = (
        0.5 * (data.xpos[left_shoulder_id] + data.xpos[right_shoulder_id])
        + np.array([0.12, 0.0, 0.12])
    )

    renderer_rgb = mujoco.Renderer(model, height=height, width=width)
    renderer_seg = mujoco.Renderer(model, height=height, width=width)
    renderer_seg.enable_segmentation_rendering()
    writer = imageio.get_writer(
        str(output),
        format="FFMPEG",
        mode="I",
        fps=float(fps),
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=1,
        output_params=[
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

    first_camera_position = None
    previous_source_frame = None
    try:
        for index, (action, source_frame) in enumerate(zip(actions, source_frames)):
            frame_rgb = _read_source_frame(
                capture,
                int(source_frame),
                previous_source_frame,
            )
            previous_source_frame = int(source_frame)

            data.qpos[:] = 0.0
            qpos_writer(data.qpos, action)
            mujoco.mj_forward(model, data)
            if camera_mode == "legacy":
                wrist_midpoint = 0.5 * (
                    data.xpos[left_wrist_id] + data.xpos[right_wrist_id]
                )
                camera_position, camera_rotation = legacy_camera_pose(
                    legacy_camera_position,
                    wrist_midpoint,
                )
            else:
                shoulder_mid = 0.5 * (
                    data.xpos[left_shoulder_id] + data.xpos[right_shoulder_id]
                )
                camera_position, camera_rotation = chest_camera_pose_from_shoulders(
                    shoulder_mid,
                    source_shoulder_mid_head,
                    head_rotation_base,
                )
            if first_camera_position is None:
                first_camera_position = camera_position.copy()
            _set_world_camera_pose(
                model,
                data,
                camera_id,
                camera_position,
                camera_rotation,
            )

            renderer_rgb.update_scene(data, camera=EGO_CAMERA_NAME)
            robot_rgb = renderer_rgb.render()
            renderer_seg.update_scene(data, camera=EGO_CAMERA_NAME)
            segmentation = renderer_seg.render()
            robot_mask = np.isin(segmentation[..., 0], arm_geom_ids)

            composed = overlay_robot(
                scene_rgb=frame_rgb,
                robot_rgb=robot_rgb,
                robot_mask=robot_mask,
                feather_sigma=feather_sigma,
                harmonize=False,
                shadow=False,
                defringe=True,
                extend_bleed=True,
            )
            writer.append_data(composed)

            if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(actions):
                logger.info("Rendered %d/%d frames", index + 1, len(actions))
    finally:
        writer.close()
        renderer_rgb.close()
        renderer_seg.close()
        capture.release()
        if str(patched_xml.parent).startswith(tempfile.gettempdir()):
            shutil.rmtree(patched_xml.parent, ignore_errors=True)

    assert first_camera_position is not None
    return OverlayVideoStats(
        frame_count=int(actions.shape[0]),
        width=width,
        height=height,
        fps=float(fps),
        fovy_degrees=float(fovy),
        first_camera_position=tuple(float(value) for value in first_camera_position),
    )
