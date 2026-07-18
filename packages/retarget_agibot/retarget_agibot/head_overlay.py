"""Render the measured G1 trajectory through the recorded head camera."""

from __future__ import annotations

import json
import logging
import math
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .controller import AgibotG1Controller
from .depth_color import (
    draw_vision_banana_ruler,
    focused_depth_to_rgb,
)
from .proxy_model import build_proxy_mjcf
from .trajectory import AgibotTrajectory
from .video import H264Writer, IndexedVideoReader


LOGGER = logging.getLogger(__name__)
HEAD_CAMERA_NAME = "agibot_recorded_head"
_OPENCV_TO_MUJOCO_CAMERA = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class HeadCameraCalibration:
    """Camera model and per-frame camera-to-data-frame poses."""

    width: int
    height: int
    fps: float
    intrinsic: np.ndarray
    distortion: np.ndarray
    rotations_data_from_camera: np.ndarray
    translations_data_from_camera: np.ndarray
    intrinsic_path: Path
    extrinsic_path: Path

    @property
    def native_render_height(self) -> int:
        """Square-pixel height before the dataset's 16:9-to-4:3 stretch."""
        fx = float(self.intrinsic[0, 0])
        fy = float(self.intrinsic[1, 1])
        return max(2, int(round(self.height * fx / fy)))

    @property
    def native_fovy_degrees(self) -> float:
        native_height = self.native_render_height
        fy_native = float(self.intrinsic[1, 1]) * native_height / self.height
        return math.degrees(2.0 * math.atan(native_height / (2.0 * fy_native)))


@dataclass(frozen=True)
class ModelFrameAlignment:
    """Rigid transform from MuJoCo fixed-base coordinates to dataset coordinates."""

    rotation_data_from_model: np.ndarray
    translation_data_from_model: np.ndarray
    rmse_m: float
    sample_count: int


@dataclass(frozen=True)
class HeadOverlayStats:
    frame_count: int
    fps: float
    width: int
    height: int
    native_render_height: int
    alignment_rmse_m: float
    first_camera_position_model: tuple[float, float, float]
    runtime_clip_counts: dict[str, int]


def load_head_camera_calibration(
    dataset_dir: str | Path,
    *,
    expected_frame_count: int | None = None,
) -> HeadCameraCalibration:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    parameter_dir = dataset_dir / "parameters"
    intrinsic_path = parameter_dir / "head_intrinsic_params.json"
    extrinsic_path = parameter_dir / "head_extrinsic_params_aligned.json"
    camera_info_path = parameter_dir / "rs_camera_info.json"

    intrinsic_raw = json.loads(intrinsic_path.read_text(encoding="utf-8"))["intrinsic"]
    camera_info = json.loads(camera_info_path.read_text(encoding="utf-8"))["d455_1"]
    extrinsic_raw = json.loads(extrinsic_path.read_text(encoding="utf-8"))

    intrinsic = np.asarray(
        [
            [float(intrinsic_raw["fx"]), 0.0, float(intrinsic_raw["ppx"])],
            [0.0, float(intrinsic_raw["fy"]), float(intrinsic_raw["ppy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(
        [
            intrinsic_raw["k1"],
            intrinsic_raw["k2"],
            intrinsic_raw["p1"],
            intrinsic_raw["p2"],
            intrinsic_raw["k3"],
        ],
        dtype=np.float64,
    )
    rotations = np.asarray(
        [item["extrinsic"]["rotation_matrix"] for item in extrinsic_raw],
        dtype=np.float64,
    )
    translations = np.asarray(
        [item["extrinsic"]["translation_vector"] for item in extrinsic_raw],
        dtype=np.float64,
    )
    if rotations.shape != (len(extrinsic_raw), 3, 3):
        raise ValueError(f"Invalid aligned head-camera rotations: {rotations.shape}")
    if translations.shape != (len(extrinsic_raw), 3):
        raise ValueError(f"Invalid aligned head-camera translations: {translations.shape}")
    if expected_frame_count is not None and len(extrinsic_raw) != expected_frame_count:
        raise ValueError(
            f"Camera poses have {len(extrinsic_raw)} frames, expected {expected_frame_count}"
        )
    determinants = np.linalg.det(rotations)
    if not np.allclose(determinants, 1.0, atol=1e-5):
        raise ValueError("Aligned camera rotations are not proper rotation matrices")

    return HeadCameraCalibration(
        width=int(camera_info["width"]),
        height=int(camera_info["height"]),
        fps=float(camera_info["fps"]),
        intrinsic=intrinsic,
        distortion=distortion,
        rotations_data_from_camera=rotations,
        translations_data_from_camera=translations,
        intrinsic_path=intrinsic_path,
        extrinsic_path=extrinsic_path,
    )


def fit_model_frame_alignment(
    model: mujoco.MjModel,
    controller: AgibotG1Controller,
    trajectory: AgibotTrajectory,
    *,
    sample_stride: int = 25,
    joint_source: str = "state",
) -> ModelFrameAlignment:
    """Align MuJoCo FK to the frame used by the recorded end/camera poses."""
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    left_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "arm_l_end_link"
    )
    right_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "arm_r_end_link"
    )
    if left_id < 0 or right_id < 0:
        raise KeyError("Model must contain arm_l_end_link and arm_r_end_link")

    frame_indices = list(range(0, trajectory.frame_count, sample_stride))
    if frame_indices[-1] != trajectory.frame_count - 1:
        frame_indices.append(trajectory.frame_count - 1)
    model_points: list[np.ndarray] = []
    data_points: list[np.ndarray] = []
    for frame_index in frame_indices:
        data = controller.set_frame(
            trajectory,
            frame_index,
            joint_source=joint_source,
            base_mode="fixed",
        )
        model_points.extend((data.xpos[left_id].copy(), data.xpos[right_id].copy()))
        data_points.extend(
            (
                trajectory.end_position[frame_index, 0],
                trajectory.end_position[frame_index, 1],
            )
        )

    source = np.asarray(model_points, dtype=np.float64)
    target = np.asarray(data_points, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_mean - rotation @ source_mean
    aligned = (rotation @ source.T).T + translation
    rmse = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    if rmse > 0.005:
        raise RuntimeError(
            f"MuJoCo/data frame alignment RMSE is unexpectedly high: {rmse:.6f} m"
        )
    return ModelFrameAlignment(
        rotation_data_from_model=rotation,
        translation_data_from_model=translation,
        rmse_m=rmse,
        sample_count=int(source.shape[0]),
    )


def camera_pose_in_model_frame(
    calibration: HeadCameraCalibration,
    alignment: ModelFrameAlignment,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return OpenCV camera position and orientation in MuJoCo model coordinates."""
    rotation_data_camera = calibration.rotations_data_from_camera[frame_index]
    translation_data_camera = calibration.translations_data_from_camera[frame_index]
    rotation_data_model = alignment.rotation_data_from_model
    translation_data_model = alignment.translation_data_from_model
    rotation_model_camera = rotation_data_model.T @ rotation_data_camera
    translation_model_camera = rotation_data_model.T @ (
        translation_data_camera - translation_data_model
    )
    return translation_model_camera, rotation_model_camera


def _build_overlay_model(
    urdf_path: str | Path,
    visual_mesh_dir: str | Path,
    fovy_degrees: float,
) -> mujoco.MjModel:
    root = ET.fromstring(build_proxy_mjcf(urdf_path, visual_mesh_dir))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Generated MJCF has no worldbody")
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": HEAD_CAMERA_NAME,
            "pos": "0 0 2",
            "quat": "1 0 0 0",
            "fovy": f"{float(fovy_degrees):.10g}",
        },
    )
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _distortion_remap(calibration: HeadCameraCalibration) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.meshgrid(
        np.arange(calibration.width, dtype=np.float32),
        np.arange(calibration.height, dtype=np.float32),
    )
    distorted_pixels = np.stack((x, y), axis=-1).reshape(-1, 1, 2)
    undistorted_pixels = cv2.undistortPoints(
        distorted_pixels,
        calibration.intrinsic,
        calibration.distortion,
        P=calibration.intrinsic,
    ).reshape(calibration.height, calibration.width, 2)
    return (
        undistorted_pixels[..., 0].astype(np.float32),
        undistorted_pixels[..., 1].astype(np.float32),
    )


class HeadOverlayRenderer:
    """MuJoCo renderer with the recorded OpenCV camera model."""

    def __init__(
        self,
        model: mujoco.MjModel,
        calibration: HeadCameraCalibration,
        alignment: ModelFrameAlignment,
    ) -> None:
        self.model = model
        self.calibration = calibration
        self.alignment = alignment
        self.native_height = calibration.native_render_height
        self.camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, HEAD_CAMERA_NAME
        )
        if self.camera_id < 0:
            raise KeyError(f"Model has no camera {HEAD_CAMERA_NAME!r}")

        self.rgb_renderer = mujoco.Renderer(
            model,
            height=self.native_height,
            width=calibration.width,
        )
        self.depth_renderer = mujoco.Renderer(
            model,
            height=self.native_height,
            width=calibration.width,
        )
        self.depth_renderer.enable_depth_rendering()
        self.segmentation_renderer = mujoco.Renderer(
            model,
            height=self.native_height,
            width=calibration.width,
        )
        self.segmentation_renderer.enable_segmentation_rendering()
        arm_body_ids = set()
        for body_id in range(model.nbody):
            body_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            )
            if body_name.startswith(("arm_l_", "arm_r_", "gripper_l_", "gripper_r_")):
                arm_body_ids.add(body_id)
        self.arm_geom_ids = np.asarray(
            [
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) in arm_body_ids
            ],
            dtype=np.int32,
        )
        if self.arm_geom_ids.size == 0:
            raise RuntimeError("No arm/gripper geoms found for overlay")
        self.distort_map_x, self.distort_map_y = _distortion_remap(calibration)

    def close(self) -> None:
        self.rgb_renderer.close()
        self.depth_renderer.close()
        self.segmentation_renderer.close()

    def render(
        self,
        data: mujoco.MjData,
        frame_index: int,
        render_mode: str = "rgb",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if render_mode not in {"rgb", "depth"}:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        camera_position, rotation_model_opencv = camera_pose_in_model_frame(
            self.calibration,
            self.alignment,
            frame_index,
        )
        rotation_model_mujoco = rotation_model_opencv @ _OPENCV_TO_MUJOCO_CAMERA
        quat_xyzw = Rotation.from_matrix(rotation_model_mujoco).as_quat()
        self.model.cam_pos[self.camera_id] = camera_position
        self.model.cam_quat[self.camera_id] = quat_xyzw[[3, 0, 1, 2]]
        mujoco.mj_forward(self.model, data)

        if render_mode == "rgb":
            self.rgb_renderer.update_scene(data, camera=HEAD_CAMERA_NAME)
            self.rgb_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            robot_image = self.rgb_renderer.render()
        else:
            self.depth_renderer.update_scene(data, camera=HEAD_CAMERA_NAME)
            robot_image = self.depth_renderer.render()
        self.segmentation_renderer.update_scene(data, camera=HEAD_CAMERA_NAME)
        segmentation = self.segmentation_renderer.render()
        robot_mask = np.isin(segmentation[..., 0], self.arm_geom_ids).astype(np.uint8) * 255

        native_cx = float(self.calibration.intrinsic[0, 2])
        native_cy = (
            float(self.calibration.intrinsic[1, 2])
            * self.native_height
            / self.calibration.height
        )
        shift = np.asarray(
            [
                [1.0, 0.0, native_cx - self.calibration.width / 2.0],
                [0.0, 1.0, native_cy - self.native_height / 2.0],
            ],
            dtype=np.float32,
        )
        native_size = (self.calibration.width, self.native_height)
        image_interpolation = (
            cv2.INTER_LINEAR if render_mode == "rgb" else cv2.INTER_NEAREST
        )
        robot_image = cv2.warpAffine(
            robot_image,
            shift,
            native_size,
            flags=image_interpolation,
            borderMode=cv2.BORDER_CONSTANT,
        )
        robot_mask = cv2.warpAffine(
            robot_mask,
            shift,
            native_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        output_size = (self.calibration.width, self.calibration.height)
        robot_image = cv2.resize(
            robot_image,
            output_size,
            interpolation=image_interpolation,
        )
        robot_mask = cv2.resize(robot_mask, output_size, interpolation=cv2.INTER_NEAREST)
        robot_image = cv2.remap(
            robot_image,
            self.distort_map_x,
            self.distort_map_y,
            image_interpolation,
            borderMode=cv2.BORDER_CONSTANT,
        )
        robot_mask = cv2.remap(
            robot_mask,
            self.distort_map_x,
            self.distort_map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        if render_mode == "depth":
            valid = (
                (robot_mask > 0)
                & np.isfinite(robot_image)
                & (robot_image > 0.0)
            )
            robot_rgb = focused_depth_to_rgb(robot_image, valid)
        else:
            robot_rgb = robot_image
        return robot_rgb, robot_mask, camera_position


def composite_robot(
    source_rgb: np.ndarray,
    robot_rgb: np.ndarray,
    robot_mask: np.ndarray,
    *,
    alpha: float,
    feather_sigma: float,
) -> np.ndarray:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    mask = robot_mask.astype(np.float32) / 255.0
    if feather_sigma > 0.0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather_sigma)
    blend = np.clip(mask * alpha, 0.0, 1.0)[..., None]
    return np.clip(source_rgb * (1.0 - blend) + robot_rgb * blend, 0, 255).astype(
        np.uint8
    )


def render_head_overlay_video(
    dataset_dir: str | Path,
    output_path: str | Path,
    *,
    urdf_path: str | Path,
    visual_mesh_dir: str | Path,
    start_frame: int = 0,
    max_frames: int | None = None,
    stride: int = 1,
    alpha: float = 0.86,
    feather_sigma: float = 1.0,
    joint_source: str = "state",
    render_mode: str = "rgb",
    render_only_output_path: str | Path | None = None,
) -> tuple[HeadOverlayStats, HeadCameraCalibration, ModelFrameAlignment]:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    render_only_path = (
        None
        if render_only_output_path is None
        else Path(render_only_output_path).expanduser().resolve()
    )
    trajectory = AgibotTrajectory.load(dataset_dir / "proprio_stats/proprio_stats.h5")
    calibration = load_head_camera_calibration(
        dataset_dir,
        expected_frame_count=trajectory.frame_count,
    )
    if stride <= 0:
        raise ValueError("stride must be positive")
    if render_mode not in {"rgb", "depth"}:
        raise ValueError(f"Unsupported render_mode {render_mode!r}")
    if not 0 <= start_frame < trajectory.frame_count:
        raise ValueError(f"start_frame outside [0, {trajectory.frame_count})")
    stop_frame = trajectory.frame_count
    if max_frames is not None:
        stop_frame = min(stop_frame, start_frame + max_frames * stride)
    frame_indices = range(start_frame, stop_frame, stride)
    output_frame_count = len(frame_indices)

    model = _build_overlay_model(
        urdf_path,
        visual_mesh_dir,
        calibration.native_fovy_degrees,
    )
    controller = AgibotG1Controller(model)
    alignment = fit_model_frame_alignment(
        model,
        controller,
        trajectory,
        joint_source=joint_source,
    )
    controller.clip_counts.clear()
    renderer = HeadOverlayRenderer(model, calibration, alignment)
    source = IndexedVideoReader(
        dataset_dir / "observations/head_color.mp4",
        start_frame,
        width=calibration.width,
        height=calibration.height,
        source_fps=calibration.fps,
    )
    output_fps = calibration.fps / stride
    first_camera_position: np.ndarray | None = None
    LOGGER.info(
        "Rendering %d %s head-camera overlay frames; alignment RMSE %.6f m",
        output_frame_count,
        render_mode,
        alignment.rmse_m,
    )
    try:
        with ExitStack() as stack:
            writer = stack.enter_context(H264Writer(output_path, output_fps))
            render_only_writer = (
                None
                if render_only_path is None
                else stack.enter_context(H264Writer(render_only_path, output_fps))
            )
            for output_index, frame_index in enumerate(frame_indices):
                source_rgb = source.read_rgb(frame_index)
                data = controller.set_frame(
                    trajectory,
                    frame_index,
                    joint_source=joint_source,
                    base_mode="fixed",
                )
                robot_rgb, robot_mask, camera_position = renderer.render(
                    data,
                    frame_index,
                    render_mode=render_mode,
                )
                if first_camera_position is None:
                    first_camera_position = camera_position.copy()
                composed = composite_robot(
                    source_rgb,
                    robot_rgb,
                    robot_mask,
                    alpha=alpha,
                    feather_sigma=feather_sigma,
                )
                if render_mode == "depth":
                    composed = draw_vision_banana_ruler(composed)
                writer.append(composed)
                if render_only_writer is not None:
                    render_only_writer.append(
                        draw_vision_banana_ruler(robot_rgb)
                        if render_mode == "depth"
                        else robot_rgb
                    )
                if output_index == 0 or (output_index + 1) % 300 == 0:
                    LOGGER.info("Rendered %d/%d frames", output_index + 1, output_frame_count)
    finally:
        source.close()
        renderer.close()

    if first_camera_position is None:
        raise RuntimeError("No output frames were rendered")
    stats = HeadOverlayStats(
        frame_count=output_frame_count,
        fps=output_fps,
        width=calibration.width,
        height=calibration.height,
        native_render_height=calibration.native_render_height,
        alignment_rmse_m=alignment.rmse_m,
        first_camera_position_model=tuple(float(v) for v in first_camera_position),
        runtime_clip_counts=dict(controller.clip_counts),
    )
    return stats, calibration, alignment
