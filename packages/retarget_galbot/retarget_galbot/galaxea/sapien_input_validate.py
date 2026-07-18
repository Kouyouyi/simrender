from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from .coordinates import opencv_camera_to_display_position, opencv_camera_to_display_rotation
from .ego_data import load_ego_hand_sequence


def _apply_display_positions(positions: np.ndarray, display_space: str) -> np.ndarray:
    if display_space != "camera":
        return np.asarray(positions, dtype=np.float64)
    return np.asarray(
        [opencv_camera_to_display_position(point) for point in positions], dtype=np.float64
    )


def _apply_display_rotations(rotations: np.ndarray, display_space: str) -> np.ndarray:
    if display_space != "camera":
        return np.asarray(rotations, dtype=np.float64)
    return np.asarray(
        [opencv_camera_to_display_rotation(rotation) for rotation in rotations],
        dtype=np.float64,
    )


def _prepare_visualization_data(
    reconstruction: Path,
    frame_ids: np.ndarray,
    display_space: str,
    show_camera: bool,
    fps: float,
) -> dict[str, np.ndarray | list | None]:
    reconstruction_dir = _resolve_reconstruction_dir(reconstruction)
    left_cam = load_ego_hand_sequence(reconstruction, hand="left", source_space="camera", fps=fps)
    right_cam = load_ego_hand_sequence(reconstruction, hand="right", source_space="camera", fps=fps)

    if display_space == "camera":
        left_positions = left_cam.trans[frame_ids].copy()
        right_positions = right_cam.trans[frame_ids].copy()
        left_rotations = np.asarray(
            [_rotvec_to_matrix(rot) for rot in left_cam.rot_axis_angle[frame_ids]]
        )
        right_rotations = np.asarray(
            [_rotvec_to_matrix(rot) for rot in right_cam.rot_axis_angle[frame_ids]]
        )
        left_positions = _apply_display_positions(left_positions, "camera")
        right_positions = _apply_display_positions(right_positions, "camera")
        left_rotations = _apply_display_rotations(left_rotations, "camera")
        right_rotations = _apply_display_rotations(right_rotations, "camera")
        camera_positions = np.zeros((len(frame_ids), 3), dtype=np.float64)
        camera_rotations = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], len(frame_ids), axis=0)
    else:
        left_world = load_ego_hand_sequence(reconstruction, hand="left", source_space="world", fps=fps)
        right_world = load_ego_hand_sequence(reconstruction, hand="right", source_space="world", fps=fps)
        left_positions = left_world.trans[frame_ids].copy()
        right_positions = right_world.trans[frame_ids].copy()
        left_rotations = np.asarray(
            [_rotvec_to_matrix(rot) for rot in left_world.rot_axis_angle[frame_ids]]
        )
        right_rotations = np.asarray(
            [_rotvec_to_matrix(rot) for rot in right_world.rot_axis_angle[frame_ids]]
        )
        if show_camera:
            camera_positions, camera_rotations = _load_camera_poses(reconstruction_dir, frame_ids)
        else:
            camera_positions = None
            camera_rotations = None

    return {
        "left_positions": left_positions,
        "right_positions": right_positions,
        "left_rotations": left_rotations,
        "right_rotations": right_rotations,
        "left_valid": left_cam.valid[frame_ids],
        "right_valid": right_cam.valid[frame_ids],
        "camera_positions": camera_positions,
        "camera_rotations": camera_rotations,
    }


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * skew
        + (1.0 - np.cos(theta)) * (skew @ skew)
    )


def _resolve_reconstruction_dir(reconstruction: Path) -> Path:
    path = reconstruction.expanduser().resolve()
    if path.is_file():
        return path.parent
    if (path / "hands.npz").exists():
        return path
    nested = path / "ego_process" / "ego_hands_reconstruction"
    if nested.exists():
        return nested
    return path


def _load_camera_poses(reconstruction_dir: Path, frame_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera_traj_path = reconstruction_dir / "camera_traj.npz"
    hands_path = reconstruction_dir / "hands.npz"
    if camera_traj_path.exists():
        cam_c2w = np.asarray(np.load(camera_traj_path)["cam_c2w"], dtype=np.float64)
    elif hands_path.exists():
        hands = np.load(hands_path, mmap_mode="r")
        cam_c2w = np.zeros((hands["R_c2w"].shape[0], 4, 4), dtype=np.float64)
        cam_c2w[:, :3, :3] = np.asarray(hands["R_c2w"], dtype=np.float64)
        cam_c2w[:, :3, 3] = np.asarray(hands["t_c2w"], dtype=np.float64)
        cam_c2w[:, 3, 3] = 1.0
    else:
        raise FileNotFoundError(
            f"Missing camera pose data. Expected camera_traj.npz or hands.npz in {reconstruction_dir}"
        )

    poses = cam_c2w[frame_ids]
    positions = poses[:, :3, 3].copy()
    rotations = poses[:, :3, :3].copy()
    return positions, rotations


def _make_material(sapien, color: tuple[float, float, float, float]):
    material = sapien.render.RenderMaterial()
    material.set_base_color(np.asarray(color, dtype=np.float32))
    material.set_roughness(0.65)
    return material


def _make_sphere(scene, sapien, name: str, radius: float, color: tuple[float, float, float, float]):
    builder = scene.create_actor_builder()
    builder.add_sphere_visual(radius=radius, material=_make_material(sapien, color))
    return builder.build_kinematic(name=name)


def _set_pose(actor, sapien, position: np.ndarray) -> None:
    actor.set_pose(sapien.Pose(np.asarray(position, dtype=np.float64)))


def _scene_bounds(positions: np.ndarray) -> tuple[np.ndarray, float]:
    if len(positions) == 0:
        return np.zeros(3, dtype=np.float64), 0.5
    center = positions.mean(axis=0)
    radius = float(np.max(np.linalg.norm(positions - center, axis=1)))
    return center, max(radius, 0.15)


def _add_trail_markers(
    scene,
    sapien,
    prefix: str,
    positions: np.ndarray,
    valid: np.ndarray,
    trail_stride: int,
    radius: float,
    color: tuple[float, float, float, float],
) -> list:
    markers: list = []
    for index in range(0, len(positions), max(trail_stride, 1)):
        if not valid[index]:
            continue
        marker = _make_sphere(scene, sapien, f"{prefix}_trail_{index}", radius, color)
        _set_pose(marker, sapien, positions[index])
        markers.append(marker)
    return markers


def _add_axis_markers(
    scene,
    sapien,
    prefix: str,
    colors: tuple[tuple[float, float, float, float], ...],
    radius: float,
) -> list:
    return [
        _make_sphere(scene, sapien, f"{prefix}_axis_{axis_index}", radius, color)
        for axis_index, color in enumerate(colors)
    ]


def _update_axis_markers(
    markers: list,
    sapien,
    position: np.ndarray,
    rotation: np.ndarray,
    axis_length: float,
) -> None:
    for axis_index, marker in enumerate(markers):
        tip = position + rotation[:, axis_index] * axis_length
        _set_pose(marker, sapien, tip)


def _frustum_points(depth: float = 0.14, half_width: float = 0.06, half_height: float = 0.04) -> np.ndarray:
    return np.asarray(
        [
            [-half_width, -half_height, depth],
            [half_width, -half_height, depth],
            [half_width, half_height, depth],
            [-half_width, half_height, depth],
        ],
        dtype=np.float64,
    )


def run_gui(
    reconstruction: str | Path,
    display_space: str = "camera",
    fps: int = 30,
    stride: int = 1,
    max_frames: int | None = None,
    loop: bool = False,
    trail_stride: int = 8,
    show_orientation: bool = True,
    show_camera: bool = True,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import sapien
    from sapien.asset import create_dome_envmap
    from sapien.utils import Viewer

    reconstruction_path = Path(reconstruction).expanduser().resolve()
    left_cam = load_ego_hand_sequence(reconstruction_path, hand="left", source_space="camera", fps=fps)
    frame_count = left_cam.frame_count
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    frame_ids = np.arange(0, frame_count, stride, dtype=int)
    if len(frame_ids) == 0:
        raise ValueError("No frames available for visualization")

    data = _prepare_visualization_data(
        reconstruction_path,
        frame_ids,
        display_space=display_space,
        show_camera=show_camera,
        fps=fps,
    )
    left_positions = data["left_positions"]
    right_positions = data["right_positions"]
    left_rotations = data["left_rotations"]
    right_rotations = data["right_rotations"]
    left_valid = data["left_valid"]
    right_valid = data["right_valid"]
    camera_positions = data["camera_positions"]
    camera_rotations = data["camera_rotations"]

    try:
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")
        scene = sapien.Scene()
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize SAPIEN. This usually means the current machine "
            "does not expose a usable Vulkan/GPU display."
        ) from exc

    scene.set_timestep(1 / fps)

    sample_positions = [np.zeros((1, 3))]
    if np.any(left_valid):
        sample_positions.append(left_positions[left_valid])
    if np.any(right_valid):
        sample_positions.append(right_positions[right_valid])
    if show_camera and camera_positions is not None:
        sample_positions.append(camera_positions)
    all_positions = np.concatenate(sample_positions, axis=0)
    center, radius = _scene_bounds(all_positions)
    ground_z = float(all_positions[:, 2].min()) - 0.05

    ground_mat = _make_material(sapien, (0.06, 0.08, 0.10, 1.0))
    scene.add_ground(ground_z, render_material=ground_mat, render_half_size=[4, 4])
    scene.set_ambient_light(np.array([0.5, 0.5, 0.5]))
    scene.add_directional_light(np.array([1, 1, -1]), np.array([2, 2, 2]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.18, 0.18, 0.2], ground_color=[0.1, 0.1, 0.1])
    )

    origin = _make_sphere(scene, sapien, "origin", 0.012, (0.9, 0.9, 0.9, 1.0))
    _set_pose(origin, sapien, np.zeros(3))

    left_current = _make_sphere(scene, sapien, "left_wrist", 0.03, (1.0, 0.1, 0.9, 1.0))
    right_current = _make_sphere(scene, sapien, "right_wrist", 0.03, (0.0, 0.8, 1.0, 1.0))
    _add_trail_markers(
        scene,
        sapien,
        "left",
        left_positions,
        left_valid,
        trail_stride,
        0.01,
        (1.0, 0.1, 0.9, 0.35),
    )
    _add_trail_markers(
        scene,
        sapien,
        "right",
        right_positions,
        right_valid,
        trail_stride,
        0.01,
        (0.0, 0.8, 1.0, 0.35),
    )

    wrist_axis_length = 0.08
    hand_axis_colors = (
        (1.0, 0.2, 0.2, 1.0),
        (0.2, 1.0, 0.2, 1.0),
        (0.2, 0.4, 1.0, 1.0),
    )
    left_axes: list = []
    right_axes: list = []
    if show_orientation:
        left_axes = _add_axis_markers(scene, sapien, "left", hand_axis_colors, 0.012)
        right_axes = _add_axis_markers(scene, sapien, "right", hand_axis_colors, 0.012)

    camera_current = None
    camera_axes: list = []
    camera_frustum: list = []
    if show_camera and camera_positions is not None and camera_rotations is not None:
        camera_current = _make_sphere(scene, sapien, "camera", 0.035, (1.0, 0.75, 0.1, 1.0))
        camera_axis_colors = (
            (1.0, 0.45, 0.05, 1.0),
            (1.0, 0.85, 0.15, 1.0),
            (0.95, 0.65, 0.05, 1.0),
        )
        camera_axes = _add_axis_markers(scene, sapien, "camera", camera_axis_colors, 0.014)
        camera_frustum = [
            _make_sphere(scene, sapien, f"camera_frustum_{index}", 0.008, (1.0, 0.8, 0.2, 0.9))
            for index in range(4)
        ]
        if display_space == "world":
            camera_valid = np.ones(len(camera_positions), dtype=bool)
            _add_trail_markers(
                scene,
                sapien,
                "camera",
                camera_positions,
                camera_valid,
                trail_stride,
                0.012,
                (1.0, 0.75, 0.1, 0.35),
            )

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.set_camera_xyz(
        float(center[0] + radius * 1.8),
        float(center[1] - radius * 2.2),
        float(center[2] + radius * 1.2),
    )
    viewer.set_camera_rpy(0.0, -0.45, 0.62)
    viewer.control_window.show_origin_frame = True
    viewer.control_window.move_speed = 0.03
    print(
        f"visualizing input | display_space={display_space} | frames={len(frame_ids)}"
    )
    if display_space == "camera":
        print(
            "camera frame: camera fixed at origin; wrists use pred_trans_cam with "
            "OpenCV(y-down)->display(z-up) conversion. Hands should appear below the camera."
        )
    else:
        print(
            "world frame: reconstruction-world coordinates are arbitrary and may not align "
            "with gravity or head-above-hands intuition."
        )

    camera_axis_length = 0.14
    if display_space == "camera":
        frustum_local = _apply_display_positions(_frustum_points(), "camera")
    else:
        frustum_local = _frustum_points()
    frame_index = 0
    while not viewer.closed:
        if left_valid[frame_index]:
            _set_pose(left_current, sapien, left_positions[frame_index])
            if show_orientation:
                _update_axis_markers(
                    left_axes,
                    sapien,
                    left_positions[frame_index],
                    left_rotations[frame_index],
                    wrist_axis_length,
                )
        if right_valid[frame_index]:
            _set_pose(right_current, sapien, right_positions[frame_index])
            if show_orientation:
                _update_axis_markers(
                    right_axes,
                    sapien,
                    right_positions[frame_index],
                    right_rotations[frame_index],
                    wrist_axis_length,
                )

        if show_camera and camera_positions is not None and camera_rotations is not None:
            camera_position = camera_positions[frame_index]
            camera_rotation = camera_rotations[frame_index]
            if camera_current is not None:
                _set_pose(camera_current, sapien, camera_position)
            _update_axis_markers(
                camera_axes,
                sapien,
                camera_position,
                camera_rotation,
                camera_axis_length,
            )
            for corner_index, marker in enumerate(camera_frustum):
                corner_world = camera_rotation @ frustum_local[corner_index] + camera_position
                _set_pose(marker, sapien, corner_world)

        viewer.render()
        time.sleep(1 / fps)

        frame_index += 1
        if frame_index >= len(frame_ids):
            if loop:
                frame_index = 0
            else:
                break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive SAPIEN viewer for pre-retarget wrist and camera coordinates."
    )
    parser.add_argument(
        "--reconstruction",
        required=True,
        help="Path to ego_hands_reconstruction/, hands.npz, or a segment folder.",
    )
    parser.add_argument(
        "--display-space",
        default="camera",
        choices=["camera", "world"],
        help=(
            "camera: ego-centric view with camera at origin (recommended). "
            "world: reconstruction-world trajectory (arbitrary up axis)."
        ),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--trail-stride",
        type=int,
        default=8,
        help="Spacing between static trail markers along each trajectory.",
    )
    parser.add_argument(
        "--no-orientation",
        action="store_true",
        help="Hide wrist orientation axis markers.",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Hide camera pose visualization.",
    )
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.trail_stride < 1:
        raise ValueError("--trail-stride must be >= 1")

    run_gui(
        reconstruction=args.reconstruction,
        display_space=args.display_space,
        fps=args.fps,
        stride=args.stride,
        max_frames=args.max_frames,
        loop=args.loop,
        trail_stride=args.trail_stride,
        show_orientation=not args.no_orientation,
        show_camera=not args.no_camera,
    )


if __name__ == "__main__":
    main()
