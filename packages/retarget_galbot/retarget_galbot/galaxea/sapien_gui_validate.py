from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np

from .urdf_utils import prepare_sapien_urdf


def _load_payload(path: str | Path) -> dict:
    with Path(path).expanduser().resolve().open("rb") as file:
        return pickle.load(file)


def _make_material(sapien, color: tuple[float, float, float, float]):
    material = sapien.render.RenderMaterial()
    material.set_base_color(np.asarray(color, dtype=np.float32))
    material.set_roughness(0.65)
    return material


def _make_marker(scene, sapien, name: str, color: tuple[float, float, float, float]):
    builder = scene.create_actor_builder()
    builder.add_sphere_visual(radius=0.025, material=_make_material(sapien, color))
    return builder.build_kinematic(name=name)


def _set_marker_pose(marker, sapien, position: np.ndarray) -> None:
    marker.set_pose(sapien.Pose(np.asarray(position, dtype=np.float64)))


def _retargeting_to_sapien_indices(robot, joint_names: list[str]) -> np.ndarray:
    active_names = [joint.get_name() for joint in robot.get_active_joints()]
    missing = [name for name in active_names if name not in joint_names]
    if missing:
        raise ValueError(f"SAPIEN active joints are absent from qpos metadata: {missing}")
    return np.asarray([joint_names.index(name) for name in active_names], dtype=int)


def run_gui(
    pickle_path: str | Path,
    output_video_path: str | Path | None = None,
    headless: bool = False,
    fps: int = 30,
    stride: int = 1,
    loop: bool = False,
    with_collisions: bool = False,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import cv2
    import sapien
    from sapien.asset import create_dome_envmap
    from sapien.utils import Viewer

    payload = _load_payload(pickle_path)
    meta = payload["meta_data"]
    frames = payload["data"][::stride]
    if meta.get("backend") != "dex_bimanual":
        raise ValueError("SAPIEN GUI validation currently expects backend=dex_bimanual")

    try:
        if headless:
            sapien.render.set_viewer_shader_dir("rt")
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(16)
            sapien.render.set_ray_tracing_path_depth(8)
            sapien.render.set_ray_tracing_denoiser("oidn")
        else:
            sapien.render.set_viewer_shader_dir("default")
            sapien.render.set_camera_shader_dir("default")

        scene = sapien.Scene()
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize SAPIEN. This usually means the current machine "
            "does not expose a usable Vulkan/GPU display. Run this script on a "
            "desktop/Vulkan node, or use visualize_output.py for non-GUI FK plots."
        ) from exc

    scene.set_timestep(1 / fps)
    ground_mat = _make_material(sapien, (0.06, 0.08, 0.10, 1.0))
    scene.add_ground(0.0, render_material=ground_mat, render_half_size=[4, 4])
    scene.set_ambient_light(np.array([0.5, 0.5, 0.5]))
    scene.add_directional_light(np.array([1, 1, -1]), np.array([2, 2, 2]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.18, 0.18, 0.2], ground_color=[0.1, 0.1, 0.1])
    )

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = with_collisions
    sapien_urdf = prepare_sapien_urdf(
        meta["robot_urdf"], include_collisions=with_collisions
    )
    robot = loader.load(str(sapien_urdf))
    robot.set_pose(sapien.Pose([0, 0, 0]))

    retargeting_to_sapien = _retargeting_to_sapien_indices(
        robot, list(meta["joint_names"])
    )
    left_marker = _make_marker(scene, sapien, "left_tcp_target", (1.0, 0.1, 0.9, 1.0))
    right_marker = _make_marker(scene, sapien, "right_tcp_target", (0.0, 0.8, 1.0, 1.0))

    camera = scene.add_camera("validation_camera", 1280, 720, 1.0, 0.05, 20)
    camera.set_local_pose(sapien.Pose([2.0, -3.0, 1.6], [0.9238795, 0.3826834, 0, 0]))

    viewer = None
    if not headless:
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.set_camera_xyz(2.0, -3.0, 1.6)
        viewer.set_camera_rpy(0.0, -0.45, 0.62)
        viewer.control_window.show_origin_frame = False
        viewer.control_window.move_speed = 0.03

    writer = None
    if output_video_path:
        output_path = Path(output_video_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (camera.get_width(), camera.get_height()),
        )

    frame_index = 0
    while True:
        frame = frames[frame_index]
        qpos = np.asarray(frame["qpos"], dtype=np.float64)
        robot.set_qpos(qpos[retargeting_to_sapien])
        _set_marker_pose(left_marker, sapien, np.asarray(frame["left"]["target_tcp_position"]))
        _set_marker_pose(right_marker, sapien, np.asarray(frame["right"]["target_tcp_position"]))

        if viewer is not None:
            viewer.render()
            if viewer.closed:
                break
            time.sleep(1 / fps)
        else:
            scene.update_render()

        if writer is not None:
            camera.take_picture()
            rgb = camera.get_picture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            writer.write(rgb[..., ::-1])

        frame_index += 1
        if frame_index >= len(frames):
            if loop and viewer is not None:
                frame_index = 0
            else:
                break

    if writer is not None:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Open a SAPIEN GUI to validate Galbot retargeting.")
    parser.add_argument("--pickle", required=True, help="Retarget output pickle.")
    parser.add_argument("--output-video", default=None, help="Optional mp4 recording path.")
    parser.add_argument("--headless", action="store_true", help="Render without opening the viewer.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--with-collisions",
        action="store_true",
        help="Load URDF collision meshes. Disabled by default to avoid STL collision warnings.",
    )
    args = parser.parse_args()
    run_gui(
        pickle_path=args.pickle,
        output_video_path=args.output_video,
        headless=args.headless,
        fps=args.fps,
        stride=args.stride,
        loop=args.loop,
        with_collisions=args.with_collisions,
    )


if __name__ == "__main__":
    main()
