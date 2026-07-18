#!/usr/bin/env python3
"""Replay an AgiBot G1 real trajectory in a MuJoCo G1_120s model."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))

from retarget_agibot import AgibotG1Controller, AgibotTrajectory, build_proxy_model  # noqa: E402
from retarget_agibot.video import (  # noqa: E402
    H264Writer,
    IndexedVideoReader,
    draw_label,
    make_camera,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("agibot-replay")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=WORKSPACE_ROOT / "datasets/agibot/proprio_stats/proprio_stats.h5",
    )
    parser.add_argument(
        "--source-video",
        type=Path,
        default=WORKSPACE_ROOT / "datasets/agibot/observations/head_color.mp4",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=WORKSPACE_ROOT
        / "assets/robots/G1_v2.3/G1_120s/G1_120s.urdf",
    )
    parser.add_argument(
        "--visual-mesh-dir",
        type=Path,
        default=WORKSPACE_ROOT
        / "assets/robots/G1_v2.3/G1_120s/mujoco_articulated/meshes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "output/agibot/g1_120s_real_trajectory_replay.mp4",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--joint-source", choices=("state", "action"), default="state")
    parser.add_argument("--base-mode", choices=("world", "relative", "fixed"), default="world")
    parser.add_argument(
        "--visual-mode",
        choices=("official", "proxy"),
        default="official",
        help="Use link-local official meshes or the primitive debug proxy.",
    )
    parser.add_argument("--show-joints", action="store_true")
    parser.add_argument(
        "--layout",
        choices=("simulation", "source_simulation", "source_follow_world"),
        default="source_follow_world",
    )
    parser.add_argument("--write-proxy-xml", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    trajectory = AgibotTrajectory.load(args.trajectory)
    if not 0 <= args.start_frame < trajectory.frame_count:
        raise ValueError(f"--start-frame outside [0, {trajectory.frame_count})")
    stop = trajectory.frame_count
    if args.max_frames is not None:
        stop = min(stop, args.start_frame + args.max_frames * args.stride)
    indices = range(args.start_frame, stop, args.stride)
    output_frames = len(indices)
    fps = trajectory.fps / args.stride

    limit_path = args.output.with_name(f"{args.output.stem}_limits.json")
    report = trajectory.write_limit_report(limit_path, args.urdf, args.joint_source)
    LOGGER.info(
        "Loaded %d frames (%.3f fps, %.3f s); URDF violations: %d",
        trajectory.frame_count,
        trajectory.fps,
        trajectory.duration_seconds,
        report["total_violation_count"],
    )

    from retarget_agibot.proxy_model import build_proxy_mjcf

    visual_mesh_dir = args.visual_mesh_dir if args.visual_mode == "official" else None
    if args.write_proxy_xml is not None:
        args.write_proxy_xml.parent.mkdir(parents=True, exist_ok=True)
        args.write_proxy_xml.write_text(
            build_proxy_mjcf(args.urdf, visual_mesh_dir), encoding="utf-8"
        )
    model = build_proxy_model(args.urdf, visual_mesh_dir)
    controller = AgibotG1Controller(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = args.show_joints
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False

    world_center = np.asarray(
        [
            float((trajectory.base_position[:, 0].min() + trajectory.base_position[:, 0].max()) / 2),
            float((trajectory.base_position[:, 1].min() + trajectory.base_position[:, 1].max()) / 2),
            1.0,
        ]
    )
    world_camera = make_camera(lookat=world_center, distance=4.2, azimuth=135, elevation=-18)
    follow_camera = make_camera(lookat=world_center, distance=2.8, azimuth=145, elevation=-15)

    needs_source = args.layout != "simulation"
    source = (
        IndexedVideoReader(
            args.source_video,
            args.start_frame,
            width=args.width,
            height=args.height,
        )
        if needs_source
        else None
    )
    LOGGER.info("Rendering %d output frames to %s", output_frames, args.output)
    try:
        with H264Writer(args.output, fps) as writer:
            for output_index, frame_index in enumerate(indices):
                data = controller.set_frame(
                    trajectory,
                    frame_index,
                    joint_source=args.joint_source,
                    base_mode=args.base_mode,
                )
                base = data.qpos[:3]
                follow_camera.lookat[:] = base + np.asarray([0.0, 0.0, 1.05])
                renderer.update_scene(data, camera=follow_camera, scene_option=scene_option)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                follow = renderer.render()
                telemetry = (
                    f"frame {frame_index}/{trajectory.frame_count - 1}  t={frame_index / trajectory.fps:.2f}s",
                    f"base xyz=({base[0]:+.3f}, {base[1]:+.3f}, {base[2]:+.3f})",
                    "gripper closed L/R="
                    f"{trajectory.effector_closed_fraction(frame_index, args.joint_source)[0]:.2f}/"
                    f"{trajectory.effector_closed_fraction(frame_index, args.joint_source)[1]:.2f}",
                )
                follow = draw_label(follow, "MuJoCo G1_120s / tracking camera", telemetry)

                panels: list[np.ndarray]
                if args.layout == "simulation":
                    panels = [follow]
                else:
                    assert source is not None
                    source_frame = source.read_rgb(frame_index)
                    source_frame = draw_label(source_frame, "Real head camera", (f"frame {frame_index}",))
                    panels = [source_frame, follow]
                    if args.layout == "source_follow_world":
                        renderer.update_scene(data, camera=world_camera, scene_option=scene_option)
                        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                        world = draw_label(
                            renderer.render(),
                            "MuJoCo G1_120s / fixed world camera",
                            ("base motion remains visible",),
                        )
                        panels.append(world)
                writer.append(np.concatenate(panels, axis=1))
                if output_index == 0 or (output_index + 1) % 300 == 0:
                    LOGGER.info("Rendered %d/%d frames", output_index + 1, output_frames)
    finally:
        renderer.close()
        if source is not None:
            source.close()

    metadata = {
        "output": str(args.output.resolve()),
        "trajectory": str(args.trajectory.resolve()),
        "source_video": str(args.source_video.resolve()) if needs_source else None,
        "urdf": str(args.urdf.resolve()),
        "visual_mode": args.visual_mode,
        "visual_mesh_dir": (
            str(args.visual_mesh_dir.resolve()) if visual_mesh_dir is not None else None
        ),
        "frame_count": output_frames,
        "fps": fps,
        "start_frame": args.start_frame,
        "stride": args.stride,
        "joint_source": args.joint_source,
        "base_mode": args.base_mode,
        "layout": args.layout,
        "runtime_clip_counts": controller.clip_counts,
        "limit_report": str(limit_path.resolve()),
        "asset_note": (
            "Official G1_120s USD meshes exported as link-local OBJ files."
            if visual_mesh_dir is not None
            else "Primitive URDF-aligned debug proxy."
        ),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Finished %s; runtime clips: %s", args.output, controller.clip_counts)


if __name__ == "__main__":
    main()
