#!/usr/bin/env python3
"""Replay RoboCOIN Galbot episode states and overlay the rendered arms."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))

from retarget_galbot.robocoin import (  # noqa: E402
    RoboCoinEpisode,
    RoboCoinHeadCamera,
    RoboCoinOverlayRenderer,
    expand_observation_states,
    validate_eef_forward_kinematics,
)
from retarget_galbot.robocoin.replay import (  # noqa: E402
    H264StreamWriter,
    composite_direct_overlay,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("robocoin-galbot-overlay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=WORKSPACE_ROOT / "datasets/RoboCOIN/Galbot_G1_use_dryer",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=PACKAGE_ROOT / "configs/robocoin_galbot_g1_head_camera.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT
        / "output/robocoin_galbot/episode_000000_arm_overlay.mp4",
    )
    parser.add_argument("--style", choices=("realistic", "debug"), default="realistic")
    parser.add_argument("--opacity", type=float, default=0.82)
    parser.add_argument("--outline", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = RoboCoinEpisode.load(args.dataset_root, args.episode_index)
    actions = expand_observation_states(episode.states)
    validation = validate_eef_forward_kinematics(actions, episode.eef_sim_pose)
    LOGGER.info("EEF FK validation: %s", validation)
    if validation["position_max_m"] > 1e-4 or validation["rotation_max_deg"] > 1e-3:
        raise RuntimeError(f"Expanded qpos failed RoboCOIN EEF validation: {validation}")

    camera = RoboCoinHeadCamera.load(args.camera_config)
    frame_count = episode.frame_count
    if args.max_frames is not None:
        frame_count = min(frame_count, max(0, int(args.max_frames)))
    if frame_count == 0:
        raise ValueError("No frames selected")

    capture = cv2.VideoCapture(str(episode.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {episode.video_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Rendering episode %d: %d frames, style=%s, opacity=%.2f -> %s",
        args.episode_index,
        frame_count,
        args.style,
        args.opacity,
        args.output,
    )
    try:
        with RoboCoinOverlayRenderer(camera, style=args.style) as renderer:
            with H264StreamWriter(args.output, episode.fps) as video_writer:
                for frame_index in range(frame_count):
                    ok, frame_bgr = capture.read()
                    if not ok:
                        raise EOFError(
                            f"Source video ended before frame {frame_index}/{frame_count}"
                        )
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    robot_rgb, robot_mask = renderer.render(actions[frame_index])
                    output = composite_direct_overlay(
                        frame_rgb,
                        robot_rgb,
                        robot_mask,
                        opacity=args.opacity,
                        outline=args.outline or args.style == "debug",
                    )
                    if args.style == "debug":
                        cv2.putText(
                            output,
                            f"episode {args.episode_index:06d} frame {frame_index:04d}",
                            (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (255, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                    video_writer.append(output)
                    if frame_index == 0 or (frame_index + 1) % 150 == 0:
                        LOGGER.info("Rendered %d/%d", frame_index + 1, frame_count)
    finally:
        capture.release()

    metadata = {
        "dataset_root": str(episode.root),
        "episode_index": args.episode_index,
        "source_video": str(episode.video_path),
        "output_video": str(args.output.resolve()),
        "frame_count": frame_count,
        "fps": episode.fps,
        "style": args.style,
        "opacity": args.opacity,
        "camera_config": str(args.camera_config.resolve()),
        "camera": camera.metadata,
        "eef_fk_validation": validation,
        "state_mapping": {
            "leg_joint1..5": "zero; RoboCOIN torso SDK values are not URDF leg angles",
            "left_arm_joint1..7": "observation.state[5:12]",
            "left_gripper_6_joints": "mimic expansion of observation.state[12]",
            "right_arm_joint1..7": "observation.state[13:20]",
            "right_gripper_6_joints": "mimic expansion of observation.state[20]",
            "head_joint1..2": "observation.state[3:5]",
        },
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s and %s", args.output, metadata_path)


if __name__ == "__main__":
    main()
