#!/usr/bin/env python3
"""Retarget an ego episode and stream a Galbot arm overlay MP4."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _find_ego_video(episode_dir: Path) -> Path:
    video = (
        episode_dir
        / "ego_process"
        / "ego_undistorted_video"
        / "raw_video_undistorted.mp4"
    )
    if not video.exists():
        raise FileNotFoundError(f"Missing undistorted ego video: {video}")
    return video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct Galbot arm overlay with chest-unscaled or legacy-scaled mode"
    )
    parser.add_argument("--episode_dir", type=Path, required=True)
    parser.add_argument("--output_mp4", type=Path, required=True)
    parser.add_argument("--actions_output", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("chest_unscaled", "legacy_scaled"),
        default="chest_unscaled",
    )
    parser.add_argument(
        "--source_scale",
        type=float,
        default=None,
        help="Optional explicit override; defaults to 1 for chest mode and auto for legacy mode",
    )
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--feather_sigma", type=float, default=1.0)
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    from retarget_galbot.egoview.chest_overlay import render_chest_overlay_video
    from retarget_galbot.pipeline import RetargetSession
    from retarget_galbot.robots import get_spec

    episode_dir = args.episode_dir.expanduser().resolve()
    spec = get_spec("galbot")
    source_scale = args.source_scale
    if source_scale is None and args.mode == "chest_unscaled":
        source_scale = 1.0
    session = RetargetSession(
        spec,
        config_path=args.config,
        source_scale=source_scale,
    )
    result = session.retarget(
        episode_dir,
        max_frames=args.max_frames,
        stride=args.stride,
    )
    if source_scale is not None and result.scale != source_scale:
        raise RuntimeError(
            f"Expected source scale {source_scale}, retarget used {result.scale}"
        )

    if args.actions_output is not None:
        actions_output = args.actions_output.expanduser().resolve()
        actions_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(actions_output, result.actions)
        logger.info("Saved actions: %s", actions_output)

    source_frames = [int(frame.frame) for frame in result.frames]
    output_fps = float(result.meta.fps) / float(args.stride)
    stats = render_chest_overlay_video(
        result.actions,
        source_frames,
        _find_ego_video(episode_dir),
        args.output_mp4.expanduser().resolve(),
        spec=spec,
        source_shoulder_mid_head=session.retargeter.source_shoulder_offset,
        head_rotation_base=session.retargeter.head_pose0.rotation,
        fps=output_fps,
        feather_sigma=args.feather_sigma,
        camera_mode="legacy" if args.mode == "legacy_scaled" else "chest",
    )
    logger.info(
        "Wrote %s: mode=%s, scale=%.6f, %d frames, %dx%d @ %.3f fps, "
        "fovy=%.3f deg, camera0=%s",
        args.output_mp4.expanduser().resolve(),
        args.mode,
        result.scale,
        stats.frame_count,
        stats.width,
        stats.height,
        stats.fps,
        stats.fovy_degrees,
        stats.first_camera_position,
    )


if __name__ == "__main__":
    main()
