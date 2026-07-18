#!/usr/bin/env python3
"""Render scaled legacy and unscaled chest-camera overlays, then compare them."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
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


def _ego_video(episode_dir: Path) -> Path:
    path = episode_dir / "ego_process/ego_undistorted_video/raw_video_undistorted.mp4"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _stitch_comparison(legacy: Path, chest: Path, output: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to stitch comparison video")
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = (
        "[0:v]drawtext=text='Original scaled + legacy camera':"
        "x=18:y=18:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.7[left];"
        "[1:v]drawtext=text='New scale=1 + chest camera':"
        "x=18:y=18:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.7[right];"
        "[left][right]hstack=inputs=2[v]"
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "warning",
            "-i",
            str(legacy),
            "-i",
            str(chest),
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-profile:v",
            "high",
            "-level:v",
            "4.1",
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "avc1",
            "-movflags",
            "+faststart",
            "-an",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    from retarget_galbot.egoview.chest_overlay import render_chest_overlay_video
    from retarget_galbot.pipeline import RetargetSession
    from retarget_galbot.robots import get_spec

    episode_dir = args.episode_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = get_spec("galbot")

    legacy_session = RetargetSession(spec, config_path=args.config)
    legacy_result = legacy_session.retarget(episode_dir, max_frames=args.max_frames)
    chest_session = RetargetSession(spec, config_path=args.config, source_scale=1.0)
    chest_result = chest_session.retarget(episode_dir, max_frames=args.max_frames)
    legacy_frames = [int(frame.frame) for frame in legacy_result.frames]
    chest_frames = [int(frame.frame) for frame in chest_result.frames]
    if legacy_frames != chest_frames:
        raise RuntimeError("Legacy and chest retarget frame indices differ")

    legacy_actions = output_dir / "galbot_actions_scaled_legacy.npy"
    chest_actions = output_dir / "galbot_actions_unscaled_chest.npy"
    np.save(legacy_actions, legacy_result.actions)
    np.save(chest_actions, chest_result.actions)
    logger.info(
        "Retarget scales: legacy=%.6f, chest=%.6f",
        legacy_result.scale,
        chest_result.scale,
    )

    common = {
        "spec": spec,
        "fps": float(legacy_result.meta.fps),
        "feather_sigma": 1.0,
    }
    legacy_video = output_dir / "legacy_scaled_direct_overlay.mp4"
    render_chest_overlay_video(
        legacy_result.actions,
        legacy_frames,
        _ego_video(episode_dir),
        legacy_video,
        source_shoulder_mid_head=legacy_session.retargeter.source_shoulder_offset,
        head_rotation_base=legacy_session.retargeter.head_pose0.rotation,
        camera_mode="legacy",
        **common,
    )
    chest_video = output_dir / "chest_unscaled_direct_overlay.mp4"
    render_chest_overlay_video(
        chest_result.actions,
        chest_frames,
        _ego_video(episode_dir),
        chest_video,
        source_shoulder_mid_head=chest_session.retargeter.source_shoulder_offset,
        head_rotation_base=chest_session.retargeter.head_pose0.rotation,
        camera_mode="chest",
        **common,
    )

    comparison = output_dir / "legacy_scaled_vs_chest_unscaled.mp4"
    _stitch_comparison(legacy_video, chest_video, comparison)
    logger.info("Comparison video: %s", comparison)


if __name__ == "__main__":
    main()
