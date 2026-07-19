#!/usr/bin/env python3
"""Run the Galbot SAM2 segmentation and ProPainter stages without retargeting."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _find_ego_video(episode_dir: Path) -> Path:
    from retarget_galbot.constants.aoe import UNDISTORTED_VIDEO_SUBDIRS

    preferred_names = ("raw_video_undistorted.mp4", "undistorted_video.mp4")
    for subdir in UNDISTORTED_VIDEO_SUBDIRS:
        video_dir = episode_dir / subdir
        for name in preferred_names:
            candidate = video_dir / name
            if candidate.exists():
                return candidate
        candidates = sorted(video_dir.glob("*.mp4"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No undistorted ego MP4 found under {episode_dir}")


def _video_metadata(path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open ego video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return frame_count, fps if fps > 0 else 30.0


def _write_mp4(path: Path, frames: np.ndarray, fps: float) -> None:
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3 or not len(frames):
        raise ValueError(f"Expected non-empty RGB frames (T,H,W,3), got {frames.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
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
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    logger.info("Wrote %s", path)


def _load_mask(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    mask = np.asarray(np.load(path)).astype(bool)
    if mask.shape != expected_shape:
        raise ValueError(f"Mask shape {mask.shape} does not match {expected_shape}")
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAM2 hand/arm masks and ProPainter completion for Galbot"
    )
    parser.add_argument("--episode_dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--skip_sam", action="store_true")
    parser.add_argument("--skip_inpaint", action="store_true")
    parser.add_argument(
        "--mask_input",
        type=Path,
        default=None,
        help="Existing bool mask .npy; implied when --skip_sam is set",
    )
    args = parser.parse_args()

    if args.start_frame < 0:
        raise ValueError("--start_frame must be >= 0")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max_frames must be >= 1")

    from retarget_galbot.egoview.render import (
        _read_first_n_frames,
        _run_propainter,
        _segment_hands_sam2,
    )

    episode_dir = args.episode_dir.expanduser().resolve()
    video_path = (args.video or _find_ego_video(episode_dir)).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames, fps = _video_metadata(video_path)
    available = total_frames - args.start_frame
    frame_count = min(args.max_frames or available, available)
    if frame_count < 1:
        raise ValueError(
            f"No frames available at start frame {args.start_frame} in {video_path}"
        )
    frames = _read_first_n_frames(
        video_path,
        frame_count,
        out_size=(args.width, args.height),
        start_frame=args.start_frame,
    )

    mask_path = output_dir / "sam2_hand_arm_mask.npy"
    if args.skip_sam or args.mask_input is not None:
        source_mask = args.mask_input or mask_path
        masks = _load_mask(source_mask.expanduser().resolve(), frames.shape[:3])
        logger.info("Loaded existing masks: %s", source_mask)
    else:
        masks = _segment_hands_sam2(
            frames,
            episode_dir,
            args.width,
            args.height,
            frame_offset=args.start_frame,
        )
        np.save(mask_path, masks)
        logger.info("Saved lossless masks: %s", mask_path)

    mask_rgb = np.repeat((masks[..., None] * 255).astype(np.uint8), 3, axis=-1)
    _write_mp4(output_dir / "sam2_hand_arm_mask.mp4", mask_rgb, fps)

    if not args.skip_inpaint:
        completed = _run_propainter(frames, masks)
        _write_mp4(output_dir / "propainter_inpaint.mp4", completed, fps)


if __name__ == "__main__":
    main()
