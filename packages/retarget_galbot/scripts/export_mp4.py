#!/usr/bin/env python3
"""Export a side-by-side MP4: egoview | mujoco/front | mujoco/top."""

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

PANEL_LABELS = ("egoview", "mujoco/front", "mujoco/top")


def _draw_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (4, 4), (12 + tw, 12 + th + baseline), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        text,
        (8, 8 + th),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _resize_panel(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def stitch_side_by_side(
    egoview: np.ndarray,
    front: np.ndarray,
    top: np.ndarray,
    *,
    panel_width: int | None = None,
    panel_height: int | None = None,
    draw_labels: bool = True,
) -> np.ndarray:
    """Stack three RGB videos horizontally into (T, H, 3W, 3)."""
    egoview = np.asarray(egoview)
    front = np.asarray(front)
    top = np.asarray(top)
    n = min(egoview.shape[0], front.shape[0], top.shape[0])
    if n == 0:
        raise ValueError("No frames to stitch")

    h = int(panel_height or egoview.shape[1])
    w = int(panel_width or egoview.shape[2])
    out = np.empty((n, h, w * 3, 3), dtype=np.uint8)
    for t in range(n):
        panels = [
            _resize_panel(egoview[t], w, h),
            _resize_panel(front[t], w, h),
            _resize_panel(top[t], w, h),
        ]
        if draw_labels:
            panels = [_draw_label(p, label) for p, label in zip(panels, PANEL_LABELS)]
        out[t] = np.concatenate(panels, axis=1)
    return out


def _write_mp4(path: Path, frames: np.ndarray, fps: float) -> None:
    """Write (T,H,W,3) RGB uint8 frames to an MP4 file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames (T,H,W,3), got {frames.shape}")
    t, h, w, _ = frames.shape
    if t == 0:
        raise ValueError(f"No frames to write for {path}")

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
    logger.info("Wrote %s (%d frames, %dx%d @ %.1ffps)", path, t, w, h, fps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export side-by-side MP4 (egoview | front | top) from LeRobot"
    )
    parser.add_argument("--lerobot_root", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for triple_view.mp4",
    )
    parser.add_argument(
        "--output_mp4",
        type=Path,
        default=None,
        help="Explicit MP4 path (default: <output_dir>/triple_view.mp4)",
    )
    parser.add_argument("--repo_id", type=str, default="aoe/galbot_retarget")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--robot", choices=["galbot"], default="galbot")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None, help="Default: constants FPS")
    parser.add_argument("--no_labels", action="store_true", help="Disable panel labels")
    args = parser.parse_args()

    from retarget_galbot.constants import FPS
    from retarget_galbot.egoview.render import render_mujoco_views
    from retarget_galbot.robots import get_spec
    from retarget_galbot.viz.rerun import _load_episode_arrays, _to_hwc_uint8

    del args.repo_id
    spec = get_spec(args.robot)
    states, _actions, _timestamps, egoview_list = _load_episode_arrays(
        args.lerobot_root,
        args.episode_index,
        max_frames=args.max_frames,
    )
    egoview = np.stack([_to_hwc_uint8(frame) for frame in egoview_list], axis=0)
    n = min(egoview.shape[0], states.shape[0])
    egoview = egoview[:n]
    states = states[:n]

    logger.info("Rendering MuJoCo front/top for MP4 export (%d frames)...", n)
    front, top = render_mujoco_views(states, spec=spec)

    fps = float(args.fps) if args.fps is not None else float(FPS)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = Path(args.output_mp4) if args.output_mp4 else out_dir / "triple_view.mp4"

    triple = stitch_side_by_side(
        egoview,
        front,
        top,
        draw_labels=not args.no_labels,
    )
    _write_mp4(mp4_path, triple, fps)
    logger.info("Done. Side-by-side MP4: %s", mp4_path)


if __name__ == "__main__":
    main()
