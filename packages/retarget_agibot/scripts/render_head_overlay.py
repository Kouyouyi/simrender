#!/usr/bin/env python3
"""Replay AgiBot G1 and overlay it on the recorded head-camera video."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE_ROOT))

from retarget_agibot.head_overlay import render_head_overlay_video  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("agibot-head-overlay")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=WORKSPACE_ROOT / "datasets/agibot",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=WORKSPACE_ROOT / "assets/robots/G1_v2.3/G1_120s/G1_120s.urdf",
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
        default=WORKSPACE_ROOT / "output/agibot/g1_120s_head_camera_depth_overlay.mp4",
    )
    parser.add_argument(
        "--render-only-output",
        type=Path,
        default=WORKSPACE_ROOT / "output/agibot/g1_120s_head_camera_depth.mp4",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.86)
    parser.add_argument("--feather-sigma", type=float, default=1.0)
    parser.add_argument("--joint-source", choices=("state", "action"), default="state")
    parser.add_argument("--render-mode", choices=("rgb", "depth"), default="depth")
    args = parser.parse_args()

    stats, calibration, alignment = render_head_overlay_video(
        args.dataset_dir,
        args.output,
        urdf_path=args.urdf,
        visual_mesh_dir=args.visual_mesh_dir,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        stride=args.stride,
        alpha=args.alpha,
        feather_sigma=args.feather_sigma,
        joint_source=args.joint_source,
        render_mode=args.render_mode,
        render_only_output_path=args.render_only_output,
    )
    metadata = {
        "output": str(args.output.expanduser().resolve()),
        "dataset_dir": str(args.dataset_dir.expanduser().resolve()),
        "source_video": str(
            (args.dataset_dir / "observations/head_color.mp4").expanduser().resolve()
        ),
        "trajectory": str(
            (args.dataset_dir / "proprio_stats/proprio_stats.h5").expanduser().resolve()
        ),
        "urdf": str(args.urdf.expanduser().resolve()),
        "visual_mesh_dir": str(args.visual_mesh_dir.expanduser().resolve()),
        "camera_intrinsic": str(calibration.intrinsic_path),
        "camera_extrinsic_aligned": str(calibration.extrinsic_path),
        "camera_model": {
            "width": calibration.width,
            "height": calibration.height,
            "fps": calibration.fps,
            "K": calibration.intrinsic.tolist(),
            "distortion_opencv_k1_k2_p1_p2_k3": calibration.distortion.tolist(),
            "native_render_height": calibration.native_render_height,
            "native_fovy_degrees": calibration.native_fovy_degrees,
        },
        "model_to_dataset_alignment": {
            "rotation": alignment.rotation_data_from_model.tolist(),
            "translation": alignment.translation_data_from_model.tolist(),
            "rmse_m": alignment.rmse_m,
            "sample_count": alignment.sample_count,
        },
        "frame_count": stats.frame_count,
        "fps": stats.fps,
        "start_frame": args.start_frame,
        "stride": args.stride,
        "joint_source": args.joint_source,
        "render_mode": args.render_mode,
        "render_only_output": str(args.render_only_output.expanduser().resolve()),
        "depth_colormap": (
            {
                "name": "Vision Banana metric depth",
                "paper": "Image Generators are Generalist Vision Learners (arXiv:2604.20329)",
                "mapping": "piecewise focused metric mapping over the Vision Banana RGB path",
                "profile": "AgiBot projected-arm range with 25 percent margin per side",
                "rgb_path": "black-red-yellow-green-cyan-blue-magenta-white",
                "observed_robust_range_m": [0.2332, 0.6880],
                "focus_range_m": [0.12, 0.80],
                "focus_margin": "25 percent of observed span on each side, rounded",
                "vertex_depths_m": [0.0, 0.12, 0.26, 0.39, 0.53, 0.66, 0.80, "inf"],
                "inverse_function": "focused_rgb_to_depth",
                "raw_rgb8_focus_range_roundtrip": {
                    "mean_absolute_error_mm": 0.133,
                    "percentile_95_absolute_error_mm": 0.253,
                    "percentile_99_absolute_error_mm": 0.264,
                    "maximum_absolute_error_mm": 0.267,
                },
                "h264_decode_qc": {
                    "sample_frames": [0, 500, 1000, 1500, 2000, 2524],
                    "eroded_arm_pixels": 174699,
                    "mean_absolute_error_mm": 1.057,
                    "median_absolute_error_mm": 0.811,
                    "percentile_95_absolute_error_mm": 2.722,
                    "percentile_99_absolute_error_mm": 5.019,
                    "maximum_absolute_error_mm": 44.948,
                },
            }
            if args.render_mode == "depth"
            else None
        ),
        "alpha": args.alpha,
        "feather_sigma": args.feather_sigma,
        "first_camera_position_model": stats.first_camera_position_model,
        "runtime_clip_counts": stats.runtime_clip_counts,
        "limitations": [
            "The original robot remains in the source video; this is an alpha overlay, not inpainting.",
            "No source-scene depth is available, so held objects cannot occlude simulated geometry correctly.",
            "MuJoCo renders a pinhole image, then the recorded principal point, anisotropic resize, and lens distortion are applied in OpenCV.",
        ],
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", args.output.expanduser().resolve())
    LOGGER.info("Metadata: %s", metadata_path.expanduser().resolve())


if __name__ == "__main__":
    main()
