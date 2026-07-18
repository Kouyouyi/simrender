from __future__ import annotations

import argparse
from pathlib import Path

from .annotations import load_action_segments
from .bimanual import BimanualDexRetargeter
from .config import load_config
from .ego_data import load_ego_hand_sequence
from .features import extract_hand_features
from .io import save_jsonl_output, save_pickle_output
from .urdf_utils import parse_movable_joints


def _run_dex_bimanual(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if config.backend != "dex_bimanual":
        raise ValueError(
            "Only backend=dex_bimanual is supported. "
            "Single-hand routes were removed from this MVP."
        )
    if config.source_space != "camera":
        raise ValueError(
            "Head-anchored retargeting requires source_space=camera so wrist poses "
            "stay in the ego camera frame before mapping to the robot head."
        )
    reference_frame = str(config.raw.get("reference_frame", "head")).lower()
    if reference_frame != "head":
        raise ValueError("Only reference_frame=head is supported in the current MVP.")

    reconstruction = args.reconstruction or args.segment
    if reconstruction is None:
        raise ValueError("Pass --reconstruction /path/to/ego_hands_reconstruction or hands.npz")

    joints = parse_movable_joints(config.urdf_path)
    left_hand = load_ego_hand_sequence(
        reconstruction,
        hand="left",
        source_space=config.source_space,
        fps=config.fps,
    )
    right_hand = load_ego_hand_sequence(
        reconstruction,
        hand="right",
        source_space=config.source_space,
        fps=config.fps,
    )
    left_sequence = extract_hand_features(
        left_hand, smooth_alpha=config.feature_smooth_alpha
    )
    right_sequence = extract_hand_features(
        right_hand, smooth_alpha=config.feature_smooth_alpha
    )
    segments = load_action_segments(left_hand.segment_dir)

    retargeter = BimanualDexRetargeter(config, joints)
    frames = retargeter.retarget(
        left_sequence,
        right_sequence,
        segments,
        max_frames=args.max_frames,
        stride=args.stride,
    )

    meta_data = {
        "backend": config.backend,
        "robot_name": config.robot_name,
        "robot_urdf": str(config.urdf_path),
        "config_path": str(config.path),
        "source_segment": str(left_hand.segment_dir),
        "source_reconstruction_dir": str(left_hand.reconstruction_dir),
        "left_source_path": left_sequence.source_path,
        "right_source_path": right_sequence.source_path,
        "left_source_type": left_sequence.source_type,
        "right_source_type": right_sequence.source_type,
        "source_space": config.source_space,
        "reference_frame": reference_frame,
        "head_link_name": str(config.raw.get("head_link_name", "head_end_effector_mount_link")),
        "position_mapping": str(config.raw.get("position_mapping", "absolute")).lower(),
        "input_frame": "camera_head_link",
        "hands": ["left", "right"],
        "fps": config.fps,
        "joint_names": list(retargeter.joint_names),
        "target_joint_names": list(config.raw.get("target_joint_names", [])),
        "target_link_names": list(config.raw.get("target_link_names", [])),
        "dof": len(retargeter.joint_names),
        "controlled_dof": len(config.raw.get("target_joint_names", [])),
        "frame_count": len(frames),
        "feature_fields": [
            "wrist_position",
            "palm_position",
            "palm_rotation",
            "finger_curl",
            "pinch_score",
            "hand_openness",
        ],
    }
    save_pickle_output(args.output, meta_data, frames)
    if args.jsonl_output:
        save_jsonl_output(args.jsonl_output, frames)
    print(f"saved {len(frames)} bimanual frames to {Path(args.output).resolve()}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retarget ego_hands_reconstruction MANO data to Galbot."
    )
    parser.add_argument("--config", required=True, help="Retargeting YAML config.")
    parser.add_argument(
        "--reconstruction",
        default=None,
        help=(
            "Path to ego_hands_reconstruction/, hands.npz, or a segment containing "
            "ego_process/ego_hands_reconstruction/hands.npz."
        ),
    )
    parser.add_argument(
        "--segment",
        default=None,
        help="Backward-compatible alias for --reconstruction.",
    )
    parser.add_argument("--output", required=True, help="Output pickle path.")
    parser.add_argument("--jsonl-output", default=None, help="Optional JSONL output path.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for quick runs.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    _run_dex_bimanual(args)


if __name__ == "__main__":
    main()
