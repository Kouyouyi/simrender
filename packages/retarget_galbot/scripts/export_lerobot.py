#!/usr/bin/env python3
"""Export Galbot retarget results to LeRobot v2.1 and/or visualize with Rerun."""

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Galbot egoview overlay episodes to LeRobot + Rerun"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode_dir", type=Path, help="Single AoE segment")
    group.add_argument("--data_root", type=Path, help="Root containing AoE episodes")

    parser.add_argument("--robot", choices=["galbot"], default="galbot")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--actions", type=Path, default=None, help="Existing *_actions.npy")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--video_start_frame",
        type=int,
        default=None,
        help="Ego video start frame; default auto-aligns by frame-count difference.",
    )

    parser.add_argument(
        "--lerobot_root",
        type=Path,
        default=Path("./output/lerobot_galbot"),
        help="LeRobot dataset root directory",
    )
    parser.add_argument("--repo_id", type=str, default="aoe/galbot_retarget")
    parser.add_argument("--task", type=str, default="", help="Override task string")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing --lerobot_root before creating a new dataset",
    )
    parser.add_argument("--skip_export", action="store_true", help="Only visualize existing dataset")
    parser.add_argument("--visualize", action="store_true", help="Open Rerun after export")
    parser.add_argument("--rrd", type=Path, default=None, help="Save .rrd instead of spawning viewer")
    parser.add_argument("--episode_index", type=int, default=0, help="Episode index for Rerun")
    parser.add_argument("--playback_speed", type=float, default=1.0)
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    from retarget_galbot.dataset import add_episode_from_arrays, create_lerobot_dataset
    from retarget_galbot.egoview import render_egoview_frames
    from retarget_galbot.pipeline import RetargetSession, discover_episodes
    from retarget_galbot.robots import get_spec

    spec = get_spec(args.robot)

    if args.skip_export:
        if not args.visualize:
            raise ValueError("--skip_export requires --visualize")
        from retarget_galbot.viz.rerun import visualize_lerobot_episode_rerun

        visualize_lerobot_episode_rerun(
            root=args.lerobot_root,
            repo_id=args.repo_id,
            episode_index=args.episode_index,
            spec=spec,
            save_rrd=args.rrd,
            playback_speed=args.playback_speed,
            max_frames=args.max_frames,
        )
        logger.info("Done.")
        return

    if args.episode_dir:
        episodes = [args.episode_dir]
    else:
        episodes = discover_episodes(args.data_root)
        if args.max_episodes:
            episodes = episodes[: args.max_episodes]

    dataset = create_lerobot_dataset(
        repo_id=args.repo_id,
        root=args.lerobot_root,
        spec=spec,
        overwrite=args.overwrite,
    )
    session = RetargetSession(spec, config_path=args.config)

    for ep_idx, ep_dir in enumerate(episodes):
        ep_dir = Path(ep_dir)
        seg_name = _segment_name(ep_dir)
        logger.info("[%d/%d] %s", ep_idx + 1, len(episodes), ep_dir)

        actions = _load_or_retarget_actions(
            ep_dir=ep_dir,
            actions_path=args.actions if len(episodes) == 1 else None,
            session=session,
            max_frames=args.max_frames,
            stride=args.stride,
        )
        ego_video = _find_ego_video(ep_dir)
        if ego_video is None:
            raise FileNotFoundError(f"No ego video under {ep_dir}")

        logger.info("Building egoview overlay for LeRobot...")
        composed = render_egoview_frames(
            actions,
            ego_video,
            spec=spec,
            episode_dir=ep_dir,
            video_start_frame=args.video_start_frame,
        )
        task = args.task or _task_from_annotations(ep_dir)
        add_episode_from_arrays(
            dataset,
            composed_frames=composed,
            actions=actions,
            task=task,
        )
        logger.info("Exported episode %s -> %s", seg_name, args.lerobot_root)

    if args.visualize:
        from retarget_galbot.viz.rerun import visualize_lerobot_episode_rerun

        visualize_lerobot_episode_rerun(
            root=args.lerobot_root,
            repo_id=args.repo_id,
            episode_index=args.episode_index,
            spec=spec,
            save_rrd=args.rrd,
            playback_speed=args.playback_speed,
            max_frames=args.max_frames,
        )

    logger.info("Done.")


def _load_or_retarget_actions(
    *,
    ep_dir: Path,
    actions_path: Path | None,
    session,
    max_frames: int | None,
    stride: int,
) -> np.ndarray:
    if actions_path is not None and Path(actions_path).exists():
        actions = np.load(actions_path)
        if max_frames is not None:
            actions = actions[:max_frames]
        return np.asarray(actions, dtype=np.float32)

    result = session.retarget(ep_dir, max_frames=max_frames, stride=stride)
    return np.asarray(result.actions, dtype=np.float32)


def _task_from_annotations(episode_dir: Path) -> str:
    from retarget_galbot.galaxea.annotations import load_action_segments

    segments = load_action_segments(episode_dir)
    verbs: list[str] = []
    for seg in segments:
        for v in seg.verbs:
            if v and v not in verbs:
                verbs.append(v)
    return " ".join(verbs)


def _segment_name(path: Path) -> str:
    path = Path(path)
    if path.name == "ego_hands_reconstruction":
        return path.parent.parent.name if path.parent.name == "ego_process" else path.parent.name
    if path.name == "hands.npz":
        return _segment_name(path.parent)
    return path.name


def _find_ego_video(episode_dir: Path) -> Path | None:
    for subdir in ("ego_process/ego_undistorted_video", "ego_undistorted_video"):
        vdir = episode_dir / subdir
        if vdir.exists():
            for f in sorted(vdir.iterdir()):
                if f.suffix == ".mp4":
                    return f
    raw = episode_dir / "raw_video.mp4"
    return raw if raw.exists() else None


if __name__ == "__main__":
    main()
