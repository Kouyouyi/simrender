#!/usr/bin/env python3
"""Retarget AoE egocentric MANO data to Galbot actions and LeRobot datasets."""

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
    parser = argparse.ArgumentParser(description="AoE/MANO -> Galbot retargeting pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode_dir", type=Path, help="Single AoE segment or ego_hands_reconstruction dir")
    group.add_argument("--data_root", type=Path, help="Root dir containing AoE episodes")

    parser.add_argument("--robot", choices=["galbot"], default="galbot")
    parser.add_argument("--config", type=Path, default=None, help="Galbot retargeting YAML")
    parser.add_argument("--output_dir", type=Path, default=None, help="Default: ./output/galbot")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames per episode")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes in batch mode")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--video_start_frame",
        type=int,
        default=None,
        help="Start frame in ego video for overlay alignment; default auto-aligns by frame-count difference.",
    )
    parser.add_argument("--write_pickle", action="store_true", help="Write Galbot MVP pickle output")
    parser.add_argument("--write_jsonl", action="store_true", help="Write Galbot MVP JSONL output")

    parser.add_argument(
        "--write_lerobot",
        action="store_true",
        help="Export egoview overlay frames + qpos to a LeRobot v2.1 dataset",
    )
    parser.add_argument(
        "--lerobot_root",
        type=Path,
        default=None,
        help="LeRobot dataset root (default: <output_dir>/lerobot)",
    )
    parser.add_argument("--repo_id", type=str, default="aoe/galbot_retarget")
    parser.add_argument("--task", type=str, default="", help="Override LeRobot task string")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing LeRobot root before creating a new dataset",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Rerun visualization with live MuJoCo front/top renders (requires --write_lerobot)",
    )
    parser.add_argument("--rrd", type=Path, default=None, help="Save Rerun .rrd instead of spawning viewer")
    parser.add_argument("--playback_speed", type=float, default=1.0)
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.visualize and not args.write_lerobot:
        raise ValueError("--visualize requires --write_lerobot (Rerun reads the LeRobot dataset)")

    from retarget_galbot.pipeline import RetargetSession, discover_episodes
    from retarget_galbot.robots import get_spec

    spec = get_spec(args.robot)
    output_dir = args.output_dir or Path("./output/galbot")
    output_dir.mkdir(parents=True, exist_ok=True)
    lerobot_root = args.lerobot_root or (output_dir / "lerobot")

    if args.episode_dir:
        episodes = [args.episode_dir]
    else:
        episodes = discover_episodes(args.data_root)
        if args.max_episodes:
            episodes = episodes[: args.max_episodes]

    session = RetargetSession(spec, config_path=args.config)
    dataset = None
    if args.write_lerobot:
        from retarget_galbot.dataset import create_lerobot_dataset

        dataset = create_lerobot_dataset(
            repo_id=args.repo_id,
            root=lerobot_root,
            spec=spec,
            overwrite=args.overwrite,
        )

    for ep_idx, ep_dir in enumerate(episodes):
        seg_name = _segment_name(ep_dir)
        npy_path = output_dir / f"{seg_name}_actions.npy"
        if args.skip_existing and npy_path.exists() and not args.write_lerobot:
            logger.info("Skipping existing: %s", seg_name)
            continue

        logger.info("[%d/%d] Processing: %s", ep_idx + 1, len(episodes), ep_dir)
        pickle_path = output_dir / f"{seg_name}_galbot.pkl" if args.write_pickle else None
        jsonl_path = output_dir / f"{seg_name}_galbot.jsonl" if args.write_jsonl else None

        result = session.retarget(
            ep_dir,
            max_frames=args.max_frames,
            stride=args.stride,
            pickle_output=pickle_path,
            jsonl_output=jsonl_path,
        )

        np.save(npy_path, result.actions)
        logger.info("  -> %s (T=%d, dim=%d)", npy_path, *result.actions.shape)

        if args.write_lerobot:
            from retarget_galbot.dataset import add_episode_from_arrays
            from retarget_galbot.egoview import render_egoview_frames

            ego_video = _find_ego_video(result.meta.episode_dir)
            if ego_video is None:
                raise FileNotFoundError(f"No ego video under {result.meta.episode_dir}")
            logger.info("Building egoview overlay for LeRobot...")
            composed = render_egoview_frames(
                result.actions,
                ego_video,
                spec=spec,
                episode_dir=result.meta.episode_dir,
                video_start_frame=args.video_start_frame,
            )
            task = args.task or _task_from_annotations(result.meta.episode_dir)
            assert dataset is not None
            add_episode_from_arrays(
                dataset,
                composed_frames=composed,
                actions=result.actions,
                task=task,
            )
            logger.info("  -> LeRobot episode written under %s", lerobot_root)

    if args.visualize:
        from retarget_galbot.viz.rerun import visualize_lerobot_episode_rerun

        visualize_lerobot_episode_rerun(
            root=lerobot_root,
            repo_id=args.repo_id,
            episode_index=0,
            spec=spec,
            save_rrd=args.rrd,
            playback_speed=args.playback_speed,
            max_frames=args.max_frames,
        )

    logger.info("Done. Output in: %s", output_dir)


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
