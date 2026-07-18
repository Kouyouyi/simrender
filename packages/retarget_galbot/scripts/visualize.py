#!/usr/bin/env python3
"""Rerun visualization for Galbot LeRobot episodes (live MuJoCo front/top)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a Galbot LeRobot episode in Rerun with live MuJoCo views"
    )
    parser.add_argument("--lerobot_root", type=Path, required=True, help="LeRobot dataset root")
    parser.add_argument("--repo_id", type=str, default="aoe/galbot_retarget")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--robot", choices=["galbot"], default="galbot")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--rrd", type=Path, default=None, help="Save .rrd instead of spawning viewer")
    parser.add_argument("--playback_speed", type=float, default=1.0)
    args = parser.parse_args()

    from retarget_galbot.viz.rerun import visualize_lerobot_episode_rerun
    from retarget_galbot.robots import get_spec

    spec = get_spec(args.robot)
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


if __name__ == "__main__":
    main()
