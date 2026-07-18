# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""LeRobot dataset export for Galbot retarget episodes.

Uses the official ``LeRobotDataset`` API.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from retarget_galbot.constants import FPS, VIDEO_HEIGHT, VIDEO_KEY, VIDEO_WIDTH
from retarget_galbot.robots import RobotSpec

logger = logging.getLogger(__name__)


def _ensure_lerobot_import() -> None:
    try:
        import lerobot  # noqa: F401
        return
    except ImportError:
        pass
    # Optional: PYTHONPATH or LEROBOT_SRC pointing at a local LeRobot checkout.
    src_raw = os.environ.get("LEROBOT_SRC")
    if src_raw:
        src = Path(src_raw)
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
    import lerobot  # noqa: F401


def build_galbot_features(joint_names: Sequence[str]) -> dict:
    """Feature schema for Galbot retarget episodes."""
    dim = len(joint_names)
    names = list(joint_names)
    return {
        VIDEO_KEY: {
            "dtype": "video",
            "shape": (VIDEO_HEIGHT, VIDEO_WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (dim,),
            "names": names,
        },
        "action": {
            "dtype": "float32",
            "shape": (dim,),
            "names": names,
        },
    }


def actions_to_state_action(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map qpos trajectory to (state[t], action[t]=qpos[t+1])."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected actions (T, D), got {actions.shape}")
    if actions.shape[0] < 1:
        raise ValueError("actions is empty")
    state = actions.copy()
    action = np.empty_like(actions)
    if actions.shape[0] == 1:
        action[:] = actions
    else:
        action[:-1] = actions[1:]
        action[-1] = actions[-1]
    return state, action


def _is_complete_lerobot_dataset(root: Path) -> bool:
    """True only when local metadata is complete enough to open offline.

    A half-written root (e.g. only ``meta/info.json``) must NOT be treated as
    valid: LeRobot would otherwise fall back to HuggingFace Hub download.
    """
    meta = root / "meta"
    required = (
        meta / "info.json",
        meta / "tasks.jsonl",
        meta / "episodes.jsonl",
        meta / "episodes_stats.jsonl",
    )
    return all(path.is_file() for path in required)


def _remove_lerobot_root(root: Path) -> None:
    import shutil

    try:
        shutil.rmtree(root)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot remove LeRobot root {root} (permission denied). "
            f"Delete it manually, e.g. `sudo rm -rf {root}`, then retry."
        ) from exc


def create_lerobot_dataset(
    *,
    repo_id: str,
    root: str | Path,
    spec: RobotSpec,
    fps: int = FPS,
    video_backend: str = "pyav",
    image_writer_threads: int = 4,
    overwrite: bool = False,
):
    """Create a new LeRobot dataset, or open an existing one for appending.

    Args:
        overwrite: If True and ``root`` already exists, delete it and recreate.
            If False and ``root`` is a complete LeRobot dataset, open it and append.
            Incomplete/corrupt roots are removed and recreated automatically.
    """
    _ensure_lerobot_import()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    if root.exists() and any(root.iterdir()):
        if overwrite:
            logger.warning("Removing existing LeRobot root (--overwrite): %s", root)
            _remove_lerobot_root(root)
        elif _is_complete_lerobot_dataset(root):
            dataset = LeRobotDataset(
                repo_id,
                root=root,
                video_backend=video_backend,
            )
            dataset.episode_buffer = dataset.create_episode_buffer()
            dataset.start_image_writer(
                num_processes=0,
                num_threads=image_writer_threads,
            )
            logger.info(
                "Opened existing LeRobot dataset at %s "
                "(episodes=%d, will append)",
                root,
                dataset.meta.total_episodes,
            )
            return dataset
        else:
            logger.warning(
                "LeRobot root is incomplete/corrupt (missing meta files): %s. "
                "Removing and recreating locally (no Hub download).",
                root,
            )
            _remove_lerobot_root(root)

    root.parent.mkdir(parents=True, exist_ok=True)
    features = build_galbot_features(spec.action_joint_names)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type=spec.robot_type,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=image_writer_threads,
        video_backend=video_backend,
    )
    logger.info("Created LeRobot dataset at %s (repo_id=%s)", root, repo_id)
    return dataset


def add_episode_from_arrays(
    dataset,
    *,
    composed_frames: np.ndarray,
    actions: np.ndarray,
    task: str = "",
    fps: int = FPS,
) -> int:
    """Append one episode from egoview overlay frames + qpos actions.

    Returns number of frames written.
    """
    frames = np.asarray(composed_frames)
    actions = np.asarray(actions, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected composed_frames (T,H,W,3), got {frames.shape}")
    if actions.shape[0] != frames.shape[0]:
        raise ValueError(
            f"Frame/action length mismatch: {frames.shape[0]} vs {actions.shape[0]}"
        )
    if frames.shape[1:3] != (VIDEO_HEIGHT, VIDEO_WIDTH):
        import cv2

        resized = np.empty(
            (frames.shape[0], VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8
        )
        for i, frame in enumerate(frames):
            resized[i] = cv2.resize(
                frame, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_AREA
            )
        frames = resized

    state, action = actions_to_state_action(actions)
    task = task or ""
    for t in range(frames.shape[0]):
        dataset.add_frame(
            {
                VIDEO_KEY: frames[t],
                "observation.state": state[t],
                "action": action[t],
            },
            task=task,
            timestamp=float(t) / float(fps),
        )
    dataset.save_episode()
    logger.info("Saved LeRobot episode with %d frames (task=%r)", frames.shape[0], task)
    return int(frames.shape[0])


def export_episode_lerobot(
    *,
    root: str | Path,
    repo_id: str,
    spec: RobotSpec,
    composed_frames: np.ndarray,
    actions: np.ndarray,
    task: str = "",
    fps: int = FPS,
    dataset=None,
    video_backend: str = "pyav",
    overwrite: bool = False,
):
    """Create-or-reuse dataset and write one episode."""
    if dataset is None:
        dataset = create_lerobot_dataset(
            repo_id=repo_id,
            root=root,
            spec=spec,
            fps=fps,
            video_backend=video_backend,
            overwrite=overwrite,
        )
    add_episode_from_arrays(
        dataset,
        composed_frames=composed_frames,
        actions=actions,
        task=task,
        fps=fps,
    )
    return dataset
