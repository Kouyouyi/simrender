# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Rerun visualization for Galbot LeRobot episodes with live MuJoCo views.

Logs three image streams per frame:
  - composed egoview overlay
  - MuJoCo front view
  - MuJoCo top-down view
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from retarget_galbot.constants import FPS, VIDEO_KEY
from retarget_galbot.egoview.render import render_mujoco_views
from retarget_galbot.robots import RobotSpec, get_spec

logger = logging.getLogger(__name__)

# Rerun entity paths for the three primary image panels.
EGOVIEW_ENTITY = VIDEO_KEY  # observation.images.egoview (composed overlay)
MUJOCO_FRONT_ENTITY = "mujoco/front"
MUJOCO_TOP_ENTITY = "mujoco/top"


def _to_hwc_uint8(image) -> np.ndarray:
    arr = np.asarray(image)
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return arr


def _load_episode_arrays(
    root: Path,
    episode_index: int,
    *,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Load state/action/timestamps/egoview without torchvision VideoReader."""
    import av
    import pandas as pd

    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    chunks_size = int(info.get("chunks_size", 1000))
    chunk_idx = episode_index // chunks_size
    parquet_path = (
        root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_index:06d}.parquet"
    )
    video_path = (
        root
        / "videos"
        / f"chunk-{chunk_idx:03d}"
        / VIDEO_KEY
        / f"episode_{episode_index:06d}.mp4"
    )
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet: {parquet_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")

    df = pd.read_parquet(parquet_path)
    if max_frames is not None:
        df = df.iloc[: int(max_frames)]

    states = np.stack([np.asarray(x, dtype=np.float32) for x in df["observation.state"]], axis=0)
    actions = np.stack([np.asarray(x, dtype=np.float32) for x in df["action"]], axis=0)
    timestamps = np.asarray(df["timestamp"], dtype=np.float64)

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    egoview: list[np.ndarray] = []
    for frame in container.decode(stream):
        egoview.append(frame.to_ndarray(format="rgb24"))
        if max_frames is not None and len(egoview) >= int(max_frames):
            break
    container.close()

    n = min(len(egoview), states.shape[0], actions.shape[0])
    if n == 0:
        raise RuntimeError(f"Empty episode data under {root} ep={episode_index}")
    if len(egoview) != states.shape[0]:
        logger.warning(
            "Video/parquet length mismatch (%d vs %d); truncating to %d",
            len(egoview),
            states.shape[0],
            n,
        )
    return states[:n], actions[:n], timestamps[:n], egoview[:n]


def visualize_lerobot_episode_rerun(
    *,
    root: str | Path,
    repo_id: str,
    episode_index: int = 0,
    spec: RobotSpec | None = None,
    fps: int = FPS,
    mode: str = "local",
    save_rrd: str | Path | None = None,
    playback_speed: float = 1.0,
    max_frames: int | None = None,
    video_backend: str = "pyav",
) -> None:
    """Load one LeRobot episode and log composed + front + top views to Rerun.

    Image panels:
      - ``observation.images.egoview`` — Stage/egoview composed overlay
      - ``mujoco/front`` — MuJoCo front camera (live from ``observation.state``)
      - ``mujoco/top`` — MuJoCo top-down camera (live from ``observation.state``)
    """
    del video_backend  # parquet+av loader avoids torchvision VideoReader
    import rerun as rr

    if spec is None:
        spec = get_spec("galbot")

    root = Path(root)
    states, actions, timestamps, egoview = _load_episode_arrays(
        root, episode_index, max_frames=max_frames
    )
    n = states.shape[0]

    logger.info(
        "Rendering MuJoCo front/top views for episode %d (%d frames)...",
        episode_index,
        n,
    )
    front_frames, top_frames = render_mujoco_views(states, spec=spec)

    app_id = f"{repo_id}/episode_{episode_index}"
    spawn = mode == "local" and save_rrd is None
    rr.init(app_id, spawn=spawn)
    if save_rrd is not None:
        save_path = Path(save_rrd)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(save_path))
        logger.info("Recording Rerun log to %s", save_path)

    dt = (1.0 / float(fps)) / max(float(playback_speed), 1e-6)
    joint_names = list(spec.action_joint_names)
    for t in range(n):
        rr.set_time("frame_index", sequence=t)
        rr.set_time("timestamp", duration=float(timestamps[t]))

        rr.log(EGOVIEW_ENTITY, rr.Image(_to_hwc_uint8(egoview[t])))
        rr.log(MUJOCO_FRONT_ENTITY, rr.Image(front_frames[t]))
        rr.log(MUJOCO_TOP_ENTITY, rr.Image(top_frames[t]))

        for dim_idx, val in enumerate(actions[t]):
            name = joint_names[dim_idx] if dim_idx < len(joint_names) else str(dim_idx)
            rr.log(f"action/{name}", rr.Scalars(float(val)))
        for dim_idx, val in enumerate(states[t]):
            name = joint_names[dim_idx] if dim_idx < len(joint_names) else str(dim_idx)
            rr.log(f"state/{name}", rr.Scalars(float(val)))

        if spawn and playback_speed > 0:
            time.sleep(dt)

    logger.info("Rerun visualization finished for episode %d", episode_index)


def visualize_arrays_rerun(
    *,
    composed_frames: np.ndarray,
    actions: np.ndarray,
    spec: RobotSpec | None = None,
    fps: int = FPS,
    app_id: str = "galbot_retarget",
    save_rrd: str | Path | None = None,
    playback_speed: float = 1.0,
) -> None:
    """Rerun visualize in-memory composed egoview + front/top MuJoCo views."""
    import rerun as rr

    if spec is None:
        spec = get_spec("galbot")
    frames = np.asarray(composed_frames)
    qpos = np.asarray(actions, dtype=np.float32)
    if frames.shape[0] != qpos.shape[0]:
        raise ValueError("composed_frames and actions length mismatch")

    front_frames, top_frames = render_mujoco_views(qpos, spec=spec)
    spawn = save_rrd is None
    rr.init(app_id, spawn=spawn)
    if save_rrd is not None:
        Path(save_rrd).parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(save_rrd))

    dt = (1.0 / float(fps)) / max(float(playback_speed), 1e-6)
    joint_names = list(spec.action_joint_names)
    for t in range(qpos.shape[0]):
        rr.set_time("frame_index", sequence=t)
        rr.set_time("timestamp", duration=float(t) / float(fps))
        rr.log(EGOVIEW_ENTITY, rr.Image(_to_hwc_uint8(frames[t])))
        rr.log(MUJOCO_FRONT_ENTITY, rr.Image(front_frames[t]))
        rr.log(MUJOCO_TOP_ENTITY, rr.Image(top_frames[t]))
        for dim_idx, val in enumerate(qpos[t]):
            name = joint_names[dim_idx] if dim_idx < len(joint_names) else str(dim_idx)
            rr.log(f"state/{name}", rr.Scalars(float(val)))
        if spawn and playback_speed > 0:
            time.sleep(dt)
