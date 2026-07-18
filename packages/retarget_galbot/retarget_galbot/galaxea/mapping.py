from __future__ import annotations

import numpy as np


def scale_wrist_absolute(
    wrist_in_head: np.ndarray,
    scale: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    wrist = np.asarray(wrist_in_head, dtype=np.float64)
    scale_vec = np.asarray(scale, dtype=np.float64)
    return wrist * scale_vec


def scale_wrist_delta(
    wrist_in_head: np.ndarray,
    wrist_reference: np.ndarray,
    scale: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    delta = np.asarray(wrist_in_head, dtype=np.float64) - np.asarray(
        wrist_reference, dtype=np.float64
    )
    scale_vec = np.asarray(scale, dtype=np.float64)
    return delta * scale_vec


def clip_workspace(
    position: np.ndarray,
    workspace_min: tuple[float, float, float] | np.ndarray,
    workspace_max: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    return np.clip(
        np.asarray(position, dtype=np.float64),
        np.asarray(workspace_min, dtype=np.float64),
        np.asarray(workspace_max, dtype=np.float64),
    )


def mapped_wrist_in_head_absolute(
    wrist_in_head: np.ndarray,
    scale: tuple[float, float, float] | np.ndarray,
    workspace_min: tuple[float, float, float] | np.ndarray,
    workspace_max: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Camera-origin wrist offset scaled into the robot-head frame."""
    return clip_workspace(scale_wrist_absolute(wrist_in_head, scale), workspace_min, workspace_max)


def mapped_wrist_delta_in_head(
    wrist_in_head: np.ndarray,
    wrist_reference: np.ndarray,
    scale: tuple[float, float, float] | np.ndarray,
    workspace_min: tuple[float, float, float] | np.ndarray,
    workspace_max: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    return clip_workspace(
        scale_wrist_delta(wrist_in_head, wrist_reference, scale),
        workspace_min,
        workspace_max,
    )


def head_frame_to_robot_base(
    head_translation: np.ndarray,
    head_rotation: np.ndarray,
    position_in_head: np.ndarray,
) -> np.ndarray:
    return head_translation + head_rotation @ np.asarray(position_in_head, dtype=np.float64)
