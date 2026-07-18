from __future__ import annotations

import numpy as np

# Galbot head_end_effector_mount_link at the IK seed pose:
#   head x -> robot base +X (forward)
#   head y -> robot base -Z (down)
#   head z -> robot base +Y (left)
#
# OpenCV ego camera frame:
#   x right, y down, z forward
#
# Map camera motion into the head link frame so wiping forward in video drives
# the robot arms forward (+X), not to the robot side (+Y).
CAMERA_TO_HEAD_LINK = np.array(
    [
        [0.0, 0.0, 1.0],   # head x = camera forward
        [0.0, 1.0, 0.0],   # head y = camera down
        [-1.0, 0.0, 0.0],  # head z = camera left
    ],
    dtype=np.float64,
)

# OpenCV camera frame -> SAPIEN z-up display frame for input visualization only.
OPENCV_TO_DISPLAY = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def opencv_camera_to_head_link_position(position: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    if position.ndim == 1:
        return (CAMERA_TO_HEAD_LINK @ position).reshape(3)
    return position @ CAMERA_TO_HEAD_LINK.T


def opencv_camera_to_head_link_rotation(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    return CAMERA_TO_HEAD_LINK @ rotation @ CAMERA_TO_HEAD_LINK.T


def opencv_camera_to_head_link_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    rotation = _rotvec_to_matrix(np.asarray(axis_angle, dtype=np.float64))
    head_rotation = opencv_camera_to_head_link_rotation(rotation)
    return _matrix_to_rotvec(head_rotation)


def opencv_camera_to_display_position(position: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    if position.ndim == 1:
        return (OPENCV_TO_DISPLAY @ position).reshape(3)
    return position @ OPENCV_TO_DISPLAY.T


def opencv_camera_to_display_rotation(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    return OPENCV_TO_DISPLAY @ rotation @ OPENCV_TO_DISPLAY.T


def opencv_camera_to_display_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    rotation = _rotvec_to_matrix(np.asarray(axis_angle, dtype=np.float64))
    display_rotation = opencv_camera_to_display_rotation(rotation)
    return _matrix_to_rotvec(display_rotation)


# Backward-compatible aliases used by older call sites.
opencv_camera_to_head_position = opencv_camera_to_head_link_position
opencv_camera_to_head_rotation = opencv_camera_to_head_link_rotation
opencv_camera_to_head_axis_angle = opencv_camera_to_head_link_axis_angle


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * skew
        + (1.0 - np.cos(theta)) * (skew @ skew)
    )


def _matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    rotvec = np.zeros(3, dtype=np.float64)
    cos_angle = (float(np.trace(rotation)) - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-12:
        return rotvec
    skew = rotation - rotation.T
    rotvec = np.array(
        [skew[2, 1], skew[0, 2], skew[1, 0]],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(rotvec))
    if norm < 1e-12:
        return rotvec
    return rotvec / norm * angle
