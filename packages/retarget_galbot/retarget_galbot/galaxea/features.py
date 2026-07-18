from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coordinates import (
    opencv_camera_to_head_link_axis_angle,
    opencv_camera_to_head_link_position,
)
from .ego_data import EgoHandSequence


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_POSE_SLICES = {
    "thumb": slice(0, 9),
    "index": slice(9, 18),
    "middle": slice(18, 27),
    "ring": slice(27, 36),
    "pinky": slice(36, 45),
}
FINGER_TIP_IDS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}
FINGER_MCP_IDS = {
    "thumb": 2,
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}
# Palm center ≈ mean of wrist + four finger MCPs (OpenPose indices).
PALM_CENTER_KEYPOINT_IDS = (0, 5, 9, 13, 17)
KEYPOINT_OPENNESS_CLOSE_THRESHOLD = 0.75


@dataclass(frozen=True)
class HandFeatures:
    wrist_position: np.ndarray
    palm_position: np.ndarray
    palm_rotation: np.ndarray
    finger_curl: dict[str, float]
    pinch_score: float
    hand_openness: float
    gripper_closed: bool


@dataclass(frozen=True)
class HandFeatureSequence:
    features: list[HandFeatures]
    valid: np.ndarray
    fps: float
    source_type: str
    source_path: str

    @property
    def frame_count(self) -> int:
        return len(self.features)


def extract_hand_features(
    sequence: EgoHandSequence,
    smooth_alpha: float = 0.35,
    keypoint_close_threshold: float = KEYPOINT_OPENNESS_CLOSE_THRESHOLD,
) -> HandFeatureSequence:
    features = []
    for frame, (trans, rot, pose) in enumerate(
        zip(
            sequence.trans,
            sequence.rot_axis_angle,
            sequence.hand_pose_axis_angle,
            strict=True,
        )
    ):
        keypoints = None
        if sequence.keypoints is not None and frame < len(sequence.keypoints):
            keypoints = _keypoints_in_head_frame(sequence, sequence.keypoints[frame])
        features.append(
            _extract_single_frame(
                _wrist_position_in_head_frame(sequence, trans),
                _palm_axis_angle_in_head_frame(sequence, rot),
                pose,
                keypoints,
                keypoint_close_threshold,
            )
        )
    smoothed = _smooth_features(features, smooth_alpha)
    smoothed = _apply_discrete_gripper_threshold(
        smoothed,
        keypoint_close_threshold,
    )
    return HandFeatureSequence(
        features=smoothed,
        valid=sequence.valid,
        fps=sequence.fps,
        source_type="ego_hands_reconstruction_head_link",
        source_path=str(sequence.source_path),
    )


def _wrist_position_in_head_frame(sequence: EgoHandSequence, wrist_position: np.ndarray) -> np.ndarray:
    wrist = np.asarray(wrist_position, dtype=np.float64)
    if sequence.source_space != "camera":
        return wrist
    return opencv_camera_to_head_link_position(wrist)


def _palm_axis_angle_in_head_frame(sequence: EgoHandSequence, palm_axis_angle: np.ndarray) -> np.ndarray:
    axis_angle = np.asarray(palm_axis_angle, dtype=np.float64)
    if sequence.source_space != "camera":
        return axis_angle
    return opencv_camera_to_head_link_axis_angle(axis_angle)


def _extract_single_frame(
    wrist_position: np.ndarray,
    palm_axis_angle: np.ndarray,
    hand_pose_axis_angle: np.ndarray,
    keypoints: np.ndarray | None,
    keypoint_close_threshold: float,
) -> HandFeatures:
    wrist = np.asarray(wrist_position, dtype=np.float64)
    rotation = _rotvec_to_matrix(np.asarray(palm_axis_angle, dtype=np.float64))
    if keypoints is not None:
        curl, openness, pinch_score = _finger_state_from_keypoints(keypoints)
        palm = _palm_center_from_keypoints(keypoints)
    else:
        curl = _finger_curl_from_mano_pose(hand_pose_axis_angle)
        openness = 1.0 - float(np.mean([curl[name] for name in FINGER_NAMES]))
        pinch_score = _pinch_from_curl(curl)
        palm = wrist.copy()
    gripper_closed = openness <= keypoint_close_threshold
    return HandFeatures(
        wrist_position=wrist,
        palm_position=palm,
        palm_rotation=rotation,
        finger_curl=curl,
        pinch_score=float(np.clip(pinch_score, 0.0, 1.0)),
        hand_openness=float(np.clip(openness, 0.0, 1.0)),
        gripper_closed=bool(gripper_closed),
    )


def _palm_center_from_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Estimate palm center as mean of wrist + index/middle/ring/pinky MCPs."""
    kpts = np.asarray(keypoints, dtype=np.float64)
    if kpts.ndim != 2 or kpts.shape[1] != 3 or kpts.shape[0] <= max(PALM_CENTER_KEYPOINT_IDS):
        raise ValueError(
            f"Expected OpenPose keypoints (N>=21, 3), got {kpts.shape}"
        )
    return np.mean(kpts[list(PALM_CENTER_KEYPOINT_IDS)], axis=0)


def _keypoints_in_head_frame(
    sequence: EgoHandSequence,
    keypoints: np.ndarray,
) -> np.ndarray:
    kpts = np.asarray(keypoints, dtype=np.float64)
    if sequence.source_space != "camera":
        return kpts
    return opencv_camera_to_head_link_position(kpts.reshape(-1, 3)).reshape(kpts.shape)


def _finger_state_from_keypoints(
    keypoints: np.ndarray,
) -> tuple[dict[str, float], float, float]:
    wrist = keypoints[0]
    palm_refs = [keypoints[i] for i in (5, 9, 13, 17)]
    palm_scale = float(np.median([np.linalg.norm(p - wrist) for p in palm_refs]))
    if palm_scale < 1e-6:
        palm_scale = 1.0

    openness_by_finger = {}
    curl = {}
    for name in FINGER_NAMES:
        tip = keypoints[FINGER_TIP_IDS[name]]
        mcp = keypoints[FINGER_MCP_IDS[name]]
        extension = np.linalg.norm(tip - wrist) / palm_scale
        base = np.linalg.norm(mcp - wrist) / palm_scale
        open_score = (extension - 0.75 * base) / 1.1
        open_score = float(np.clip(open_score, 0.0, 1.0))
        openness_by_finger[name] = open_score
        curl[name] = float(1.0 - open_score)

    openness = float(np.mean([openness_by_finger[name] for name in FINGER_NAMES]))
    thumb_index = np.linalg.norm(
        keypoints[FINGER_TIP_IDS["thumb"]] - keypoints[FINGER_TIP_IDS["index"]]
    ) / palm_scale
    pinch_score = float(np.clip(1.0 - thumb_index / 1.2, 0.0, 1.0))
    return curl, openness, pinch_score


def _finger_curl_from_mano_pose(hand_pose_axis_angle: np.ndarray) -> dict[str, float]:
    pose = np.asarray(hand_pose_axis_angle, dtype=np.float64)
    if pose.shape != (45,):
        raise ValueError(f"Expected MANO hand pose with shape (45,), got {pose.shape}")

    curl: dict[str, float] = {}
    for name, pose_slice in FINGER_POSE_SLICES.items():
        joints = pose[pose_slice].reshape(3, 3)
        # Axis-angle magnitudes give a compact trajectory-level bend signal.
        # Values around 1.2 rad already correspond to a clearly curled finger.
        value = float(np.mean(np.linalg.norm(joints, axis=1)) / 1.2)
        curl[name] = float(np.clip(value, 0.0, 1.0))
    return curl


def _pinch_from_curl(curl: dict[str, float]) -> float:
    thumb = curl["thumb"]
    index = curl["index"]
    other_fingers = np.mean([curl["middle"], curl["ring"], curl["pinky"]])
    pinch_like = 0.65 * min(thumb, index) + 0.35 * max(0.0, index - 0.35 * other_fingers)
    return float(np.clip(pinch_like, 0.0, 1.0))


def _smooth_features(
    features: list[HandFeatures],
    smooth_alpha: float,
) -> list[HandFeatures]:
    alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
    if not features or alpha >= 1.0:
        return features

    smoothed: list[HandFeatures] = []
    previous_wrist = features[0].wrist_position
    previous_palm = features[0].palm_position
    previous_open = features[0].hand_openness
    previous_pinch = features[0].pinch_score
    previous_curl = features[0].finger_curl
    for item in features:
        wrist = alpha * item.wrist_position + (1.0 - alpha) * previous_wrist
        palm = alpha * item.palm_position + (1.0 - alpha) * previous_palm
        openness = alpha * item.hand_openness + (1.0 - alpha) * previous_open
        pinch = alpha * item.pinch_score + (1.0 - alpha) * previous_pinch
        curl = {
            name: alpha * item.finger_curl[name] + (1.0 - alpha) * previous_curl[name]
            for name in FINGER_NAMES
        }
        smoothed.append(
            HandFeatures(
                wrist_position=wrist,
                palm_position=palm,
                palm_rotation=item.palm_rotation,
                finger_curl=curl,
                pinch_score=float(np.clip(pinch, 0.0, 1.0)),
                hand_openness=float(np.clip(openness, 0.0, 1.0)),
                gripper_closed=bool(item.gripper_closed),
            )
        )
        previous_wrist = wrist
        previous_palm = palm
        previous_open = openness
        previous_pinch = pinch
        previous_curl = curl
    return smoothed


def _apply_discrete_gripper_threshold(
    features: list[HandFeatures],
    keypoint_close_threshold: float,
) -> list[HandFeatures]:
    return [
        HandFeatures(
            wrist_position=item.wrist_position,
            palm_position=item.palm_position,
            palm_rotation=item.palm_rotation,
            finger_curl=item.finger_curl,
            pinch_score=item.pinch_score,
            hand_openness=item.hand_openness,
            gripper_closed=bool(item.hand_openness <= keypoint_close_threshold),
        )
        for item in features
    ]


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
