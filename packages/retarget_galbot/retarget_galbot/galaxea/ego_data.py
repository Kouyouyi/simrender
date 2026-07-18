from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


HAND_TO_INDEX = {"left": 0, "right": 1}


@dataclass(frozen=True)
class EgoHandSequence:
    segment_dir: Path
    reconstruction_dir: Path
    source_path: Path
    hand: str
    valid: np.ndarray
    trans: np.ndarray
    rot_axis_angle: np.ndarray
    hand_pose_axis_angle: np.ndarray
    keypoints: np.ndarray | None
    betas: np.ndarray
    fps: float
    source_space: str

    @property
    def frame_count(self) -> int:
        return int(self.trans.shape[0])


def load_ego_hand_sequence(
    reconstruction: str | Path, hand: str, source_space: str = "camera", fps: float = 30
) -> EgoHandSequence:
    input_path = Path(reconstruction).expanduser().resolve()
    hands_path = _resolve_hands_npz(input_path)
    reconstruction_dir = hands_path.parent
    segment_path = _infer_segment_dir(input_path, reconstruction_dir)

    hand_key = hand.lower()
    if hand_key not in HAND_TO_INDEX:
        raise ValueError("hand must be either left or right")
    hand_index = HAND_TO_INDEX[hand_key]

    data = np.load(hands_path, mmap_mode="r")
    source_space_key = source_space.lower()
    if source_space_key not in {"camera", "world"}:
        raise ValueError("source_space must be either camera or world")
    suffix = "_cam" if source_space_key == "camera" else ""
    trans_name = f"pred_trans{suffix}"
    rot_name = f"pred_rot{suffix}"
    if trans_name not in data or rot_name not in data:
        raise KeyError(f"{hands_path} does not contain {trans_name}/{rot_name}")
    for required in ("pred_valid", "pred_hand_pose", "pred_betas"):
        if required not in data:
            raise KeyError(f"{hands_path} does not contain {required}")

    valid = data["pred_valid"][hand_index].astype(bool)
    trans = np.asarray(data[trans_name][hand_index], dtype=np.float64)
    rot = np.asarray(data[rot_name][hand_index], dtype=np.float64)
    hand_pose = np.asarray(data["pred_hand_pose"][hand_index], dtype=np.float64)
    betas = np.asarray(data["pred_betas"][hand_index], dtype=np.float64)
    _ensure_keypoints_sidecar(reconstruction_dir)
    keypoints = _load_keypoints(reconstruction_dir, hand_key, source_space_key)
    return EgoHandSequence(
        segment_dir=segment_path,
        reconstruction_dir=reconstruction_dir,
        source_path=hands_path,
        hand=hand_key,
        valid=valid,
        trans=trans,
        rot_axis_angle=rot,
        hand_pose_axis_angle=hand_pose,
        keypoints=keypoints,
        betas=betas,
        fps=fps,
        source_space=source_space_key,
    )


def _ensure_keypoints_sidecar(reconstruction_dir: Path) -> None:
    """Generate hands_keypoints.npz via MANO FK when missing/outdated."""
    from retarget_galbot.aoe.mano_sidecar import _ensure_sidecar

    try:
        _ensure_sidecar(reconstruction_dir)
    except (FileNotFoundError, OSError):
        # Palm center falls back to wrist when keypoints are unavailable.
        return


def _load_keypoints(
    reconstruction_dir: Path,
    hand: str,
    source_space: str,
) -> np.ndarray | None:
    sidecar_path = reconstruction_dir / "hands_keypoints.npz"
    if not sidecar_path.exists():
        return None

    key = f"{hand}_keypoints_cam" if source_space == "camera" else f"{hand}_keypoints_world"
    sidecar = np.load(sidecar_path, mmap_mode="r")
    if key not in sidecar:
        return None
    return np.asarray(sidecar[key], dtype=np.float64)


def first_valid_reference(sequence: EgoHandSequence) -> np.ndarray:
    valid_indices = np.flatnonzero(sequence.valid)
    if len(valid_indices) == 0:
        return np.zeros(3, dtype=np.float64)
    return sequence.trans[int(valid_indices[0])].copy()


def _resolve_hands_npz(path: Path) -> Path:
    if path.is_file():
        if path.name != "hands.npz":
            raise FileNotFoundError(f"Expected hands.npz, got: {path}")
        return path

    candidates = [
        path / "hands.npz",
        path / "ego_process" / "ego_hands_reconstruction" / "hands.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing ego hand reconstruction file. Expected one of: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _infer_segment_dir(input_path: Path, reconstruction_dir: Path) -> Path:
    if input_path.is_file():
        input_path = input_path.parent
    if reconstruction_dir.name == "ego_hands_reconstruction":
        ego_process = reconstruction_dir.parent
        if ego_process.name == "ego_process":
            return ego_process.parent
    return input_path
