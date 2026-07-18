"""Galbot retargeting pipeline (Pinocchio IK → MuJoCo action qpos)."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np

from retarget_galbot.galaxea.annotations import load_action_segments
from retarget_galbot.galaxea.bimanual import BimanualDexRetargeter
from retarget_galbot.galaxea.config import load_config
from retarget_galbot.galaxea.ego_data import load_ego_hand_sequence
from retarget_galbot.galaxea.features import extract_hand_features
from retarget_galbot.galaxea.io import save_jsonl_output, save_pickle_output
from retarget_galbot.galaxea.urdf_utils import parse_movable_joints
from retarget_galbot.robots import RobotSpec

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "galbot_dex_bimanual.yml"


@dataclasses.dataclass
class GalbotEpisodeMeta:
    episode_dir: Path
    reconstruction_dir: Path
    fps: float
    joint_names: list[str]


@dataclasses.dataclass
class RetargetResult:
    actions: np.ndarray
    frames: list
    spec_name: str
    meta: GalbotEpisodeMeta
    task_description: str = ""
    scale: float = 1.0


class RetargetSession:
    """Reusable Galbot retargeting session."""

    def __init__(
        self,
        spec: RobotSpec,
        config_path: str | Path | None = None,
        source_scale: float | None = None,
    ):
        self.spec = spec
        self.config = load_config(config_path or DEFAULT_CONFIG)
        if source_scale is not None:
            raw = dict(self.config.raw)
            raw["source_scale"] = float(source_scale)
            self.config = dataclasses.replace(self.config, raw=raw)
        self.joints = parse_movable_joints(self.config.urdf_path)
        self.retargeter = BimanualDexRetargeter(self.config, self.joints)

        if list(self.retargeter.joint_names) != list(spec.action_joint_names):
            logger.warning(
                "Pinocchio joint order differs from MuJoCo action order; "
                "actions will be reordered for visualization."
            )

    def retarget(
        self,
        episode_dir: Path | str,
        refine_shoulder: bool = True,
        frame: str = "cam",
        max_frames: int | None = None,
        stride: int = 1,
        pickle_output: Path | None = None,
        jsonl_output: Path | None = None,
    ) -> RetargetResult:
        del refine_shoulder, frame

        left_hand = load_ego_hand_sequence(
            episode_dir,
            hand="left",
            source_space=self.config.source_space,
            fps=self.config.fps,
        )
        right_hand = load_ego_hand_sequence(
            episode_dir,
            hand="right",
            source_space=self.config.source_space,
            fps=self.config.fps,
        )
        left_sequence = extract_hand_features(
            left_hand,
            smooth_alpha=self.config.feature_smooth_alpha,
            keypoint_close_threshold=float(
                self.config.raw.get("gripper", {}).get("keypoint_close_threshold", 0.68)
            ),
        )
        right_sequence = extract_hand_features(
            right_hand,
            smooth_alpha=self.config.feature_smooth_alpha,
            keypoint_close_threshold=float(
                self.config.raw.get("gripper", {}).get("keypoint_close_threshold", 0.68)
            ),
        )
        segments = load_action_segments(left_hand.segment_dir)

        frames = self.retargeter.retarget(
            left_sequence,
            right_sequence,
            segments,
            max_frames=max_frames,
            stride=stride,
        )
        if not frames:
            raise ValueError(f"No Galbot frames generated for {episode_dir}")

        actions = np.asarray([frame.qpos for frame in frames], dtype=np.float32)
        actions = self._reorder_actions_if_needed(actions)
        if actions.shape[1] != self.spec.action_dim:
            raise ValueError(
                f"Expected action_dim={self.spec.action_dim}, got {actions.shape}"
            )

        meta_data = self._meta_data(left_hand, len(frames))
        if pickle_output:
            save_pickle_output(pickle_output, meta_data, frames)
        if jsonl_output:
            save_jsonl_output(jsonl_output, frames)

        meta = GalbotEpisodeMeta(
            episode_dir=left_hand.segment_dir,
            reconstruction_dir=left_hand.reconstruction_dir,
            fps=self.config.fps,
            joint_names=list(self.spec.action_joint_names),
        )
        logger.info("Galbot episode T=%d, action_dim=%d", actions.shape[0], actions.shape[1])
        return RetargetResult(
            actions=actions,
            frames=frames,
            spec_name=self.spec.name,
            meta=meta,
            scale=float(self.retargeter.last_source_scale),
        )

    def _reorder_actions_if_needed(self, actions: np.ndarray) -> np.ndarray:
        source_names = list(self.retargeter.joint_names)
        target_names = list(self.spec.action_joint_names)
        if source_names == target_names:
            return actions
        source = {name: i for i, name in enumerate(source_names)}
        reordered = np.zeros((actions.shape[0], len(target_names)), dtype=actions.dtype)
        for out_i, name in enumerate(target_names):
            if name in source:
                reordered[:, out_i] = actions[:, source[name]]
        return reordered

    def _meta_data(self, left_hand, frame_count: int) -> dict:
        return {
            "backend": self.config.backend,
            "robot_name": self.config.robot_name,
            "robot_urdf": str(self.config.urdf_path),
            "robot_mjcf": str(self.spec.mjcf_path),
            "config_path": str(self.config.path),
            "source_segment": str(left_hand.segment_dir),
            "source_reconstruction_dir": str(left_hand.reconstruction_dir),
            "source_space": self.config.source_space,
            "reference_frame": str(self.config.raw.get("reference_frame", "head")),
            "head_link_name": str(self.config.raw.get("head_link_name", "head_end_effector_mount_link")),
            "position_mapping": str(self.config.raw.get("position_mapping", "absolute")).lower(),
            "source_scale": float(self.retargeter.last_source_scale),
            "fps": self.config.fps,
            "joint_names": list(self.spec.action_joint_names),
            "pinocchio_joint_names": list(self.retargeter.joint_names),
            "frame_count": frame_count,
        }


def discover_episodes(data_root: Path) -> list[Path]:
    episodes = []
    data_root = Path(data_root)
    for entry in sorted(data_root.iterdir()):
        if not entry.is_dir():
            continue
        for sub in ("ego_process/ego_hands_reconstruction", "ego_hands_reconstruction"):
            if (entry / sub / "hands.npz").exists():
                episodes.append(entry)
                break
    logger.info("Discovered %d episodes under %s", len(episodes), data_root)
    return episodes
