from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GripperConfig:
    open_value: float
    close_value: float
    pinch_close_threshold: float
    openness_open_threshold: float
    close_verbs: tuple[str, ...]
    open_verbs: tuple[str, ...]


@dataclass(frozen=True)
class RetargetConfig:
    path: Path
    backend: str
    robot_name: str
    urdf_path: Path
    hand: str
    fps: float
    arm_joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...]
    tcp_link_name: str | None
    source_space: str
    workspace_scale: tuple[float, float, float]
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    low_pass_alpha: float
    feature_smooth_alpha: float
    gripper: GripperConfig
    raw: dict[str, Any]


def _tuple_str(values: Any) -> tuple[str, ...]:
    return tuple(str(value) for value in (values or []))


def _tuple_float3(values: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if values is None:
        return default
    parsed = tuple(float(value) for value in values)
    if len(parsed) != 3:
        raise ValueError("workspace_scale must have exactly three values")
    return parsed


def load_config(path: str | Path) -> RetargetConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict) or "retargeting" not in raw:
        raise ValueError(f"{config_path} must contain a top-level retargeting block")

    cfg = raw["retargeting"]
    base_dir = config_path.parent
    urdf_path = Path(str(cfg["urdf_path"])).expanduser()
    if not urdf_path.is_absolute():
        urdf_path = (base_dir / urdf_path).resolve()

    gripper_cfg = cfg.get("gripper", {}) or {}
    gripper = GripperConfig(
        open_value=float(gripper_cfg.get("open_value", 0.0)),
        close_value=float(gripper_cfg.get("close_value", 1.65)),
        pinch_close_threshold=float(gripper_cfg.get("pinch_close_threshold", 0.55)),
        openness_open_threshold=float(gripper_cfg.get("openness_open_threshold", 0.62)),
        close_verbs=tuple(v.lower() for v in _tuple_str(gripper_cfg.get("close_verbs"))),
        open_verbs=tuple(v.lower() for v in _tuple_str(gripper_cfg.get("open_verbs"))),
    )

    return RetargetConfig(
        path=config_path,
        backend=str(cfg.get("backend", "heuristic")).lower(),
        robot_name=str(cfg.get("robot_name", "galbot")),
        urdf_path=urdf_path,
        hand=str(cfg.get("hand", "right")).lower(),
        fps=float(cfg.get("fps", 30)),
        arm_joint_names=_tuple_str(cfg.get("arm_joint_names")),
        gripper_joint_names=_tuple_str(cfg.get("gripper_joint_names")),
        tcp_link_name=cfg.get("tcp_link_name"),
        source_space=str(cfg.get("source_space", "camera")).lower(),
        workspace_scale=_tuple_float3(cfg.get("workspace_scale"), (1.0, 1.0, 1.0)),
        workspace_min=_tuple_float3(cfg.get("workspace_min"), (-0.8, -0.8, -0.8)),
        workspace_max=_tuple_float3(cfg.get("workspace_max"), (0.8, 0.8, 0.8)),
        low_pass_alpha=float(cfg.get("low_pass_alpha", 1.0)),
        feature_smooth_alpha=float(cfg.get("feature_smooth_alpha", cfg.get("low_pass_alpha", 1.0))),
        gripper=gripper,
        raw=cfg,
    )
