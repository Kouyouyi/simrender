# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""RobotSpec: unified per-(robot + end-effector) configuration object."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

MimicRule = tuple[str, str, float, float]
HandBodyPredicate = Callable[[str], bool]
MjcfPatcher = Callable[[Path, float], Path]
QPosWriterFactory = Callable[[Any], Callable[[np.ndarray, np.ndarray], None]]


@dataclass(frozen=True)
class RobotSpec:
    """Immutable description of a robot + end-effector variant."""

    name: str
    display_name: str
    mjcf_path: Path
    action_dim: int
    action_joint_names: list[str]
    mimic_rules: list[MimicRule] = field(default_factory=list)
    hand_body_predicate: HandBodyPredicate = field(default=lambda _name: False)
    patch_mjcf: MjcfPatcher | None = None
    standing_height: float = 0.0
    head_mesh_names: tuple[str, ...] = ()
    left_shoulder_body: str = "left_arm_link1"
    right_shoulder_body: str = "right_arm_link1"
    left_wrist_body: str = "left_arm_link7"
    right_wrist_body: str = "right_arm_link7"
    has_floating_base: bool = False
    floating_base_joint: str = ""
    robot_type: str = ""
    modality: dict = field(default_factory=dict)
    qpos_writer_factory: QPosWriterFactory | None = None
    left_hand_clamp_joint_ids: np.ndarray | None = None
    right_hand_clamp_joint_ids: np.ndarray | None = None


_REGISTRY: dict[str, RobotSpec] = {}


def register(spec: RobotSpec) -> RobotSpec:
    """Register a spec by name."""
    _REGISTRY[spec.name] = spec
    return spec


def get_spec(name: str) -> RobotSpec:
    if name not in _REGISTRY:
        from retarget_galbot.robots import galbot  # noqa: F401
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown robot spec {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def all_specs() -> dict[str, RobotSpec]:
    """Return a copy of the registry."""
    from retarget_galbot.robots import galbot  # noqa: F401
    return dict(_REGISTRY)
