# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Galbot egoview synthesis: SAM2 → ProPainter → MuJoCo robot overlay."""

from retarget_galbot.egoview.overlay import OverlayStatsEMA, overlay_robot
from retarget_galbot.egoview.render import render_egoview_frames, render_mujoco_views

__all__ = [
    "OverlayStatsEMA",
    "overlay_robot",
    "render_egoview_frames",
    "render_mujoco_views",
]
