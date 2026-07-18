# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""AOE dataset path and file-name constants."""

from __future__ import annotations

from pathlib import Path

# ── AOE dataset file names ───────────────────────────────────────────────────

HANDS_NPZ_NAME = "hands.npz"
SIDECAR_NPZ_NAME = "hands_keypoints.npz"

HANDS_RECON_SUBDIRS = (
    "ego_hands_reconstruction",
    Path("ego_process") / "ego_hands_reconstruction",
)

UNDISTORTED_VIDEO_SUBDIRS = (
    "ego_undistorted_video",
    Path("ego_process") / "ego_undistorted_video",
)
