# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Shared constants for the AoE → Galbot retargeting pipeline."""

from pathlib import Path

# ── Paths (relative to project root) ────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
MANO_MODELS_DIR = ASSETS_DIR / "mano_models"
# Fallback to the shared assets/mano/ location for MANO models.
# Users should run assets/mano/download_mano.sh before using this package.
SHARED_MANO_DIR = PROJECT_ROOT.parent.parent / "assets" / "mano"

# ── LeRobot format constants ────────────────────────────────────────────────

FPS = 30
VIDEO_KEY = "observation.images.egoview"
VIDEO_HEIGHT = 480
VIDEO_WIDTH = 640
