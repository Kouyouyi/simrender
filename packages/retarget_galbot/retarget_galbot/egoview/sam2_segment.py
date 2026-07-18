# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM2 hand/arm segmentation for Galbot egoview synthesis.

Adapted from Open-AoE Phantom Stage-3 hand removal (SAM2 video seeding).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


class HandArmSegmenter:
    """Small adapter around SAM2's video predictor.

    ``egoview.render`` owns seed selection and propagation. This class only builds
    the predictor and converts in-memory RGB frames to the JPEG-folder input
    layout expected by SAM2.
    """

    def __init__(
        self,
        model_cfg: str | None = None,
        checkpoint: str | Path | None = None,
        device: str | None = None,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = False,
    ) -> None:
        from sam2.build_sam import build_sam2_video_predictor

        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu

        cfg = model_cfg or os.environ.get("SAM2_MODEL_CFG", DEFAULT_SAM2_CFG)
        ckpt_raw = checkpoint or os.environ.get("SAM2_CHECKPOINT")
        if not ckpt_raw:
            raise FileNotFoundError(
                "SAM2 checkpoint not set. Pass checkpoint=... or export "
                "SAM2_CHECKPOINT=/path/to/sam2.1_hiera_base_plus.pt"
            )
        ckpt = Path(ckpt_raw)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {ckpt}. Set SAM2_CHECKPOINT to a "
                "valid sam2.1_hiera_base_plus.pt path."
            )

        self.predictor = build_sam2_video_predictor(
            cfg,
            str(ckpt),
            device=self.device,
            vos_optimized=False,
        )

    def _build_inference_state(self, frames_rgb: np.ndarray):
        """Create a SAM2 inference state from ``(T,H,W,3)`` RGB uint8 frames."""
        frames = np.asarray(frames_rgb)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected RGB video array (T,H,W,3), got {frames.shape}")

        if self._tmpdir is not None:
            self._tmpdir.cleanup()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="sam2_frames_")
        frame_dir = Path(self._tmpdir.name)

        for idx, frame in enumerate(frames):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_dir / f"{idx:05d}.jpg"), bgr)

        return self.predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
            async_loading_frames=False,
        )

    def __del__(self) -> None:
        if getattr(self, "_tmpdir", None) is not None:
            self._tmpdir.cleanup()
