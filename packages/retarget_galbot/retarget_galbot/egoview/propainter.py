# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ProPainter subprocess wrapper for Galbot egoview hand removal.

Adapted from Open-AoE Phantom Stage-3 inpainting (Phantom used E2FGVI; this
repo uses ProPainter).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


DEFAULT_PROPAINTER_ENV = "propainter"
DEFAULT_CHUNK_SIZE = 50
DEFAULT_SUBVIDEO_LENGTH = 100


def repository_root() -> Path:
    """Return the simrender checkout containing the Galbot package."""
    return Path(__file__).resolve().parents[4]


def bundled_propainter_root() -> Path:
    """Return the bundled, weight-free ProPainter runtime source directory."""
    return repository_root() / "third_party" / "ProPainter"


class ProPainterRunner:
    """Run ProPainter in its own Python environment.

    The Galbot retarget/SAM2/MuJoCo path runs in the main process. ProPainter is
    launched as a subprocess over temporary frame/mask folders; its
    ``inpaint_out.mp4`` is read back as RGB frames.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        conda_env: str | None = None,
        python_executable: str | Path | None = None,
    ) -> None:
        root_raw = root or os.environ.get("PROPAINTER_ROOT") or bundled_propainter_root()
        self.root = Path(root_raw).expanduser().resolve()
        self.conda_env = conda_env or os.environ.get(
            "PROPAINTER_CONDA_ENV", DEFAULT_PROPAINTER_ENV
        )
        if python_executable is not None:
            self.python_executable = Path(python_executable)
        elif "PROPAINTER_PYTHON" in os.environ:
            self.python_executable = Path(os.environ["PROPAINTER_PYTHON"])
        else:
            self.python_executable = None
        self.chunk_size = int(
            os.environ.get("PROPAINTER_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
        )
        self.subvideo_length = int(
            os.environ.get("PROPAINTER_SUBVIDEO_LENGTH", DEFAULT_SUBVIDEO_LENGTH)
        )
        self.use_fp16 = os.environ.get("PROPAINTER_FP16", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        self.mask_dilation = int(os.environ.get("PROPAINTER_MASK_DILATION", "4"))
        self.neighbor_length = int(os.environ.get("PROPAINTER_NEIGHBOR_LENGTH", "10"))
        self.ref_stride = int(os.environ.get("PROPAINTER_REF_STRIDE", "10"))
        self.raft_iter = int(os.environ.get("PROPAINTER_RAFT_ITER", "20"))

        if not self.root.exists():
            raise FileNotFoundError(
                f"ProPainter repo not found: {self.root}. Clone it or set "
                "PROPAINTER_ROOT."
            )
        script = self.root / "inference_propainter.py"
        if not script.exists():
            raise FileNotFoundError(f"ProPainter inference script not found: {script}")

    def run(
        self,
        ego_video: np.ndarray,
        arm_mask: np.ndarray,
        neighbor_stride: int = 5,
        ref_length: int = 10,
        num_ref: int = -1,
    ) -> np.ndarray:
        del neighbor_stride, ref_length, num_ref

        frames = np.asarray(ego_video, dtype=np.uint8)
        masks = np.asarray(arm_mask).astype(bool)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected RGB video array (T,H,W,3), got {frames.shape}")
        if masks.shape != frames.shape[:3]:
            raise ValueError(f"Mask shape {masks.shape} does not match {frames.shape[:3]}")

        if not masks.any():
            return frames.copy()

        if frames.shape[0] > self.chunk_size:
            outputs = []
            total = frames.shape[0]
            for start in range(0, total, self.chunk_size):
                end = min(start + self.chunk_size, total)
                print(f"ProPainter chunk {start}:{end} / {total}", flush=True)
                outputs.append(self._run_worker(frames[start:end], masks[start:end]))
            return np.concatenate(outputs, axis=0)

        return self._run_worker(frames, masks)

    def _run_worker(self, frames: np.ndarray, masks: np.ndarray) -> np.ndarray:
        tmp_root = Path(
            os.environ.get("RETARGET_TMPDIR", tempfile.gettempdir())
        )
        tmp_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="propainter_egoview_", dir=tmp_root) as tmp:
            tmpdir = Path(tmp)
            frame_dir = tmpdir / "frames"
            mask_dir = tmpdir / "masks"
            output_dir = tmpdir / "results"
            frame_dir.mkdir()
            mask_dir.mkdir()

            for idx, frame in enumerate(frames):
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(frame_dir / f"{idx:05d}.png"), bgr)
                mask = (masks[idx].astype(np.uint8) * 255)
                cv2.imwrite(str(mask_dir / f"{idx:05d}.png"), mask)

            cmd = self._command(frame_dir, mask_dir, output_dir, frames.shape)
            env = os.environ.copy()
            env.setdefault("TMPDIR", str(tmp_root))
            proc = subprocess.run(
                cmd,
                cwd=str(self.root),
                env=env,
                text=True,
                stdout=None,
                stderr=None,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "ProPainter subprocess failed with exit code "
                    f"{proc.returncode}. See ProPainter logs above."
                )

            out_video = output_dir / frame_dir.name / "inpaint_out.mp4"
            if not out_video.exists():
                raise RuntimeError(
                    "ProPainter finished without writing expected output: "
                    f"{out_video}"
                )
            return self._read_video(out_video, len(frames), (frames.shape[2], frames.shape[1]))

    def _command(
        self,
        frame_dir: Path,
        mask_dir: Path,
        output_dir: Path,
        shape: tuple[int, int, int, int],
    ) -> list[str]:
        _, height, width, _ = shape
        args = [
            str(self.root / "inference_propainter.py"),
            "--video",
            str(frame_dir),
            "--mask",
            str(mask_dir),
            "--output",
            str(output_dir),
            "--height",
            str(height),
            "--width",
            str(width),
            "--save_fps",
            "30",
            "--subvideo_length",
            str(self.subvideo_length),
            "--mask_dilation",
            str(self.mask_dilation),
            "--neighbor_length",
            str(self.neighbor_length),
            "--ref_stride",
            str(self.ref_stride),
            "--raft_iter",
            str(self.raft_iter),
        ]
        if self.use_fp16:
            args.append("--fp16")

        if self.python_executable is not None:
            return [str(self.python_executable), *args]

        conda = shutil.which("conda")
        if conda is None:
            raise RuntimeError(
                "Neither PROPAINTER_PYTHON nor conda is available. Export "
                "PROPAINTER_PYTHON to the ProPainter environment interpreter."
            )
        return [conda, "run", "-n", self.conda_env, "python", *args]

    @staticmethod
    def _read_video(
        video_path: Path,
        expected_frames: int,
        out_size: tuple[int, int],
    ) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open ProPainter output: {video_path}")

        frames = []
        while len(frames) < expected_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != out_size[0] or frame.shape[0] != out_size[1]:
                frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if not frames:
            raise RuntimeError(f"No frames read from ProPainter output: {video_path}")
        while len(frames) < expected_frames:
            frames.append(frames[-1].copy())
        return np.stack(frames).astype(np.uint8)
