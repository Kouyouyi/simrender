"""Rendering and standards-compliant H.264 output helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np


class H264Writer:
    def __init__(self, path: str | Path, fps: float) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(
            str(self.path),
            format="FFMPEG",
            mode="I",
            fps=float(fps),
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
            output_params=[
                "-preset",
                "medium",
                "-crf",
                "18",
                "-profile:v",
                "high",
                "-level:v",
                "4.0",
                "-tag:v",
                "avc1",
                "-movflags",
                "+faststart",
            ],
        )

    def append(self, rgb_frame: np.ndarray) -> None:
        self._writer.append_data(np.asarray(rgb_frame, dtype=np.uint8))

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "H264Writer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def make_camera(*, lookat: np.ndarray, distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def draw_label(frame: np.ndarray, title: str, lines: tuple[str, ...] = ()) -> np.ndarray:
    output = np.ascontiguousarray(frame.copy())
    height = 38 + 23 * len(lines)
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], height), (12, 16, 18), thickness=-1)
    cv2.addWeighted(overlay, 0.78, output, 0.22, 0, output)
    cv2.putText(output, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (14, 51 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (215, 225, 230),
            1,
            cv2.LINE_AA,
        )
    return output


class IndexedVideoReader:
    """Decode source frames through ffmpeg, avoiding OpenCV's fragile AV1 path."""

    def __init__(
        self,
        path: str | Path,
        start_frame: int = 0,
        *,
        width: int,
        height: int,
        source_fps: float = 30.0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.frame_bytes = self.width * self.height * 3
        start_seconds = float(start_frame) / float(source_fps)
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-ss",
            f"{start_seconds:.9f}",
            "-vf",
            f"scale={self.width}:{self.height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.frame_bytes * 2,
        )
        if self.process.stdout is None:
            raise RuntimeError(f"Failed to create ffmpeg pipe for {path}")
        self.next_index = int(start_frame)

    def _read_one(self, frame_index: int) -> np.ndarray:
        assert self.process.stdout is not None
        payload = self.process.stdout.read(self.frame_bytes)
        if len(payload) != self.frame_bytes:
            error = b""
            if self.process.stderr is not None:
                error = self.process.stderr.read()
            message = error.decode("utf-8", errors="replace").strip()
            raise EOFError(
                f"ffmpeg ended before source frame {frame_index}; "
                f"received {len(payload)}/{self.frame_bytes} bytes. {message}"
            )
        return np.frombuffer(payload, dtype=np.uint8).reshape(self.height, self.width, 3)

    def read_rgb(self, frame_index: int) -> np.ndarray:
        if frame_index < self.next_index:
            raise ValueError("IndexedVideoReader only supports increasing frame indices")
        while self.next_index < frame_index:
            self._read_one(self.next_index)
            self.next_index += 1
        rgb = self._read_one(frame_index)
        self.next_index += 1
        return rgb

    def close(self) -> None:
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait()
