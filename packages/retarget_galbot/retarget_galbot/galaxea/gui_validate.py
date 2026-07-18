from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def _load_payload(path: str | Path) -> dict:
    with Path(path).expanduser().resolve().open("rb") as file:
        return pickle.load(file)


def _compute_tcp_positions(urdf_path: str, qpos: np.ndarray, frame_name: str) -> np.ndarray:
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    frame_id = model.getFrameId(frame_name)
    positions = []
    for frame_qpos in qpos:
        pin.forwardKinematics(model, data, frame_qpos)
        pin.updateFramePlacements(model, data)
        positions.append(data.oMf[frame_id].translation.copy())
    return np.asarray(positions)


class RetargetGUI:
    def __init__(self, payload: dict):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        self.plt = plt
        self.Button = Button
        self.Slider = Slider
        self.payload = payload
        self.meta = payload["meta_data"]
        self.frames = payload["data"]
        self.qpos = np.asarray([frame["qpos"] for frame in self.frames], dtype=np.float64)
        self.timestamps = np.asarray([frame["timestamp"] for frame in self.frames])
        self.left_target = np.asarray(
            [frame["left"]["target_tcp_position"] for frame in self.frames],
            dtype=np.float64,
        )
        self.right_target = np.asarray(
            [frame["right"]["target_tcp_position"] for frame in self.frames],
            dtype=np.float64,
        )
        self.left_actual = _compute_tcp_positions(
            self.meta["robot_urdf"], self.qpos, "left_gripper_tcp_link"
        )
        self.right_actual = _compute_tcp_positions(
            self.meta["robot_urdf"], self.qpos, "right_gripper_tcp_link"
        )
        self.left_error = np.linalg.norm(self.left_target - self.left_actual, axis=1)
        self.right_error = np.linalg.norm(self.right_target - self.right_actual, axis=1)
        controlled = set(self.meta.get("target_joint_names", []))
        self.joint_names = self.meta["joint_names"]
        self.controlled_indices = [
            index for index, name in enumerate(self.joint_names) if name in controlled
        ]
        if not self.controlled_indices:
            self.controlled_indices = list(range(len(self.joint_names)))
        self.is_playing = False
        self.current_frame = 0
        self._build()

    def _build(self) -> None:
        self.fig = self.plt.figure(figsize=(14, 9))
        grid = self.fig.add_gridspec(
            3, 3, height_ratios=[1.5, 1.0, 0.16], width_ratios=[1.25, 1.0, 1.0]
        )
        self.ax3d = self.fig.add_subplot(grid[:2, 0], projection="3d")
        self.ax_err = self.fig.add_subplot(grid[0, 1:])
        self.ax_bar = self.fig.add_subplot(grid[1, 1:])
        self.ax_slider = self.fig.add_subplot(grid[2, :2])
        self.ax_button = self.fig.add_subplot(grid[2, 2])

        self.ax3d.plot(
            self.left_target[:, 0],
            self.left_target[:, 1],
            self.left_target[:, 2],
            "m--",
            linewidth=1,
            label="left target",
        )
        self.ax3d.plot(
            self.left_actual[:, 0],
            self.left_actual[:, 1],
            self.left_actual[:, 2],
            "m-",
            linewidth=1,
            label="left actual",
        )
        self.ax3d.plot(
            self.right_target[:, 0],
            self.right_target[:, 1],
            self.right_target[:, 2],
            "c--",
            linewidth=1,
            label="right target",
        )
        self.ax3d.plot(
            self.right_actual[:, 0],
            self.right_actual[:, 1],
            self.right_actual[:, 2],
            "c-",
            linewidth=1,
            label="right actual",
        )
        self.left_target_dot = self.ax3d.scatter([], [], [], c="magenta", s=60)
        self.left_actual_dot = self.ax3d.scatter([], [], [], c="purple", s=60)
        self.right_target_dot = self.ax3d.scatter([], [], [], c="cyan", s=60)
        self.right_actual_dot = self.ax3d.scatter([], [], [], c="blue", s=60)
        self.ax3d.set_title("TCP Target vs FK Actual")
        self.ax3d.set_xlabel("x")
        self.ax3d.set_ylabel("y")
        self.ax3d.set_zlabel("z")
        self.ax3d.legend(loc="upper left")
        self._equalize_3d_axes()

        self.ax_err.plot(self.timestamps, self.left_error * 1000, "m", label="left")
        self.ax_err.plot(self.timestamps, self.right_error * 1000, "c", label="right")
        self.err_cursor = self.ax_err.axvline(self.timestamps[0], color="k", alpha=0.4)
        self.ax_err.set_title("TCP Position Error")
        self.ax_err.set_ylabel("mm")
        self.ax_err.grid(True, alpha=0.3)
        self.ax_err.legend()

        labels = [self.joint_names[index] for index in self.controlled_indices]
        self.bar_container = self.ax_bar.barh(
            np.arange(len(self.controlled_indices)),
            self.qpos[0, self.controlled_indices],
            color="#5b8fd9",
        )
        self.ax_bar.set_yticks(np.arange(len(labels)))
        self.ax_bar.set_yticklabels(labels, fontsize=7)
        self.ax_bar.set_title("Controlled Joint Qpos")
        self.ax_bar.grid(True, axis="x", alpha=0.25)

        self.slider = self.Slider(
            self.ax_slider,
            "frame",
            0,
            len(self.frames) - 1,
            valinit=0,
            valstep=1,
        )
        self.slider.on_changed(lambda value: self.update(int(value)))
        self.button = self.Button(self.ax_button, "Play / Pause")
        self.button.on_clicked(self._toggle_play)
        self.status_text = self.fig.text(0.02, 0.02, "", fontsize=10)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.update(0)

    def _equalize_3d_axes(self) -> None:
        points = np.concatenate(
            [self.left_target, self.left_actual, self.right_target, self.right_actual],
            axis=0,
        )
        center = points.mean(axis=0)
        radius = np.max(np.ptp(points, axis=0)) * 0.55
        radius = max(radius, 0.1)
        self.ax3d.set_xlim(center[0] - radius, center[0] + radius)
        self.ax3d.set_ylim(center[1] - radius, center[1] + radius)
        self.ax3d.set_zlim(center[2] - radius, center[2] + radius)

    def _set_scatter(self, artist, point: np.ndarray) -> None:
        artist._offsets3d = ([point[0]], [point[1]], [point[2]])

    def update(self, frame: int) -> None:
        self.current_frame = int(np.clip(frame, 0, len(self.frames) - 1))
        i = self.current_frame
        self._set_scatter(self.left_target_dot, self.left_target[i])
        self._set_scatter(self.left_actual_dot, self.left_actual[i])
        self._set_scatter(self.right_target_dot, self.right_target[i])
        self._set_scatter(self.right_actual_dot, self.right_actual[i])
        self.err_cursor.set_xdata([self.timestamps[i], self.timestamps[i]])
        values = self.qpos[i, self.controlled_indices]
        for bar, value in zip(self.bar_container, values):
            bar.set_width(value)
        self.ax_bar.relim()
        self.ax_bar.autoscale_view(scalex=True, scaley=False)
        self.status_text.set_text(
            f"frame={i}  t={self.timestamps[i]:.3f}s  "
            f"left_err={self.left_error[i] * 1000:.2f}mm  "
            f"right_err={self.right_error[i] * 1000:.2f}mm"
        )
        self.fig.canvas.draw_idle()

    def _toggle_play(self, _event=None) -> None:
        self.is_playing = not self.is_playing
        while self.is_playing and self.plt.fignum_exists(self.fig.number):
            next_frame = (self.current_frame + 1) % len(self.frames)
            self.slider.set_val(next_frame)
            self.plt.pause(0.03)

    def _on_key(self, event) -> None:
        if event.key == "right":
            self.slider.set_val(min(self.current_frame + 1, len(self.frames) - 1))
        elif event.key == "left":
            self.slider.set_val(max(self.current_frame - 1, 0))
        elif event.key == " ":
            self._toggle_play()


def main() -> None:
    parser = argparse.ArgumentParser(description="Matplotlib GUI validation for retarget output.")
    parser.add_argument("--pickle", required=True, help="Retarget output pickle.")
    parser.add_argument("--snapshot", default=None, help="Optional PNG snapshot path instead of opening GUI.")
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    payload = _load_payload(args.pickle)
    if payload.get("meta_data", {}).get("backend") != "dex_bimanual":
        raise ValueError("gui_validate currently expects backend=dex_bimanual")
    gui = RetargetGUI(payload)
    gui.fig.tight_layout(rect=[0, 0.04, 1, 1])
    if args.snapshot:
        output = Path(args.snapshot).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        gui.fig.savefig(output, dpi=160)
        print(f"saved GUI snapshot to {output}")
    else:
        gui.plt.show()


if __name__ == "__main__":
    main()
