from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


def _load_payload(path: str | Path):
    with Path(path).expanduser().resolve().open("rb") as file:
        return pickle.load(file)


def _tcp_positions(urdf_path: str, qpos_array: np.ndarray, frame_name: str) -> np.ndarray:
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    frame_id = model.getFrameId(frame_name)
    positions = []
    for qpos in qpos_array:
        pin.forwardKinematics(model, data, qpos)
        pin.updateFramePlacements(model, data)
        positions.append(data.oMf[frame_id].translation.copy())
    return np.asarray(positions)


def _plot_bimanual(payload: dict, output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    meta = payload["meta_data"]
    frames = payload["data"]
    qpos = np.asarray([frame["qpos"] for frame in frames], dtype=np.float64)
    t = np.asarray([frame["timestamp"] for frame in frames], dtype=np.float64)

    left_target = np.asarray(
        [frame["left"]["target_tcp_position"] for frame in frames], dtype=np.float64
    )
    right_target = np.asarray(
        [frame["right"]["target_tcp_position"] for frame in frames], dtype=np.float64
    )
    left_actual = _tcp_positions(meta["robot_urdf"], qpos, "left_gripper_tcp_link")
    right_actual = _tcp_positions(meta["robot_urdf"], qpos, "right_gripper_tcp_link")
    left_error = np.linalg.norm(left_target - left_actual, axis=1)
    right_error = np.linalg.norm(right_target - right_actual, axis=1)

    fig = plt.figure(figsize=(14, 10))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(left_target[:, 0], left_target[:, 1], left_target[:, 2], "m--", label="left target")
    ax3d.plot(left_actual[:, 0], left_actual[:, 1], left_actual[:, 2], "m-", label="left actual")
    ax3d.plot(right_target[:, 0], right_target[:, 1], right_target[:, 2], "c--", label="right target")
    ax3d.plot(right_actual[:, 0], right_actual[:, 1], right_actual[:, 2], "c-", label="right actual")
    ax3d.set_title("TCP Trajectories")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.legend()

    ax_err = fig.add_subplot(2, 2, 2)
    ax_err.plot(t, left_error * 1000.0, label="left")
    ax_err.plot(t, right_error * 1000.0, label="right")
    ax_err.set_title("TCP Position Error")
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("error (mm)")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend()

    ax_q = fig.add_subplot(2, 1, 2)
    controlled = set(meta.get("target_joint_names", []))
    joint_names = meta["joint_names"]
    controlled_indices = [
        index for index, name in enumerate(joint_names) if name in controlled
    ]
    image = qpos[:, controlled_indices].T if controlled_indices else qpos.T
    im = ax_q.imshow(image, aspect="auto", interpolation="nearest", cmap="viridis")
    ax_q.set_title("Controlled Joint Qpos")
    ax_q.set_xlabel("frame")
    ax_q.set_ylabel("joint")
    labels = [joint_names[i] for i in controlled_indices] if controlled_indices else joint_names
    ax_q.set_yticks(np.arange(len(labels)))
    ax_q.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=ax_q, fraction=0.02, pad=0.01)

    fig.suptitle(
        f"{meta.get('backend')} | frames={len(frames)} | "
        f"left max={left_error.max()*1000:.2f}mm | "
        f"right max={right_error.max()*1000:.2f}mm"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize retarget output with FK validation.")
    parser.add_argument("--pickle", required=True, help="Retarget output pickle.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    args = parser.parse_args()

    payload = _load_payload(args.pickle)
    if payload.get("meta_data", {}).get("backend") != "dex_bimanual":
        raise ValueError("visualize_output currently expects a dex_bimanual pickle")
    _plot_bimanual(payload, Path(args.output).expanduser().resolve())
    print(f"saved visualization to {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
    main()
