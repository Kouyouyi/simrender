from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from retarget_galbot.egoview.chest_overlay import (
    calibrated_vertical_fov,
    chest_camera_pose_from_shoulders,
    legacy_camera_pose,
)
from retarget_galbot.galaxea.bimanual import BimanualDexRetargeter
from retarget_galbot.robots import get_spec


def test_bundled_galbot_model_loads() -> None:
    spec = get_spec("galbot")
    assert spec.mjcf_path.exists()
    assert (spec.mjcf_path.parent / "SOURCE.md").exists()
    assert (spec.mjcf_path.parent / "LICENSE").exists()
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    assert spec.action_dim == 33
    assert model.njnt == spec.action_dim
    assert model.nmesh == 86
    for joint_name in spec.action_joint_names:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) >= 0


def test_chest_camera_pose_inverts_source_shoulder_offset() -> None:
    shoulder_mid = np.array([0.0, 0.0, 1.4])
    source_shoulder_mid_head = np.array([0.0, -0.05, 0.0])
    rotation_base_head = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )

    position, rotation = chest_camera_pose_from_shoulders(
        shoulder_mid,
        source_shoulder_mid_head,
        rotation_base_head,
    )

    np.testing.assert_allclose(position, [0.0, 0.0, 1.35])
    np.testing.assert_allclose(rotation[:, 0], [0.0, -1.0, 0.0])  # camera right
    np.testing.assert_allclose(rotation[:, 1], [0.0, 0.0, 1.0])   # camera up
    np.testing.assert_allclose(-rotation[:, 2], [1.0, 0.0, 0.0])  # view forward


def test_calibrated_vertical_fov_uses_video_intrinsics(tmp_path: Path) -> None:
    info_dir = tmp_path / "ego_process" / "ego_undistorted_video"
    info_dir.mkdir(parents=True)
    (info_dir / "undistorted_video_info.json").write_text(
        '{"cameraParams":{"resolution":"1920x1080","fy_pixels":1921.0,'
        '"cx_pixels":960.0,"cy_pixels":540.0}}'
    )

    fovy, cx, cy = calibrated_vertical_fov(tmp_path, 960, 540)

    expected = math.degrees(2.0 * math.atan(1080.0 / (2.0 * 1921.0)))
    assert math.isclose(fovy, expected)
    assert cx == 480.0
    assert cy == 270.0


def test_calibrated_vertical_fov_rescales_stale_full_hd_intrinsics(
    tmp_path: Path,
) -> None:
    info_dir = tmp_path / "ego_process" / "ego_undistorted_video"
    info_dir.mkdir(parents=True)
    (info_dir / "undistorted_video_info.json").write_text(
        '{"cameraParams":{"resolution":"1280x720","fy_pixels":971.662,'
        '"cx_pixels":960.75,"cy_pixels":540.75}}'
    )

    fovy, cx, cy = calibrated_vertical_fov(tmp_path, 1280, 720)

    expected = math.degrees(2.0 * math.atan((2.0 * 540.75) / (2.0 * 971.662)))
    assert math.isclose(fovy, expected)
    assert math.isclose(cx, 640.0, abs_tol=1e-9)
    assert math.isclose(cy, 360.0, abs_tol=1e-9)


def test_legacy_camera_looks_at_wrist_midpoint() -> None:
    position, rotation = legacy_camera_pose(
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.5]),
    )

    np.testing.assert_allclose(position, [0.0, 0.0, 1.0])
    expected_forward = np.array([1.0, 0.0, -0.5])
    expected_forward /= np.linalg.norm(expected_forward)
    np.testing.assert_allclose(-rotation[:, 2], expected_forward)


def test_source_scale_override_disables_episode_scaling() -> None:
    retargeter = object.__new__(BimanualDexRetargeter)
    retargeter.config = SimpleNamespace(raw={"source_scale": 1.0})

    scale = retargeter._episode_source_to_robot_scale(None, None, 0)

    assert scale == 1.0
