from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from retarget_agibot import AgibotG1Controller, AgibotTrajectory, build_proxy_model
from retarget_agibot.head_overlay import (
    camera_pose_in_model_frame,
    fit_model_frame_alignment,
    load_head_camera_calibration,
)
from retarget_agibot.depth_color import (
    MUJOCO_NEAR_FIELD_SCALE_C,
    VISION_BANANA_RGB_VERTICES,
    focused_depth_to_rgb,
    focused_rgb_to_depth,
    vision_banana_depth_to_rgb,
    vision_banana_rgb_to_depth,
    vision_banana_vertex_depths,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
TRAJECTORY = WORKSPACE_ROOT / "datasets/agibot/proprio_stats/proprio_stats.h5"
URDF = (
    WORKSPACE_ROOT
    / "assets/robots/G1_v2.3/G1_120s/G1_120s.urdf"
)
VISUAL_MESH_DIR = (
    WORKSPACE_ROOT / "assets/robots/G1_v2.3/G1_120s/mujoco_articulated/meshes"
)
REQUIRES_SAMPLE_DATA = pytest.mark.skipif(
    not TRAJECTORY.exists(),
    reason="external AgiBot sample dataset is not bundled with simrender",
)


def _joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id >= 0
    return float(data.qpos[int(model.jnt_qposadr[joint_id])])


def test_bundled_official_model_loads_without_sample_data() -> None:
    model = build_proxy_model(URDF, VISUAL_MESH_DIR)
    assert model.nmesh == 43
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "idx67_arm_r_joint7") >= 0


@REQUIRES_SAMPLE_DATA
def test_trajectory_layout_and_gripper_semantics() -> None:
    trajectory = AgibotTrajectory.load(TRAJECTORY)
    assert trajectory.frame_count == 2525
    assert np.isclose(trajectory.fps, 30.0, atol=0.1)
    assert np.allclose(trajectory.gripper_closed_fraction[0], 0.0)
    assert np.allclose(trajectory.gripper_closed_fraction[500], 1.0)
    assert np.allclose(trajectory.state_gripper_closed_fraction[0], 0.0)
    assert np.allclose(trajectory.state_gripper_closed_fraction[500], 1.0, atol=0.01)
    assert trajectory.limit_report(URDF)["total_violation_count"] == 0


@REQUIRES_SAMPLE_DATA
def test_mesh_model_has_all_controlled_joints_and_replays_a_frame() -> None:
    trajectory = AgibotTrajectory.load(TRAJECTORY)
    model = build_proxy_model(URDF, VISUAL_MESH_DIR)
    controller = AgibotG1Controller(model)
    data = controller.set_frame(trajectory, 1000)
    assert np.all(np.isfinite(data.qpos))
    assert model.nmesh == 43
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "idx67_arm_r_joint7") >= 0
    assert controller.clip_counts == {}


@REQUIRES_SAMPLE_DATA
def test_120s_gripper_inverts_dataset_closed_fraction() -> None:
    trajectory = AgibotTrajectory.load(TRAJECTORY)
    model = build_proxy_model(URDF, VISUAL_MESH_DIR)
    controller = AgibotG1Controller(model)
    open_data = controller.set_frame(trajectory, 0)
    assert np.isclose(
        _joint_qpos(model, open_data, "idx41_gripper_l_outer_joint1"), 1.0
    )

    closed_data = controller.set_frame(trajectory, 500)

    expected = {
        "idx41_gripper_l_outer_joint1": 0.0,
        "idx31_gripper_l_inner_joint1": 0.0,
        "idx49_gripper_l_outer_joint2": 0.0,
        "idx39_gripper_l_inner_joint2": 0.0,
        "idx42_gripper_l_outer_joint3": 0.0,
        "idx32_gripper_l_inner_joint3": 0.0,
        "idx43_gripper_l_outer_joint4": 0.0,
        "idx33_gripper_l_inner_joint4": 0.0,
    }
    for name, value in expected.items():
        assert np.isclose(_joint_qpos(model, closed_data, name), value, atol=0.01)

    action_data = controller.set_frame(trajectory, 500, joint_source="action")
    for name, value in expected.items():
        assert np.isclose(_joint_qpos(model, action_data, name), value)


@REQUIRES_SAMPLE_DATA
def test_head_camera_calibration_matches_video_and_trajectory() -> None:
    trajectory = AgibotTrajectory.load(TRAJECTORY)
    calibration = load_head_camera_calibration(
        WORKSPACE_ROOT / "datasets/agibot",
        expected_frame_count=trajectory.frame_count,
    )
    assert (calibration.width, calibration.height) == (640, 480)
    assert calibration.native_render_height == 360
    assert calibration.rotations_data_from_camera.shape == (2525, 3, 3)
    assert calibration.translations_data_from_camera.shape == (2525, 3)
    assert np.allclose(np.linalg.det(calibration.rotations_data_from_camera), 1.0)


@REQUIRES_SAMPLE_DATA
def test_model_frame_alignment_reproduces_recorded_end_positions() -> None:
    trajectory = AgibotTrajectory.load(TRAJECTORY)
    model = build_proxy_model(URDF, VISUAL_MESH_DIR)
    controller = AgibotG1Controller(model)
    alignment = fit_model_frame_alignment(
        model,
        controller,
        trajectory,
        sample_stride=100,
    )
    assert alignment.rmse_m < 1e-3

    data = controller.set_frame(trajectory, 0, base_mode="fixed")
    for side_index, body_name in enumerate(("arm_l_end_link", "arm_r_end_link")):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        predicted = (
            alignment.rotation_data_from_model @ data.xpos[body_id]
            + alignment.translation_data_from_model
        )
        np.testing.assert_allclose(
            predicted,
            trajectory.end_position[0, side_index],
            atol=1e-4,
        )

    calibration = load_head_camera_calibration(
        WORKSPACE_ROOT / "datasets/agibot",
        expected_frame_count=trajectory.frame_count,
    )
    position, rotation = camera_pose_in_model_frame(calibration, alignment, 0)
    np.testing.assert_allclose(position, [0.351416, -0.0121, 1.542923], atol=2e-4)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)


def test_vision_banana_metric_depth_color_vertices() -> None:
    depths = vision_banana_vertex_depths()
    colors = vision_banana_depth_to_rgb(depths[:-1])
    expected = np.round(VISION_BANANA_RGB_VERTICES[:-1] * 255.0).astype(np.uint8)
    np.testing.assert_allclose(colors, expected, atol=1)
    assert np.all(np.diff(depths[:-1]) > 0.0)
    np.testing.assert_allclose(
        depths[:-1],
        [0.0, 0.8012, 1.8322, 3.2288, 5.2753, 8.7083, 16.4575],
        atol=1e-3,
    )
    far = vision_banana_depth_to_rgb(np.asarray([1e8], dtype=np.float32))[0]
    np.testing.assert_allclose(far, [255, 255, 255], atol=1)
    near_field_depths = vision_banana_vertex_depths(
        scale_c=MUJOCO_NEAR_FIELD_SCALE_C
    )
    np.testing.assert_allclose(
        near_field_depths[:-1],
        [0.0, 0.1202, 0.2748, 0.4843, 0.7913, 1.3062, 2.4686],
        atol=1e-3,
    )
    samples = np.linspace(0.05, 1.5, 1000, dtype=np.float32)
    encoded = vision_banana_depth_to_rgb(
        samples,
        scale_c=MUJOCO_NEAR_FIELD_SCALE_C,
    )
    decoded = vision_banana_rgb_to_depth(
        encoded,
        scale_c=MUJOCO_NEAR_FIELD_SCALE_C,
    )
    assert float(np.max(np.abs(decoded - samples))) < 0.002

    focused_vertices = focused_depth_to_rgb(
        np.asarray([0.0, 0.12, 0.256, 0.392, 0.528, 0.664, 0.8])
    )
    np.testing.assert_allclose(
        focused_vertices,
        np.round(VISION_BANANA_RGB_VERTICES[:-1] * 255.0).astype(np.uint8),
        atol=1,
    )
    focused_samples = np.linspace(0.12, 0.8, 1000, dtype=np.float32)
    focused_decoded = focused_rgb_to_depth(focused_depth_to_rgb(focused_samples))
    assert float(np.max(np.abs(focused_decoded - focused_samples))) < 0.001
