from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from retarget_galbot.robocoin import (
    RoboCoinEpisode,
    RoboCoinHeadCamera,
    expand_observation_states,
    validate_eef_forward_kinematics,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = WORKSPACE_ROOT / "datasets/RoboCOIN/Galbot_G1_use_dryer"
CAMERA_CONFIG = (
    WORKSPACE_ROOT
    / "packages/retarget_galbot/configs/robocoin_galbot_g1_head_camera.json"
)
REQUIRES_ROBOCOIN_DATA = pytest.mark.skipif(
    not DATASET_ROOT.exists(),
    reason="external RoboCOIN dataset is not bundled with simrender",
)


@REQUIRES_ROBOCOIN_DATA
def test_episode_zero_expands_to_project_galbot_layout() -> None:
    episode = RoboCoinEpisode.load(DATASET_ROOT, 0)
    actions = expand_observation_states(episode.states)
    assert episode.frame_count == 658
    assert actions.shape == (658, 33)
    np.testing.assert_allclose(actions[:, :5], 0.0)
    np.testing.assert_allclose(actions[:, 5:12], episode.states[:, 5:12])
    np.testing.assert_allclose(actions[:, 31:33], episode.states[:, 3:5])
    np.testing.assert_allclose(actions[:, 12], episode.states[:, 12])
    np.testing.assert_allclose(actions[:, 13], -episode.states[:, 12])


@REQUIRES_ROBOCOIN_DATA
def test_expanded_states_reproduce_published_eef_pose() -> None:
    episode = RoboCoinEpisode.load(DATASET_ROOT, 0)
    actions = expand_observation_states(episode.states)
    report = validate_eef_forward_kinematics(actions, episode.eef_sim_pose)
    assert report["position_max_m"] < 1e-5
    assert report["rotation_max_deg"] < 1e-3


def test_camera_calibration_is_rigid_and_finite() -> None:
    camera = RoboCoinHeadCamera.load(CAMERA_CONFIG)
    assert camera.offset_in_head.shape == (3,)
    assert camera.delta_rotvec_opencv.shape == (3,)
    assert np.all(np.isfinite(camera.offset_in_head))
    assert 30.0 < camera.vertical_fov_degrees < 90.0


def test_public_gripper_joint_layout_matches_mimic_rules() -> None:
    states = np.zeros((1, 21), dtype=np.float64)
    states[0, 12] = 0.4
    states[0, 20] = 0.7

    actions = expand_observation_states(states)

    np.testing.assert_allclose(actions[0, 12:18], [0.4, -0.4, 0.4, 0.4, 0.4, -0.4])
    np.testing.assert_allclose(actions[0, 25:31], [0.7, -0.7, 0.7, 0.7, 0.7, -0.7])
