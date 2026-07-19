# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""MuJoCo + egoview synthesis for Galbot retarget trajectories.

Builds LeRobot ``observation.images.egoview`` via SAM2 → ProPainter → robot
overlay, and live MuJoCo front/top views for Rerun.

Adapted from Open-AoE Phantom Stage-3 (segment → inpaint → overlay); this repo
uses ProPainter for inpainting.
"""

from __future__ import annotations

from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path

import cv2
import mujoco
import numpy as np

from retarget_galbot.constants.aoe import HANDS_RECON_SUBDIRS
from retarget_galbot.robots import RobotSpec, get_spec

logger = logging.getLogger(__name__)


# Fingertip + MCP indices for SAM2 seeding
_SEED_KPT_IDS = [4, 8, 12, 16, 20, 5, 9, 13, 17]


def _sam2_autocast_context(device: str):
    """Enable BF16 autocast only for CUDA SAM2 inference."""
    if str(device).startswith("cuda"):
        import torch

        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


# ── Camera intrinsics ───────────────────────────────────────────────────────


def _load_hands_focal(episode_dir: Path) -> float | None:
    """Load HaWoR focal length from hands.npz when available."""
    for sub in HANDS_RECON_SUBDIRS:
        hands_path = episode_dir / sub / "hands.npz"
        if hands_path.exists():
            return float(np.load(hands_path)["focal"])
    return None


def _load_camera_intrinsics(
    episode_dir: Path,
    target_w: int,
    target_h: int,
) -> tuple[float, float, float, float, int, int]:
    """Load camera intrinsics and compute scaled (fx, fy, cx, cy).

    Uses HaWoR ``hands.npz`` focal for projection when available, since 3D
    keypoints are reconstructed in that camera model. Principal point and
    resolution come from ``undistorted_video_info.json`` when present.

    Returns (fx_scaled, fy_scaled, cx_scaled, cy_scaled, orig_w, orig_h).
    """
    episode_dir = Path(episode_dir)
    fx_orig = fy_orig = cx_orig = cy_orig = None
    orig_w = orig_h = None

    # Try undistorted_video_info.json
    from retarget_galbot.constants.aoe import UNDISTORTED_VIDEO_SUBDIRS
    for sub in UNDISTORTED_VIDEO_SUBDIRS:
        info_path = episode_dir / sub / "undistorted_video_info.json"
        if not info_path.exists():
            continue
        try:
            with open(info_path) as f:
                info = json.load(f)
            cp = info.get("cameraParams", {})
            fx_orig = cp.get("fx_pixels")
            fy_orig = cp.get("fy_pixels")
            cx_orig = cp.get("cx_pixels")
            cy_orig = cp.get("cy_pixels")
            res_str = cp.get("resolution", "")
            if "x" in res_str:
                parts = res_str.split("x")
                orig_w, orig_h = int(parts[0]), int(parts[1])
            if fx_orig is not None and orig_w is not None:
                break
        except Exception:
            continue

    # Fallback: detect resolution from video
    if orig_w is None:
        for sub in UNDISTORTED_VIDEO_SUBDIRS:
            vdir = episode_dir / sub
            if not vdir.exists():
                continue
            for f in vdir.iterdir():
                if f.suffix == ".mp4":
                    cap = cv2.VideoCapture(str(f))
                    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    break
            if orig_w is not None:
                break

        if orig_w is None:
            orig_w, orig_h = 1920, 1080

    hands_focal = _load_hands_focal(episode_dir)
    if hands_focal is not None:
        logger.info(
            "Using hands.npz focal %.1f for keypoint projection",
            hands_focal,
        )
        fx_orig = fy_orig = hands_focal
    elif fx_orig is None or fy_orig is None:
        fx_orig = fy_orig = 788.0

    if cx_orig is None:
        cx_orig = orig_w / 2.0
        cy_orig = orig_h / 2.0
    elif (
        abs(float(cx_orig) - orig_w / 2.0) > 0.1 * orig_w
        or abs(float(cy_orig) - orig_h / 2.0) > 0.1 * orig_h
    ):
        logger.info(
            "Camera principal point %.1f, %.1f is inconsistent with video "
            "resolution %dx%d; using image center for keypoint projection",
            cx_orig,
            cy_orig,
            orig_w,
            orig_h,
        )
        cx_orig = orig_w / 2.0
        cy_orig = orig_h / 2.0

    sx = target_w / orig_w
    sy = target_h / orig_h
    return (fx_orig * sx, fy_orig * sy, cx_orig * sx, cy_orig * sy, orig_w, orig_h)


def _project_3d_to_pixel(
    pts_3d: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) cam-local points to pixel coords.

    Returns (pixels (N, 2), in_frame (N,) bool).
    """
    N = pts_3d.shape[0]
    pix = np.zeros((N, 2), dtype=np.float32)
    in_frame = np.zeros(N, dtype=bool)

    z = pts_3d[:, 2]
    valid_z = z > 0.05
    if valid_z.any():
        u = fx * pts_3d[valid_z, 0] / z[valid_z] + cx
        v = fy * pts_3d[valid_z, 1] / z[valid_z] + cy
        pix[valid_z, 0] = u
        pix[valid_z, 1] = v
        in_frame[valid_z] = (u >= 0) & (u < width) & (v >= 0) & (v < height)

    return pix, in_frame


# ── Helpers ─────────────────────────────────────────────────────────────────


def _draw_label(img: np.ndarray, text: str) -> np.ndarray:
    """Overlay a small white-on-black label in the top-left corner."""
    out = img.copy()
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(out, (4, 4), (8 + tw, 8 + th + baseline), (0, 0, 0), thickness=-1)
    cv2.putText(
        out, text, (6, 6 + th),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


def _read_first_n_frames(
    video_path: Path,
    n: int,
    out_size: tuple[int, int] | None = None,
    start_frame: int = 0,
) -> np.ndarray:
    """Read n frames from an MP4 as RGB uint8 (n, H, W, 3).

    Does NOT resample temporally — takes frames sequentially from start_frame.
    If out_size is given as (width, height), frames are resized while reading
    to avoid holding full-resolution video in memory.
    """
    default_h, default_w = (out_size[1], out_size[0]) if out_size else (480, 640)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video: %s", video_path)
        return np.zeros((n, default_h, default_w, 3), dtype=np.uint8)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if out_size is not None:
            rgb = cv2.resize(rgb, out_size, interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    cap.release()

    if not frames:
        return np.zeros((n, default_h, default_w, 3), dtype=np.uint8)

    arr = np.stack(frames)
    if arr.shape[0] < n:
        pad = np.broadcast_to(arr[-1:], (n - arr.shape[0],) + arr.shape[1:]).copy()
        arr = np.concatenate([arr, pad], axis=0)
    return arr


def _infer_video_start_frame(
    video_path: Path,
    n_action_frames: int,
    episode_dir: Path | None = None,
) -> int:
    """Infer an ego-video start frame that aligns with a shorter action stream."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    n_reference_frames = _load_keypoint_frame_count(episode_dir) or n_action_frames
    if n_video_frames <= 0 or n_video_frames <= n_reference_frames:
        return 0

    offset = n_video_frames - n_reference_frames
    logger.info(
        "Ego video has %d frames, keypoints/actions reference has %d; "
        "starting video at frame %d for overlay alignment",
        n_video_frames,
        n_reference_frames,
        offset,
    )
    return offset


def _load_keypoint_frame_count(episode_dir: Path | None) -> int | None:
    if episode_dir is None:
        return None
    episode_dir = Path(episode_dir)
    for sub in HANDS_RECON_SUBDIRS:
        sidecar_path = episode_dir / sub / "hands_keypoints.npz"
        if not sidecar_path.exists():
            continue
        try:
            sidecar = np.load(sidecar_path, mmap_mode="r")
            for key in ("left_keypoints_cam", "right_keypoints_cam"):
                if key in sidecar:
                    return int(sidecar[key].shape[0])
        except Exception:
            return None
    return None


def _setup_mujoco(spec: RobotSpec, width: int, height: int):
    """Set up MuJoCo model, data, qpos_writer for a spec."""
    model = mujoco.MjModel.from_xml_path(str(spec.mjcf_path))
    data = mujoco.MjData(model)
    qpos_writer = spec.qpos_writer_factory(model)
    setattr(qpos_writer, "_spec", spec)
    return model, data, qpos_writer


def _set_action(data, qpos_writer, action):
    """Set action into MuJoCo data."""
    spec = getattr(qpos_writer, "_spec", None)
    data.qpos[:] = 0.0
    if spec is not None and spec.has_floating_base:
        data.qpos[2] = spec.standing_height
        data.qpos[3] = 1.0
    qpos_writer(data.qpos, action)
    mujoco.mj_forward(data.model, data)


def _make_camera(azimuth: float, elevation: float, distance: float,
                 lookat: tuple = (0.0, 0.0, 1.1)) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance
    cam.lookat[:] = list(lookat)
    return cam


# ── Keypoint projection ────────────────────────────────────────────────────


def _resolve_sidecar_path(hands_dir: Path) -> Path | None:
    """Return hands_keypoints.npz, generating from hands.npz via MANO FK if needed."""
    from retarget_galbot.aoe.mano_sidecar import _ensure_sidecar

    try:
        return _ensure_sidecar(hands_dir)
    except FileNotFoundError:
        logger.warning("No hands.npz in %s — keypoints unavailable", hands_dir)
        return None


def _load_keypoint_projections(
    episode_dir: Path | None,
    T: int,
    width: int,
    height: int,
) -> dict | None:
    """Preload and project all MANO keypoints to pixel coordinates.

    Returns dict with 'left_pix' (T,21,2), 'right_pix' (T,21,2),
    'left_valid' (T,), 'right_valid' (T,), 'left_in_frame' (T,21) bool,
    'right_in_frame' (T,21) bool. Returns None if data unavailable.
    """
    if episode_dir is None:
        return None

    episode_dir = Path(episode_dir)
    hands_dir = None
    for sub in HANDS_RECON_SUBDIRS:
        candidate = episode_dir / sub
        if candidate.exists():
            hands_dir = candidate
            break
    if hands_dir is None:
        return None

    sidecar_path = _resolve_sidecar_path(hands_dir)
    if sidecar_path is None:
        return None

    sidecar = np.load(sidecar_path, allow_pickle=True)

    fx, fy, cx, cy, _, _ = _load_camera_intrinsics(episode_dir, width, height)

    result = {}
    for side in ("left", "right"):
        kpts_3d = sidecar[f"{side}_keypoints_cam"]  # (N, 21, 3)
        valid = sidecar[f"{side}_valid"].astype(bool)

        n = min(T, kpts_3d.shape[0], len(valid))
        pix = np.zeros((T, 21, 2), dtype=np.float32)
        in_frame = np.zeros((T, 21), dtype=bool)
        padded_valid = np.zeros(T, dtype=bool)
        padded_valid[:n] = valid[:n]

        for t in range(n):
            if not valid[t]:
                continue
            p, inf = _project_3d_to_pixel(kpts_3d[t], fx, fy, cx, cy, width, height)
            pix[t] = p
            in_frame[t] = inf

        result[f"{side}_pix"] = pix
        result[f"{side}_valid"] = padded_valid
        result[f"{side}_in_frame"] = in_frame

    return result



# ── Main renderers ──────────────────────────────────────────────────────────


def resolve_episode_dir(
    ego_video_path: str | Path | None = None,
    episode_dir: str | Path | None = None,
) -> Path | None:
    """Resolve AoE segment dir from an explicit path or ego-video parents."""
    if episode_dir is not None:
        return Path(episode_dir)
    if ego_video_path is None:
        return None
    ego_video_path = Path(ego_video_path)
    for parent in ego_video_path.parents:
        if (parent / "ego_process").exists():
            return parent
    return None


def prepare_ego_frames(
    actions: np.ndarray,
    ego_video_path: str | Path,
    *,
    episode_dir: str | Path | None = None,
    video_start_frame: int | None = None,
    width: int = 640,
    height: int = 480,
) -> tuple[np.ndarray, Path | None, int]:
    """Load ego RGB frames aligned to the action stream.

    Returns:
        (ego_arr (T,H,W,3) uint8, episode_dir, video_start_frame)
    """
    ego_video_path = Path(ego_video_path)
    T = int(actions.shape[0])
    ep_dir = resolve_episode_dir(ego_video_path, episode_dir)
    if video_start_frame is None:
        video_start_frame = _infer_video_start_frame(ego_video_path, T, ep_dir)
    else:
        video_start_frame = max(0, int(video_start_frame))
        logger.info("Using explicit ego video start frame: %d", video_start_frame)
    ego_arr = _read_first_n_frames(
        ego_video_path,
        T,
        out_size=(width, height),
        start_frame=video_start_frame,
    )
    return ego_arr, ep_dir, video_start_frame


def render_egoview_frames(
    actions: np.ndarray,
    ego_video_path: str | Path,
    *,
    spec: RobotSpec | None = None,
    episode_dir: str | Path | None = None,
    video_start_frame: int | None = None,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Run egoview overlay pipeline; return frames (T,H,W,3) uint8."""
    if spec is None:
        spec = get_spec("galbot")
    ego_arr, ep_dir, video_start_frame = prepare_ego_frames(
        actions,
        ego_video_path,
        episode_dir=episode_dir,
        video_start_frame=video_start_frame,
        width=width,
        height=height,
    )
    model, data, qpos_writer = _setup_mujoco(spec, width, height)
    try:
        _mask, _inpaint, egoview = _run_egoview_pipeline(
            ego_arr,
            actions,
            spec,
            model,
            data,
            qpos_writer,
            width,
            height,
            episode_dir=ep_dir,
            frame_offset=video_start_frame,
        )
    finally:
        # Renderers inside the egoview pipeline are closed there; model/data are local.
        pass
    return egoview


def render_mujoco_views(
    actions: np.ndarray,
    *,
    spec: RobotSpec | None = None,
    width: int = 640,
    height: int = 480,
    cam_azimuth_front: float = 180.0,
    cam_elevation_front: float = -12.0,
    cam_distance_front: float = 3.2,
    cam_azimuth_top: float = 90.0,
    cam_elevation_top: float = -88.0,
    cam_distance_top: float = 3.8,
    lookat: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Render MuJoCo front + top-down views for an action trajectory.

    Returns:
        (front_frames (T,H,W,3), top_frames (T,H,W,3)) uint8
    """
    if spec is None:
        spec = get_spec("galbot")
    T = int(actions.shape[0])
    model, data, qpos_writer = _setup_mujoco(spec, width, height)
    cam_front = _make_camera(
        cam_azimuth_front, cam_elevation_front, cam_distance_front, lookat
    )
    cam_top = _make_camera(
        cam_azimuth_top, cam_elevation_top, cam_distance_top, lookat
    )
    renderer_front = mujoco.Renderer(model, height=height, width=width)
    renderer_top = mujoco.Renderer(model, height=height, width=width)
    front_frames = np.empty((T, height, width, 3), dtype=np.uint8)
    top_frames = np.empty((T, height, width, 3), dtype=np.uint8)
    try:
        for t in range(T):
            _set_action(data, qpos_writer, actions[t])
            renderer_front.update_scene(data, camera=cam_front)
            front_frames[t] = renderer_front.render()
            renderer_top.update_scene(data, camera=cam_top)
            top_frames[t] = renderer_top.render()
    finally:
        renderer_front.close()
        renderer_top.close()
    return front_frames, top_frames



# ── Egoview: SAM2 + ProPainter + Overlay ───────────────────────────────────


def _run_egoview_pipeline(
    ego_arr: np.ndarray,
    actions: np.ndarray,
    spec: RobotSpec,
    model, data, qpos_writer,
    width: int,
    height: int,
    episode_dir: Path | None = None,
    frame_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run SAM2 segmentation + ProPainter + MuJoCo ego overlay.

    Returns:
        (mask_frames (T,H,W) bool,
         inpaint_frames (T,H,W,3) uint8,
         egoview_frames (T,H,W,3) uint8)
    """
    import torch
    from retarget_galbot.egoview.overlay import overlay_robot, OverlayStatsEMA

    T, H, W = ego_arr.shape[:3]

    logger.info("Egoview 1/3: SAM2 segmentation (%d frames)...", T)
    mask_frames = _segment_hands_sam2(
        ego_arr,
        episode_dir,
        width,
        height,
        frame_offset=frame_offset,
    )
    coverage = mask_frames.astype(float).mean() * 100
    logger.info("Egoview 1/3 done. Mask coverage: %.1f%%", coverage)

    logger.info("Egoview 2/3: ProPainter inpainting (%d frames)...", T)
    inpaint_frames = _run_propainter(ego_arr, mask_frames)
    logger.info("Egoview 2/3 done.")

    logger.info("Egoview 3/3: MuJoCo ego render + overlay (%d frames)...", T)

    from retarget_galbot.robots.mjcf_patch import (
        EGO_CAMERA_NAME,
        patch_mjcf_local,
    )

    patcher = spec.patch_mjcf or patch_mjcf_local
    patched_xml = Path(patcher(spec.mjcf_path, 70.0))
    ego_model = mujoco.MjModel.from_xml_path(str(patched_xml))
    ego_data = mujoco.MjData(ego_model)
    ego_qpos_writer = spec.qpos_writer_factory(ego_model)
    setattr(ego_qpos_writer, "_spec", spec)

    ego_cam_id = mujoco.mj_name2id(
        ego_model, mujoco.mjtObj.mjOBJ_CAMERA, EGO_CAMERA_NAME
    )

    lid = mujoco.mj_name2id(ego_model, mujoco.mjtObj.mjOBJ_BODY,
                            spec.left_shoulder_body)
    rid = mujoco.mj_name2id(ego_model, mujoco.mjtObj.mjOBJ_BODY,
                            spec.right_shoulder_body)
    ego_data.qpos[:] = 0.0
    if spec.has_floating_base:
        ego_data.qpos[2] = spec.standing_height
        ego_data.qpos[3] = 1.0
    mujoco.mj_forward(ego_model, ego_data)
    shoulder_anchor = 0.5 * (ego_data.xpos[lid] + ego_data.xpos[rid])

    arm_body_ids = set()
    arm_geom_ids = []
    for body_id in range(ego_model.nbody):
        body_name = (
            mujoco.mj_id2name(ego_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        )
        if spec.hand_body_predicate(body_name):
            arm_body_ids.add(body_id)
    for gid in range(ego_model.ngeom):
        if int(ego_model.geom_bodyid[gid]) in arm_body_ids:
            arm_geom_ids.append(gid)
    if not arm_geom_ids:
        raise RuntimeError("No arm/gripper geoms matched the robot body predicate")
    arm_geom_ids_array = np.asarray(arm_geom_ids, dtype=np.int32)
    hidden_geom_mask = np.ones(ego_model.ngeom, dtype=bool)
    hidden_geom_mask[arm_geom_ids_array] = False
    ego_model.geom_rgba[hidden_geom_mask, 3] = 0.0

    renderer_ego = mujoco.Renderer(ego_model, height=height, width=width)
    renderer_seg = mujoco.Renderer(ego_model, height=height, width=width)
    renderer_seg.enable_segmentation_rendering()

    scene_ema = OverlayStatsEMA(alpha=0.1)
    egoview_frames = np.empty_like(ego_arr)

    cam_offset = np.array([0.12, 0.0, 0.12])

    for t in range(T):
        ego_data.qpos[:] = 0.0
        if spec.has_floating_base:
            ego_data.qpos[2] = spec.standing_height
            ego_data.qpos[3] = 1.0
        ego_qpos_writer(ego_data.qpos, actions[t])
        mujoco.mj_forward(ego_model, ego_data)

        lwrist_id = mujoco.mj_name2id(
            ego_model, mujoco.mjtObj.mjOBJ_BODY, spec.left_wrist_body)
        rwrist_id = mujoco.mj_name2id(
            ego_model, mujoco.mjtObj.mjOBJ_BODY, spec.right_wrist_body)
        wrist_mid = 0.5 * (ego_data.xpos[lwrist_id] + ego_data.xpos[rwrist_id])

        cam_pos = shoulder_anchor + cam_offset

        fwd = wrist_mid - cam_pos
        fwd_norm = np.linalg.norm(fwd)
        if fwd_norm > 1e-6:
            fwd = fwd / fwd_norm
        else:
            fwd = np.array([1.0, 0.0, 0.0])
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        right_norm = np.linalg.norm(right)
        if right_norm > 1e-6:
            right = right / right_norm
        else:
            right = np.array([0.0, 1.0, 0.0])
        up = np.cross(right, fwd)

        R_cam = np.column_stack([right, up, -fwd])

        from scipy.spatial.transform import Rotation as R_cls
        quat_xyzw = R_cls.from_matrix(R_cam).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0],
                              quat_xyzw[1], quat_xyzw[2]])

        ego_model.cam_pos[ego_cam_id] = cam_pos
        ego_model.cam_quat[ego_cam_id] = quat_wxyz
        ego_data.cam_xpos[ego_cam_id] = cam_pos
        ego_data.cam_xmat[ego_cam_id] = R_cam.flatten()

        renderer_ego.update_scene(ego_data, camera=EGO_CAMERA_NAME)
        robot_rgb = renderer_ego.render()

        renderer_seg.update_scene(ego_data, camera=EGO_CAMERA_NAME)
        seg = renderer_seg.render()
        geom_ids = seg[..., 0]
        robot_mask = np.isin(geom_ids, arm_geom_ids_array)

        egoview_frames[t] = overlay_robot(
            scene_rgb=inpaint_frames[t],
            robot_rgb=robot_rgb,
            robot_mask=robot_mask,
            feather_sigma=1.5,
            harmonize=True,
            shadow=False,
            scene_ema=scene_ema,
        )

    renderer_ego.close()
    renderer_seg.close()

    import shutil
    import tempfile

    tmpdir = patched_xml.parent
    for candidate in (tmpdir, tmpdir.parent):
        if str(candidate).startswith(tempfile.gettempdir()):
            shutil.rmtree(candidate, ignore_errors=True)
            break

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Egoview 3/3 done.")

    return mask_frames, inpaint_frames, egoview_frames


def _segment_hands_sam2(
    ego_arr: np.ndarray,
    episode_dir: Path | None = None,
    width: int = 640,
    height: int = 480,
    frame_offset: int = 0,
) -> np.ndarray:
    """Run SAM2 hand segmentation using FK keypoint projections as seeds.

    Simple approach: for each hand, pick the 3 best-spread frames with
    the most in-frame fingertip keypoints. Use those as point prompts.
    """
    T, H, W, _ = ego_arr.shape
    chunk_size = int(os.environ.get("RETARGET_SAM2_CHUNK_SIZE", "300"))
    if chunk_size > 0 and T > chunk_size:
        masks = np.zeros((T, H, W), dtype=bool)
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            logger.info("SAM2 chunk %d:%d / %d", start, end, T)
            masks[start:end] = _segment_hands_sam2(
                ego_arr[start:end],
                episode_dir,
                width,
                height,
                frame_offset=frame_offset + start,
            )
        return masks

    return _segment_hands_sam2_chunk(
        ego_arr,
        episode_dir,
        width,
        height,
        frame_offset=frame_offset,
    )


def _segment_hands_sam2_chunk(
    ego_arr: np.ndarray,
    episode_dir: Path | None = None,
    width: int = 640,
    height: int = 480,
    frame_offset: int = 0,
) -> np.ndarray:
    try:
        from retarget_galbot.egoview.sam2_segment import HandArmSegmenter
    except Exception as exc:
        logger.warning("SAM2 interface unavailable (%s); using empty masks", exc)
        T, H, W, _ = ego_arr.shape
        return np.zeros((T, H, W), dtype=bool)

    T, H, W, _ = ego_arr.shape
    segmenter = HandArmSegmenter()

    if episode_dir is None:
        logger.warning("No episode_dir for SAM2 seeding — mask will be empty")
        return np.zeros((T, H, W), dtype=bool)

    episode_dir = Path(episode_dir)
    hands_dir = None
    for sub in HANDS_RECON_SUBDIRS:
        candidate = episode_dir / sub
        if candidate.exists():
            hands_dir = candidate
            break

    if hands_dir is None:
        logger.warning("No hands_recon_dir found — mask will be empty")
        return np.zeros((T, H, W), dtype=bool)

    sidecar_path = _resolve_sidecar_path(hands_dir)
    if sidecar_path is None:
        logger.warning("Sidecar unavailable — mask will be empty")
        return np.zeros((T, H, W), dtype=bool)

    sidecar = np.load(sidecar_path, allow_pickle=True)

    fx, fy, cx, cy, _, _ = _load_camera_intrinsics(episode_dir, width, height)

    # Per-hand: find best seed frames
    per_hand_seed_frames = {}
    per_hand_seed_points = {}

    for side in ("left", "right"):
        kpts_3d = sidecar[f"{side}_keypoints_cam"]
        valid = sidecar[f"{side}_valid"].astype(bool)
        n_frames = min(T, len(valid) - frame_offset, kpts_3d.shape[0] - frame_offset)
        if n_frames <= 0:
            per_hand_seed_frames[side] = []
            per_hand_seed_points[side] = {}
            continue

        candidates = []
        for t in range(n_frames):
            global_t = frame_offset + t
            if not valid[global_t]:
                continue
            # Project fingertip + MCP keypoints
            seeds = []
            for kid in _SEED_KPT_IDS:
                z = kpts_3d[global_t, kid, 2]
                if z > 0.05:
                    u = fx * kpts_3d[global_t, kid, 0] / z + cx
                    v = fy * kpts_3d[global_t, kid, 1] / z + cy
                    if 0 <= u < width and 0 <= v < height:
                        seeds.append([u, v])
            if len(seeds) >= 3:
                candidates.append((len(seeds), t, seeds))

        candidates.sort(key=lambda x: -x[0])

        # Pick up to 3 well-spread frames
        best_frames = []
        seed_pts = {}
        for score, fidx, pts in candidates:
            if all(abs(fidx - bf) > max(T // 10, 5) for bf in best_frames) or not best_frames:
                best_frames.append(fidx)
                seed_pts[fidx] = pts
                if len(best_frames) >= 3:
                    break

        per_hand_seed_frames[side] = best_frames
        per_hand_seed_points[side] = seed_pts
        logger.info("SAM2 %s: %d seed frames %s (global offset %d, best has %d points)",
                    side, len(best_frames), best_frames, frame_offset,
                    len(seed_pts[best_frames[0]]) if best_frames else 0)

    # Run SAM2 inference
    import torch

    with torch.inference_mode(), _sam2_autocast_context(segmenter.device):
        state = segmenter._build_inference_state(ego_arr)
        segmenter.predictor.reset_state(state)

        obj_id_map = {"left": 0, "right": 1}

        for side in ("left", "right"):
            oid = obj_id_map[side]
            for fidx in per_hand_seed_frames[side]:
                seeds = per_hand_seed_points[side].get(fidx, [])
                if not seeds:
                    continue
                seed_arr = np.array(seeds, dtype=np.float32)
                segmenter.predictor.add_new_points_or_box(
                    state, frame_idx=int(fidx), obj_id=oid,
                    points=seed_arr,
                    labels=np.ones(len(seed_arr), dtype=np.int32),
                )

        masks = np.zeros((T, H, W), dtype=bool)

        def _collect(reverse: bool):
            for out_idx, _out_obj_ids, out_logits in \
                    segmenter.predictor.propagate_in_video(state, reverse=reverse):
                for i in range(out_logits.shape[0]):
                    m = (out_logits[i] > 0.0).cpu().numpy()
                    if m.ndim == 3:
                        m = m[0]
                    masks[out_idx] |= m

        _collect(reverse=False)
        _collect(reverse=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return masks


def _run_propainter(
    ego_arr: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Run ProPainter inpainting."""
    try:
        from retarget_galbot.egoview.propainter import ProPainterRunner
    except Exception as exc:
        logger.warning("ProPainter interface unavailable (%s); using original frames", exc)
        return ego_arr.copy()

    runner = ProPainterRunner()
    result = runner.run(
        ego_video=ego_arr,
        arm_mask=masks,
        neighbor_stride=5,
        ref_length=10,
    )
    return result
