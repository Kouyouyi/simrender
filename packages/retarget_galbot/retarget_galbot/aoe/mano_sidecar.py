# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""AoE hands sidecar generation via MANO forward kinematics.

If ``hands_keypoints.npz`` is missing or outdated but ``hands.npz`` exists,
runs MANO FK to produce the OpenPose-21 keypoint sidecar used by Galbot
ego loaders and egoview synthesis.

Adapted from Open-AoE Phantom ``aoe/episode`` MANO FK sidecar helpers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from retarget_galbot.constants.aoe import (
    HANDS_NPZ_NAME,
    SIDECAR_NPZ_NAME,
)

logger = logging.getLogger(__name__)

SIDECAR_VERSION = 3


def _patch_legacy_mano_pickle_deps() -> None:
    """Patch old chumpy/MANO pickle dependencies for modern Python/NumPy."""
    import collections
    import inspect

    if not hasattr(inspect, "getargspec"):
        ArgSpec = collections.namedtuple(
            "ArgSpec", ["args", "varargs", "keywords", "defaults"]
        )

        def getargspec(func):
            spec = inspect.getfullargspec(func)
            return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

        inspect.getargspec = getargspec

    legacy_numpy_aliases = {
        "bool": np.bool_,
        "int": int,
        "long": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }
    for name, value in legacy_numpy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _sidecar_needs_regeneration(sidecar_path: Path) -> bool:
    """Return True when sidecar is missing or was generated with an old schema."""
    if not sidecar_path.exists():
        return True
    try:
        data = np.load(sidecar_path, allow_pickle=True)
        meta_raw = data.get("metadata")
        if meta_raw is None:
            return True
        meta = json.loads(str(meta_raw))
        return int(meta.get("version", 0)) < SIDECAR_VERSION
    except Exception:
        return True



def _ensure_sidecar(hands_recon_dir: Path) -> Path:
    """Ensure hands_keypoints.npz exists; generate via MANO FK if missing.

    Returns path to the sidecar file.
    """
    sidecar_path = hands_recon_dir / SIDECAR_NPZ_NAME
    if sidecar_path.exists() and not _sidecar_needs_regeneration(sidecar_path):
        return sidecar_path

    hands_path = hands_recon_dir / HANDS_NPZ_NAME
    if not hands_path.exists():
        raise FileNotFoundError(
            f"Neither {sidecar_path} nor {hands_path} found. "
            "Cannot generate sidecar without hands.npz."
        )

    if sidecar_path.exists():
        logger.info("Regenerating outdated sidecar via MANO FK: %s", sidecar_path)
    else:
        logger.info("Sidecar missing, generating via MANO FK: %s", sidecar_path)
    _generate_sidecar_from_hands(hands_path, sidecar_path)
    return sidecar_path


def _generate_sidecar_from_hands(hands_path: Path, sidecar_path: Path) -> None:
    """Run MANO forward kinematics to produce hands_keypoints.npz.

    CRITICAL: cam-space keypoints must be computed via world-space FK + R_w2c
    transform, NOT via direct cam-param FK. MANO LBS rotates around J_root
    (rest-pose wrist), so direct cam FK introduces a (I - R_w2c) @ J_root
    offset that grows with camera rotation (100-350px in practice).
    """
    from retarget_galbot.constants import MANO_MODELS_DIR, SHARED_MANO_DIR

    # Use shared MANO models if local directory is empty
    _mano_dir = MANO_MODELS_DIR
    if not (_mano_dir / "MANO_RIGHT.pkl").exists() and (SHARED_MANO_DIR / "MANO_RIGHT.pkl").exists():
        _mano_dir = SHARED_MANO_DIR

    hands = np.load(hands_path)
    T = hands["pred_hand_pose"].shape[1]

    has_world = "pred_rot" in hands and "pred_trans" in hands
    has_w2c = "R_w2c" in hands and "t_w2c" in hands

    arrays = {}

    for side_idx, side in enumerate(["left", "right"]):
        valid = hands["pred_valid"][side_idx].astype(bool)
        hand_pose = hands["pred_hand_pose"][side_idx]  # (T, 45)
        betas = hands["pred_betas"][side_idx]           # (T, 10)

        # World-space FK (always generated if data available)
        if has_world:
            rot_w = hands["pred_rot"][side_idx]      # (T, 3)
            trans_w = hands["pred_trans"][side_idx]   # (T, 3)
            kpts_world, wpose_world = _run_mano_fk(
                side, rot_w, trans_w, hand_pose, betas, valid,
                _mano_dir,
            )
            arrays[f"{side}_keypoints_world"] = kpts_world.astype(np.float32)
            arrays[f"{side}_wrist_pose_world"] = wpose_world.astype(np.float32)

        # Cam-space: transform world FK results via R_w2c / t_w2c
        if has_world and has_w2c:
            R_w2c = hands["R_w2c"]  # (T, 3, 3)
            t_w2c = hands["t_w2c"]  # (T, 3)
            kpts_cam = np.zeros_like(kpts_world)
            wpose_cam = np.zeros_like(wpose_world)
            for t in range(T):
                if not valid[t]:
                    continue
                kpts_cam[t] = (R_w2c[t] @ kpts_world[t].T).T + t_w2c[t]
                wrist_pos_cam = kpts_cam[t, 0]
                R_wrist_w = Rotation.from_rotvec(rot_w[t]).as_matrix()
                R_wrist_cam = R_w2c[t] @ R_wrist_w
                q_xyzw = Rotation.from_matrix(R_wrist_cam).as_quat()
                wpose_cam[t, :3] = wrist_pos_cam
                wpose_cam[t, 3] = q_xyzw[3]
                wpose_cam[t, 4:7] = q_xyzw[:3]
            arrays[f"{side}_keypoints_cam"] = kpts_cam.astype(np.float32)
            arrays[f"{side}_wrist_pose_cam"] = wpose_cam.astype(np.float32)
        else:
            # Fallback: direct cam FK (less accurate, only if world data missing)
            rot_c = hands["pred_rot_cam"][side_idx]
            trans_c = hands["pred_trans_cam"][side_idx]
            kpts_cam, wpose_cam = _run_mano_fk(
                side, rot_c, trans_c, hand_pose, betas, valid,
                _mano_dir,
            )
            arrays[f"{side}_keypoints_cam"] = kpts_cam.astype(np.float32)
            arrays[f"{side}_wrist_pose_cam"] = wpose_cam.astype(np.float32)

        arrays[f"{side}_valid"] = valid

    if "R_c2w" in hands and "t_c2w" in hands:
        R_c2w = hands["R_c2w"]  # (T, 3, 3)
        t_c2w = hands["t_c2w"]  # (T, 3)
        quat_wxyz = np.zeros((T, 4), dtype=np.float32)
        for t in range(T):
            r = Rotation.from_matrix(R_c2w[t])
            q_xyzw = r.as_quat()
            quat_wxyz[t] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        cam_pose = np.concatenate([t_c2w, quat_wxyz], axis=-1).astype(np.float32)
        arrays["camera_pose_world"] = cam_pose

    metadata = {
        "generator": "retarget_galbot",
        "source": str(hands_path),
        "version": SIDECAR_VERSION,
    }
    arrays["metadata"] = np.array(json.dumps(metadata))

    np.savez(sidecar_path, **arrays)
    logger.info("Generated sidecar: %s (%d frames)", sidecar_path, T)


# MANO mesh vertex indices for each fingertip (OpenPose convention).
_FINGERTIP_VERTEX_IDS = (745, 317, 444, 556, 673)
# thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip

# Reorder: MANO 16 joints → OpenPose 21 joints (without tips).
# MANO order: 0=wrist, 1-3=index, 4-6=middle, 7-9=pinky, 10-12=ring, 13-15=thumb
# OpenPose: 0=wrist, 1-3=thumb(CMC,MCP,IP), 5-7=index, 9-11=middle,
#           13-15=ring, 17-19=pinky; tips at 4,8,12,16,20 from mesh vertices.
_MANO16_TO_OPENPOSE21 = [
    0,              # 0: wrist
    13, 14, 15,     # 1-3: thumb CMC, MCP, IP
    -1,             # 4: thumb tip (from vertex)
    1, 2, 3,        # 5-7: index MCP, PIP, DIP
    -1,             # 8: index tip (from vertex)
    4, 5, 6,        # 9-11: middle MCP, PIP, DIP
    -1,             # 12: middle tip (from vertex)
    10, 11, 12,     # 13-15: ring MCP, PIP, DIP
    -1,             # 16: ring tip (from vertex)
    7, 8, 9,        # 17-19: pinky MCP, PIP, DIP
    -1,             # 20: pinky tip (from vertex)
]
_TIP_SLOTS = [4, 8, 12, 16, 20]


def _run_mano_fk(
    side: str,
    rot_aa: np.ndarray,
    trans: np.ndarray,
    hand_pose: np.ndarray,
    betas: np.ndarray,
    valid: np.ndarray,
    mano_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Run MANO forward kinematics to get 21 keypoints + wrist 7D pose.

    Does full LBS on the mesh to extract fingertip vertex positions,
    then combines with the 16 kinematic joints to form 21-point layout.

    Returns:
        keypoints: (T, 21, 3) float64
        wrist_pose: (T, 7) float64 [x,y,z, qw,qx,qy,qz]
    """
    import pickle
    import torch

    _patch_legacy_mano_pickle_deps()

    T = rot_aa.shape[0]

    mano_file = mano_dir / f"MANO_{side.upper()}.pkl"
    if not mano_file.exists():
        raise FileNotFoundError(f"MANO model not found: {mano_file}")

    with open(mano_file, "rb") as f:
        try:
            mano_data = pickle.load(f, encoding="latin1")
        except ModuleNotFoundError as e:
            if e.name == "chumpy":
                raise ModuleNotFoundError(
                    "MANO pickle loading requires the legacy package 'chumpy'. "
                    "Install it in the active environment with: "
                    "python -m pip install --no-build-isolation chumpy==0.70"
                ) from e
            raise

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MANO pkl stores some arrays as chumpy objects; force to numpy first
    shapedirs = torch.tensor(
        np.array(mano_data["shapedirs"]), dtype=torch.float32, device=device)
    posedirs = torch.tensor(
        np.array(mano_data["posedirs"]), dtype=torch.float32, device=device)
    v_template = torch.tensor(
        np.array(mano_data["v_template"]), dtype=torch.float32, device=device)
    J_regressor = torch.tensor(
        np.array(mano_data["J_regressor"].todense()),
        dtype=torch.float32, device=device)
    weights = torch.tensor(
        np.array(mano_data["weights"]), dtype=torch.float32, device=device)
    kintree_table = mano_data["kintree_table"]

    tip_vids = list(_FINGERTIP_VERTEX_IDS)

    keypoints_all = np.zeros((T, 21, 3), dtype=np.float64)
    wrist_pose_all = np.zeros((T, 7), dtype=np.float64)

    batch_size = 64
    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        if not np.any(valid[start:end]):
            continue

        b_rot = torch.tensor(rot_aa[start:end], dtype=torch.float32, device=device)
        b_trans = torch.tensor(trans[start:end], dtype=torch.float32, device=device)
        b_pose = torch.tensor(hand_pose[start:end], dtype=torch.float32, device=device)
        b_betas = torch.tensor(betas[start:end], dtype=torch.float32, device=device)
        B = b_rot.shape[0]

        # 1. Shape blend shapes
        v_shaped = v_template + torch.einsum("vcs,bs->bvc", shapedirs, b_betas)
        J = torch.einsum("jv,bvc->bjc", J_regressor, v_shaped)

        # 2. Pose
        full_pose = torch.cat([b_rot, b_pose], dim=-1).reshape(B, 16, 3)
        rot_mats = _batch_rodrigues(full_pose.reshape(-1, 3)).reshape(B, 16, 3, 3)

        # 3. Pose blend shapes
        ident = torch.eye(3, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).reshape(B, -1)
        v_posed = v_shaped + torch.einsum("vcp,bp->bvc", posedirs, pose_feature)

        # 4. Build kinematic chain → world transforms
        world_transforms = _kinematic_chain(J, rot_mats, kintree_table)
        joint_pos_16 = world_transforms[:, :, :3, 3]  # (B, 16, 3)

        # 5. Full LBS on fingertip vertices only
        tip_pos = _lbs_vertices(
            v_posed[:, tip_vids], J, world_transforms, weights[tip_vids])

        # 6. Assemble 21 keypoints
        kpt21 = torch.zeros(B, 21, 3, device=device)
        for out_idx, mano_idx in enumerate(_MANO16_TO_OPENPOSE21):
            if mano_idx >= 0:
                kpt21[:, out_idx] = joint_pos_16[:, mano_idx]
        for slot_i, slot in enumerate(_TIP_SLOTS):
            kpt21[:, slot] = tip_pos[:, slot_i]

        kpt21_np = kpt21.cpu().numpy()
        trans_np = b_trans.cpu().numpy()
        rot0_np = rot_mats[:, 0].cpu().numpy()

        for i in range(B):
            t_idx = start + i
            if not valid[t_idx]:
                continue
            kpts = kpt21_np[i] + trans_np[i:i+1]
            keypoints_all[t_idx] = kpts

            wrist_pos = kpts[0]
            r = Rotation.from_matrix(rot0_np[i])
            q_xyzw = r.as_quat()
            wrist_pose_all[t_idx, :3] = wrist_pos
            wrist_pose_all[t_idx, 3] = q_xyzw[3]
            wrist_pose_all[t_idx, 4:7] = q_xyzw[:3]

    return keypoints_all, wrist_pose_all


def _batch_rodrigues(rot_vecs):
    """Axis-angle (N, 3) → rotation matrices (N, 3, 3)."""
    import torch

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.cos(angle).unsqueeze(-1)
    sin = torch.sin(angle).unsqueeze(-1)

    rx, ry, rz = rot_dir[:, 0:1], rot_dir[:, 1:2], rot_dir[:, 2:3]
    zeros = torch.zeros_like(rx)
    K = torch.cat([
        zeros, -rz, ry,
        rz, zeros, -rx,
        -ry, rx, zeros,
    ], dim=1).reshape(-1, 3, 3)

    eye = torch.eye(3, device=rot_vecs.device).unsqueeze(0)
    return eye + sin * K + (1 - cos) * torch.bmm(K, K)


def _kinematic_chain(J, rot_mats, kintree_table):
    """Build world 4x4 transforms for each joint in the kinematic chain.

    Returns:
        (B, n_joints, 4, 4) world transforms.
    """
    import torch

    B, n_joints = rot_mats.shape[:2]
    device = J.device

    transforms = torch.zeros(B, n_joints, 4, 4, device=device)
    for i in range(n_joints):
        if i == 0:
            transforms[:, i, :3, :3] = rot_mats[:, i]
            transforms[:, i, :3, 3] = J[:, i]
        else:
            parent = int(kintree_table[0, i])
            local_t = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1).clone()
            local_t[:, :3, :3] = rot_mats[:, i]
            local_t[:, :3, 3] = J[:, i] - J[:, parent]
            transforms[:, i] = torch.bmm(transforms[:, parent].clone(), local_t)
        transforms[:, i, 3, 3] = 1.0
    return transforms


def _lbs_vertices(v_posed_subset, J, world_transforms, weights_subset):
    """Apply LBS to a subset of vertices.

    Args:
        v_posed_subset: (B, V_sub, 3) posed vertex positions (rest pose)
        J: (B, 16, 3) rest-pose joint positions
        world_transforms: (B, 16, 4, 4) world transforms from kinematic chain
        weights_subset: (V_sub, 16) skinning weights for the subset

    Returns:
        (B, V_sub, 3) world-space vertex positions
    """
    import torch

    B = v_posed_subset.shape[0]
    V = v_posed_subset.shape[1]
    n_joints = world_transforms.shape[1]
    device = v_posed_subset.device

    # Subtract rest-pose joint position to get "from rest" transforms
    rest_inv = torch.zeros(B, n_joints, 4, 4, device=device)
    rest_inv[:, :, :3, :3] = torch.eye(3, device=device)
    rest_inv[:, :, :3, 3] = -J
    rest_inv[:, :, 3, 3] = 1.0

    # Combined transform: T_j = world_j @ inv(rest_j)
    T = torch.bmm(
        world_transforms.reshape(B * n_joints, 4, 4),
        rest_inv.reshape(B * n_joints, 4, 4),
    ).reshape(B, n_joints, 4, 4)

    # Blend transforms per vertex using skinning weights: (V_sub, 16)
    W = weights_subset.unsqueeze(0).expand(B, -1, -1)  # (B, V_sub, 16)
    # (B, V_sub, 4, 4) = sum over joints of w_j * T_j
    T_blended = torch.einsum("bvj,bjmn->bvmn", W, T)

    # Apply blended transform to each vertex
    v_homo = torch.ones(B, V, 4, device=device)
    v_homo[:, :, :3] = v_posed_subset
    v_world = torch.einsum("bvmn,bvn->bvm", T_blended, v_homo)

    return v_world[:, :, :3]
