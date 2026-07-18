# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Edge-fusion robot overlay for Galbot egoview synthesis.

Adapted from Open-AoE Phantom M3 compositing (Reinhard LAB harmonization,
feathered alpha, optional contact shadow).

Pastes the MuJoCo-rendered Galbot onto a ProPainter-inpainted background and
fixes paste artifacts (color mismatch, hard edges, optional contact shadow).
Pure numpy + cv2 — cheap to import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

# ── Per-episode rolling scene stats (used to damp temporal flicker) ──────────

@dataclass
class OverlayStatsEMA:
    """Exponential moving average of (L_mean, L_std, a_mean, a_std, b_mean, b_std).

    The harmonizer recomputes scene stats every frame from a ring sampled
    around the robot; raw per-frame stats jitter (the ring moves with the
    robot, and the background changes shot-to-shot). An EMA over the stats
    keeps the harmonization smooth across time so the robot does not visibly
    pulse in color.
    """
    alpha: float = 0.1
    state: Optional[np.ndarray] = field(default=None)   # shape (6,)

    def update(self, stats: np.ndarray) -> np.ndarray:
        if self.state is None:
            self.state = stats.astype(np.float64).copy()
        else:
            self.state = (1.0 - self.alpha) * self.state + self.alpha * stats
        return self.state.copy()


# ── Color harmonization (Reinhard mean+std in LAB) ───────────────────────────

def _lab_stats(img_lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return shape-(6,) [L_mu, L_std, a_mu, a_std, b_mu, b_std] over masked pixels."""
    sel = img_lab[mask]
    if len(sel) == 0:
        # Degenerate ring (robot fills frame). Use a fallback neutral.
        return np.array([128.0, 1.0, 128.0, 1.0, 128.0, 1.0], dtype=np.float64)
    mu = sel.mean(axis=0)
    sd = sel.std(axis=0) + 1e-6
    return np.array([mu[0], sd[0], mu[1], sd[1], mu[2], sd[2]], dtype=np.float64)


def harmonize_color(
    robot_rgb: np.ndarray,             # (H, W, 3) uint8
    robot_mask: np.ndarray,            # (H, W) bool
    scene_rgb: np.ndarray,             # (H, W, 3) uint8
    l_strength: float = 0.4,
    ab_strength: float = 0.7,
    ring_dilate: int = 30,
    exclude_mask: Optional[np.ndarray] = None,   # extra mask (e.g. arm mask) to exclude from sampling
    scene_ema: Optional[OverlayStatsEMA] = None,
) -> np.ndarray:
    """Match the robot's LAB mean/std to the scene's, sampled from a ring
    around the robot. L and ab are weighted separately because the chroma
    mismatch is the dominant artifact while luminance is mostly fine.

    Returns the harmonized robot RGB (uint8). Pixels outside the robot mask
    are unchanged.
    """
    H, W, _ = robot_rgb.shape
    if not robot_mask.any():
        return robot_rgb.copy()

    # Ring sample: dilate(mask, ring_dilate) − mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_dilate * 2 + 1,) * 2)
    dilated = cv2.dilate(robot_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = dilated & ~robot_mask
    if exclude_mask is not None:
        ring &= ~exclude_mask.astype(bool)

    scene_lab = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    scene_stats = _lab_stats(scene_lab, ring)
    if scene_ema is not None:
        scene_stats = scene_ema.update(scene_stats)

    robot_lab = cv2.cvtColor(robot_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    robot_stats = _lab_stats(robot_lab, robot_mask)

    # Map each channel toward (scene_mu, scene_std)
    out = robot_lab.copy()
    for c, strength in enumerate([l_strength, ab_strength, ab_strength]):
        r_mu, r_sd = robot_stats[2 * c], robot_stats[2 * c + 1]
        s_mu, s_sd = scene_stats[2 * c], scene_stats[2 * c + 1]
        # Target = ((x − r_mu) * s_sd/r_sd) + s_mu; lerp by strength.
        target = ((robot_lab[..., c] - r_mu) * (s_sd / r_sd)) + s_mu
        out[..., c] = (1.0 - strength) * robot_lab[..., c] + strength * target

    out = np.clip(out, 0, 255).astype(np.uint8)
    out_rgb = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)

    # Only modify inside the robot mask.
    result = robot_rgb.copy()
    result[robot_mask] = out_rgb[robot_mask]
    return result


# ── Feathered alpha (avoid silhouette shrink) ────────────────────────────────

def feather_alpha(mask: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Soft alpha [0,1] from a binary mask.

    A symmetric Gaussian on a binary mask shrinks the silhouette by ~sigma
    pixels — fingers in particular look thinner. We pre-dilate by ceil(sigma)
    so that after blurring the silhouette is restored to its original extent
    while the outer edge has a smooth gradient.
    """
    if sigma <= 0:
        return mask.astype(np.float32)
    r = int(np.ceil(sigma))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dil = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
    ksize = 2 * int(np.ceil(3.0 * sigma)) + 1
    alpha = cv2.GaussianBlur(dil, (ksize, ksize), sigmaX=float(sigma), sigmaY=float(sigma))
    return np.clip(alpha, 0.0, 1.0)


# ── Bleed-zone extension (paint robot color into the feather radius) ─────────

def extend_robot_into_bleed_zone(
    rgb: np.ndarray,
    mask: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Grow the robot RGB outward by ``radius`` pixels via iterative ring fill.

    The MuJoCo renderer paints the robot on top of a light skybox/floor
    background. When the feathered alpha extends ~radius pixels past the
    original silhouette, the alpha-blend pulls in those bg pixels and the
    overlaid edge looks cold/white — at odds with the warm robot interior.

    This function "paints" the robot's interior colors outward into the bleed
    zone: ring-by-ring it takes each new boundary pixel's value from the
    mean of its already-filled 3×3 neighbors. After this, the downstream
    harmonize step sees an extended mask and warms the bleed zone too, so
    when the feathered alpha mixes pixels in that ring they match the warm
    robot tone instead of the cold render bg.

    Returns ``(extended_rgb, extended_mask)``. ``radius=0`` is a no-op.
    """
    if radius <= 0 or not mask.any():
        return rgb, mask
    extended_rgb = rgb.astype(np.float32).copy()
    extended_mask = mask.copy()
    # Zero out pixels outside the current mask so the box-filter mean of
    # neighbours uses only filled pixels.
    extended_rgb[~extended_mask] = 0.0
    weight = extended_mask.astype(np.float32)
    kernel3 = np.ones((3, 3), np.uint8)
    for _ in range(radius):
        # Compute the next ring (pixels just outside current mask).
        dil_mask = cv2.dilate(extended_mask.astype(np.uint8), kernel3,
                              iterations=1).astype(bool)
        ring = dil_mask & ~extended_mask
        if not ring.any():
            break
        # Mean of filled 3×3 neighbours.
        blurred = cv2.boxFilter(extended_rgb, ddepth=-1, ksize=(3, 3), normalize=False)
        wblur = cv2.boxFilter(weight, ddepth=-1, ksize=(3, 3), normalize=False)
        safe = ring & (wblur > 0)
        if not safe.any():
            break
        w_safe = wblur[safe][:, None]
        b_safe = blurred[safe]
        extended_rgb[safe] = b_safe / w_safe
        # Update the mask + weight so the next iteration uses these filled pixels.
        extended_mask |= safe
        weight[safe] = 1.0
    return np.clip(extended_rgb, 0, 255).astype(np.uint8), extended_mask


# ── 1-pixel ring de-fringing (kill the dark halo from rendering on black) ────

def defringe_silhouette(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace the 1-pixel boundary ring of rgb with the local interior mean.

    MuJoCo renders the robot over a black-ish skybox/background, so the
    silhouette boundary pixels are partially blended toward black. If we
    feather-alpha-blend those into the scene we get a dark halo. Fix: for
    every pixel that is inside the mask but has any 3×3 neighbor outside
    the mask, replace it with the mean of its 3×3 interior neighbors.
    """
    if not mask.any():
        return rgb
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                       iterations=1).astype(bool)
    boundary = mask & ~eroded
    if not boundary.any():
        return rgb

    interior_rgb = rgb.astype(np.float32)
    interior_rgb[~eroded] = 0.0
    weight = eroded.astype(np.float32)
    blurred = cv2.boxFilter(interior_rgb, ddepth=-1, ksize=(3, 3), normalize=False)
    wblur = cv2.boxFilter(weight, ddepth=-1, ksize=(3, 3), normalize=False)

    safe = (wblur > 0) & boundary
    if not safe.any():
        return rgb
    w_safe = wblur[safe][:, None]               # (N, 1)
    b_safe = blurred[safe]                      # (N, 3)
    out = rgb.copy()
    out[safe] = np.clip(b_safe / w_safe, 0, 255).astype(np.uint8)
    return out


# ── Hand contact shadow (depth-gated, hand-mask only) ────────────────────────

def hand_contact_shadow(
    hand_mask: np.ndarray,             # (H, W) bool
    depth: np.ndarray,                 # (H, W) float32, meters
    z_lo: float,
    z_hi: float,
    dy: int = 8,
    sigma: float = 6.0,
    darken: float = 0.35,
) -> np.ndarray:
    """Soft shadow alpha in [0, darken] driven by the rendered hand mask.

    The shadow strength is gated by the typical depth of the hand pixel:
    shadow_alpha = clip((z − z_lo) / (z_hi − z_lo), 0, 1) — so as the hand
    moves further from the camera (toward the table in this dataset),
    shadow fades in; when the user raises their hand near the head the
    shadow fades out. Without real scene depth this is a heuristic, but it
    matches the typical EgoDex camera (looking down at a table) well.

    Output shape (H, W) float32, suitable for multiplying scene RGB by
    ``(1 - alpha[..., None])``.
    """
    if not hand_mask.any():
        return np.zeros(hand_mask.shape, dtype=np.float32)

    # Per-pixel gate: blend by depth.
    depth_clamped = np.clip(depth, z_lo, z_hi)
    if z_hi - z_lo > 1e-3:
        gate = (depth_clamped - z_lo) / (z_hi - z_lo)
    else:
        gate = np.ones_like(depth, dtype=np.float32)
    gate = gate.astype(np.float32)

    # Use the median depth within the hand mask as a single per-frame
    # "are hands near the table" scalar — avoids per-pixel jitter from
    # tiny fingers having weird depth.
    hand_depth_med = float(np.median(depth[hand_mask]))
    scalar_gate = float(np.clip(
        (hand_depth_med - z_lo) / max(z_hi - z_lo, 1e-3), 0.0, 1.0,
    ))

    base = hand_mask.astype(np.float32) * scalar_gate
    # Drop-shift downward by dy (positive dy = down in image coords).
    H = base.shape[0]
    shifted = np.zeros_like(base)
    if dy > 0:
        shifted[dy:] = base[: H - dy]
    elif dy < 0:
        shifted[: H + dy] = base[-dy:]
    else:
        shifted = base.copy()
    ksize = 2 * int(np.ceil(3.0 * sigma)) + 1
    soft = cv2.GaussianBlur(shifted, (ksize, ksize), sigmaX=float(sigma), sigmaY=float(sigma))
    return np.clip(soft * darken, 0.0, darken)


# ── Driver ───────────────────────────────────────────────────────────────────

def overlay_robot(
    scene_rgb: np.ndarray,             # (H, W, 3) uint8 — inpainted bg
    robot_rgb: np.ndarray,             # (H, W, 3) uint8 — MuJoCo render
    robot_mask: np.ndarray,            # (H, W) bool
    hand_mask: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    *,
    feather_sigma: float = 1.5,
    harmonize: bool = True,
    harmonize_l: float = 0.4,
    harmonize_ab: float = 0.7,
    ring_dilate: int = 30,
    shadow: bool = True,
    shadow_z_lo: Optional[float] = None,
    shadow_z_hi: Optional[float] = None,
    shadow_dy: int = 8,
    shadow_sigma: float = 6.0,
    shadow_darken: float = 0.35,
    scene_ema: Optional[OverlayStatsEMA] = None,
    exclude_mask: Optional[np.ndarray] = None,
    defringe: bool = True,
    extend_bleed: bool = True,
) -> np.ndarray:
    """One-shot egoview overlay: cleans robot edge, harmonizes color, draws
    soft drop-shadow under the hands, alpha-blends. Returns (H, W, 3) uint8.

    Pixel ops happen in this order to avoid the "darken-then-overwrite"
    failure mode where a shadow leaks under a feathered robot edge, and to
    eliminate the cold-render-bg halo at the feathered edge:

      robot_ext     = extend_robot_into_bleed_zone(robot, radius)
      robot_h       = harmonize(defringe(robot_ext), extended_mask, scene)
      scene_dark    = scene * (1 - shadow_alpha)
      alpha         = feather(robot_mask, sigma)        # original silhouette
      out           = scene_dark * (1 - alpha) + robot_h * alpha
    """
    # Step 1: grow robot RGB outward into the bleed zone so the feather
    # edge mixes against warm robot color, not the cold MuJoCo skybox.
    if extend_bleed and feather_sigma > 0:
        bleed_radius = int(np.ceil(feather_sigma) + np.ceil(3.0 * feather_sigma))
        robot_processed, extended_mask = extend_robot_into_bleed_zone(
            robot_rgb, robot_mask, bleed_radius,
        )
    else:
        robot_processed = robot_rgb
        extended_mask = robot_mask

    if defringe:
        robot_processed = defringe_silhouette(robot_processed, extended_mask)
    if harmonize:
        robot_processed = harmonize_color(
            robot_processed, extended_mask, scene_rgb,
            l_strength=harmonize_l, ab_strength=harmonize_ab,
            ring_dilate=ring_dilate, exclude_mask=exclude_mask,
            scene_ema=scene_ema,
        )

    # Shadow on scene
    if shadow and hand_mask is not None and depth is not None \
            and shadow_z_lo is not None and shadow_z_hi is not None:
        shadow_alpha = hand_contact_shadow(
            hand_mask, depth, shadow_z_lo, shadow_z_hi,
            dy=shadow_dy, sigma=shadow_sigma, darken=shadow_darken,
        )
        scene_dark = (scene_rgb.astype(np.float32)
                      * (1.0 - shadow_alpha[..., None])).astype(np.uint8)
    else:
        scene_dark = scene_rgb

    # Feathered alpha blend
    alpha = feather_alpha(robot_mask, feather_sigma)[..., None]
    out = scene_dark.astype(np.float32) * (1.0 - alpha) \
        + robot_processed.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)
