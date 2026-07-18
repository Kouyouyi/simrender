"""Metric depth encoding used by Vision Banana (Gabeur et al., 2026)."""

from __future__ import annotations

import cv2
import numpy as np


# First-order 3D Hilbert path along RGB cube edges.
VISION_BANANA_RGB_VERTICES = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)
VISION_BANANA_PAPER_SCALE_C = 10.0 / 3.0
MUJOCO_NEAR_FIELD_SCALE_C = 0.5
MUJOCO_FOCUS_NEAR_M = 0.12
MUJOCO_FOCUS_FAR_M = 0.80


def vision_banana_curve(
    depth_m: np.ndarray | float,
    *,
    scale_c: float = VISION_BANANA_PAPER_SCALE_C,
) -> np.ndarray:
    """Map metric depth in [0, inf) to the paper's normalized curved distance."""
    if scale_c <= 0.0:
        raise ValueError("scale_c must be positive")
    depth = np.maximum(np.asarray(depth_m, dtype=np.float32), 0.0)
    return 1.0 - np.power(1.0 + depth / (3.0 * scale_c), -2.0)


def vision_banana_inverse_curve(
    curved: np.ndarray | float,
    *,
    scale_c: float = VISION_BANANA_PAPER_SCALE_C,
) -> np.ndarray:
    """Invert the depth curve, returning metric depth in meters."""
    if scale_c <= 0.0:
        raise ValueError("scale_c must be positive")
    value = np.asarray(curved, dtype=np.float64)
    clipped = np.clip(value, 0.0, np.nextafter(1.0, 0.0))
    return 3.0 * scale_c * (np.power(1.0 - clipped, -0.5) - 1.0)


def vision_banana_curve_to_rgb(curved: np.ndarray) -> np.ndarray:
    """Interpolate normalized distances along the RGB-cube Hilbert path."""
    value = np.clip(np.asarray(curved, dtype=np.float32), 0.0, 1.0)
    path = value * (len(VISION_BANANA_RGB_VERTICES) - 1)
    segment = np.minimum(path.astype(np.int32), len(VISION_BANANA_RGB_VERTICES) - 2)
    fraction = np.clip(path - segment, 0.0, 1.0)[..., None]
    low = VISION_BANANA_RGB_VERTICES[segment]
    high = VISION_BANANA_RGB_VERTICES[segment + 1]
    return low * (1.0 - fraction) + high * fraction


def vision_banana_depth_to_rgb(
    depth_m: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    scale_c: float = VISION_BANANA_PAPER_SCALE_C,
) -> np.ndarray:
    """Encode metric depth as the paper's black-red-...-white RGB colors."""
    depth = np.asarray(depth_m, dtype=np.float32)
    rgb = vision_banana_curve_to_rgb(vision_banana_curve(depth, scale_c=scale_c))
    if valid_mask is not None:
        rgb = np.where(np.asarray(valid_mask, dtype=bool)[..., None], rgb, 0.0)
    return np.round(rgb * 255.0).astype(np.uint8)


def vision_banana_rgb_to_curve(rgb: np.ndarray) -> np.ndarray:
    """Project RGB onto the nearest RGB-cube path and return path position."""
    color = np.asarray(rgb, dtype=np.float32) / 255.0
    distances: list[np.ndarray] = []
    fractions: list[np.ndarray] = []
    for segment in range(len(VISION_BANANA_RGB_VERTICES) - 1):
        start = VISION_BANANA_RGB_VERTICES[segment]
        direction = VISION_BANANA_RGB_VERTICES[segment + 1] - start
        denominator = float(direction @ direction)
        fraction = np.clip(
            np.sum((color - start) * direction, axis=-1) / denominator,
            0.0,
            1.0,
        )
        projected = start + fraction[..., None] * direction
        distances.append(np.sum((color - projected) ** 2, axis=-1))
        fractions.append(fraction)
    distance_stack = np.stack(distances, axis=-1)
    fraction_stack = np.stack(fractions, axis=-1)
    nearest_segment = np.argmin(distance_stack, axis=-1)
    nearest_fraction = np.take_along_axis(
        fraction_stack,
        nearest_segment[..., None],
        axis=-1,
    )[..., 0]
    return (nearest_segment + nearest_fraction) / (
        len(VISION_BANANA_RGB_VERTICES) - 1
    )


def vision_banana_rgb_to_depth(
    rgb: np.ndarray,
    *,
    scale_c: float = VISION_BANANA_PAPER_SCALE_C,
) -> np.ndarray:
    """Decode RGB by projection onto the nearest RGB-cube path segment."""
    curved = vision_banana_rgb_to_curve(rgb)
    return vision_banana_inverse_curve(curved, scale_c=scale_c)


def focused_depth_to_curve(
    depth_m: np.ndarray | float,
    *,
    focus_near_m: float = MUJOCO_FOCUS_NEAR_M,
    focus_far_m: float = MUJOCO_FOCUS_FAR_M,
    tail_scale_c: float = MUJOCO_NEAR_FIELD_SCALE_C,
) -> np.ndarray:
    """Allocate five of seven color segments to the measured arm depth range."""
    if not 0.0 < focus_near_m < focus_far_m:
        raise ValueError("Expected 0 < focus_near_m < focus_far_m")
    depth = np.maximum(np.asarray(depth_m, dtype=np.float32), 0.0)
    curved = np.empty_like(depth, dtype=np.float32)
    lower = depth < focus_near_m
    middle = (depth >= focus_near_m) & (depth <= focus_far_m)
    upper = depth > focus_far_m
    curved[lower] = depth[lower] / focus_near_m / 7.0
    curved[middle] = 1.0 / 7.0 + (
        (depth[middle] - focus_near_m)
        / (focus_far_m - focus_near_m)
        * (5.0 / 7.0)
    )
    tail = vision_banana_curve(
        depth[upper] - focus_far_m,
        scale_c=tail_scale_c,
    )
    curved[upper] = 6.0 / 7.0 + tail / 7.0
    return curved


def focused_curve_to_depth(
    curved: np.ndarray | float,
    *,
    focus_near_m: float = MUJOCO_FOCUS_NEAR_M,
    focus_far_m: float = MUJOCO_FOCUS_FAR_M,
    tail_scale_c: float = MUJOCO_NEAR_FIELD_SCALE_C,
) -> np.ndarray:
    """Invert :func:`focused_depth_to_curve`."""
    if not 0.0 < focus_near_m < focus_far_m:
        raise ValueError("Expected 0 < focus_near_m < focus_far_m")
    value = np.clip(
        np.asarray(curved, dtype=np.float64),
        0.0,
        np.nextafter(1.0, 0.0),
    )
    depth = np.empty_like(value, dtype=np.float64)
    lower = value < 1.0 / 7.0
    middle = (value >= 1.0 / 7.0) & (value <= 6.0 / 7.0)
    upper = value > 6.0 / 7.0
    depth[lower] = value[lower] * 7.0 * focus_near_m
    depth[middle] = focus_near_m + (
        (value[middle] - 1.0 / 7.0)
        * (7.0 / 5.0)
        * (focus_far_m - focus_near_m)
    )
    tail_curved = (value[upper] - 6.0 / 7.0) * 7.0
    depth[upper] = focus_far_m + vision_banana_inverse_curve(
        tail_curved,
        scale_c=tail_scale_c,
    )
    return depth


def focused_depth_to_rgb(
    depth_m: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Encode depth using the fixed focused MuJoCo arm ruler."""
    rgb = vision_banana_curve_to_rgb(focused_depth_to_curve(depth_m))
    if valid_mask is not None:
        rgb = np.where(np.asarray(valid_mask, dtype=bool)[..., None], rgb, 0.0)
    return np.round(rgb * 255.0).astype(np.uint8)


def focused_rgb_to_depth(rgb: np.ndarray) -> np.ndarray:
    """Decode RGB produced by :func:`focused_depth_to_rgb`."""
    return focused_curve_to_depth(vision_banana_rgb_to_curve(rgb))


def vision_banana_vertex_depths(
    *,
    scale_c: float = VISION_BANANA_PAPER_SCALE_C,
) -> np.ndarray:
    """Metric depths at the eight color-cube vertices; the final value is inf."""
    curved = np.linspace(0.0, 1.0, len(VISION_BANANA_RGB_VERTICES))
    depths = vision_banana_inverse_curve(curved, scale_c=scale_c)
    depths[-1] = np.inf
    return depths


def draw_vision_banana_ruler(
    image_rgb: np.ndarray,
) -> np.ndarray:
    """Draw a compact, fixed metric-depth color ruler on an RGB image."""
    output = np.ascontiguousarray(image_rgb.copy())
    height, width = output.shape[:2]
    bar_width = min(392, width - 48)
    bar_height = 14
    left = (width - bar_width) // 2
    top = 28
    panel_top = 5
    panel_bottom = top + bar_height + 25

    overlay = output.copy()
    cv2.rectangle(
        overlay,
        (left - 18, panel_top),
        (left + bar_width + 18, panel_bottom),
        (8, 11, 14),
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.78, output, 0.22, 0.0, output)

    curved = np.linspace(0.0, 1.0, bar_width, dtype=np.float32)
    gradient = np.round(vision_banana_curve_to_rgb(curved) * 255.0).astype(np.uint8)
    output[top : top + bar_height, left : left + bar_width] = gradient[None, :, :]
    cv2.rectangle(
        output,
        (left - 1, top - 1),
        (left + bar_width, top + bar_height),
        (235, 238, 240),
        thickness=1,
    )
    cv2.putText(
        output,
        "Focused MuJoCo arm depth (m)",
        (left, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (245, 247, 250),
        1,
        cv2.LINE_AA,
    )

    vertex_depths = focused_curve_to_depth(
        np.linspace(0.0, 1.0, len(VISION_BANANA_RGB_VERTICES))
    )
    vertex_depths[-1] = np.inf
    labels = tuple(
        "inf"
        if not np.isfinite(depth)
        else ("0" if depth < 0.005 else f"{depth:.2f}")
        for depth in vertex_depths
    )
    for index, label in enumerate(labels):
        x = left + int(round(index * (bar_width - 1) / 7.0))
        cv2.line(
            output,
            (x, top + bar_height),
            (x, top + bar_height + 4),
            (245, 247, 250),
            1,
        )
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.31, 1)[0]
        text_x = int(np.clip(x - text_size[0] // 2, 2, width - text_size[0] - 2))
        cv2.putText(
            output,
            label,
            (text_x, top + bar_height + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.31,
            (245, 247, 250),
            1,
            cv2.LINE_AA,
        )
    return output
