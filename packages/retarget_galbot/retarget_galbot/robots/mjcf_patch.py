# SPDX-FileCopyrightText: Copyright (c) 2026 Open-AoE Contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""MJCF camera-injection for Galbot MuJoCo visualization.

Adapted from Open-AoE Phantom ``robots/mjcf_patch`` (temp-dir asset symlink +
camera inject). Camera names are Galbot-specific
(``galbot_ego`` / ``galbot_external``).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

EGO_CAMERA_NAME = "galbot_ego"
EXT_CAMERA_NAME = "galbot_external"

_EXT_CAMERA_XML = (
    f'<camera name="{EXT_CAMERA_NAME}" mode="fixed" '
    f'pos="2.2 -1.8 1.4" xyaxes="0.7071 0.7071 0 -0.3 0.3 0.9" fovy="50"/>'
)


def _ego_camera_xml(fovy: float) -> str:
    return (
        f'<camera name="{EGO_CAMERA_NAME}" mode="fixed" '
        f'pos="0 0 1.7" quat="1 0 0 0" fovy="{fovy}"/>'
    )


def _inject_cameras(xml: str, fovy: float) -> str:
    wb_idx = xml.find("<worldbody>")
    if wb_idx < 0:
        raise RuntimeError("No <worldbody> in MJCF")
    insert_at = xml.find(">", wb_idx) + 1
    inject = "\n    " + _ego_camera_xml(fovy) + "\n    " + _EXT_CAMERA_XML
    return xml[:insert_at] + inject + xml[insert_at:]


def patch_mjcf_local(src_xml_path: Path, fovy: float) -> Path:
    """Copy MJCF into a temp dir with asset symlinks and inject viz cameras."""
    src_xml_path = Path(src_xml_path).resolve()
    xml = src_xml_path.read_text()
    modified = _inject_cameras(xml, fovy)

    tmpdir = Path(tempfile.mkdtemp(prefix="aoe_mjcf_"))
    src_dir = src_xml_path.parent
    for entry in src_dir.iterdir():
        if entry.is_dir():
            os.symlink(entry, tmpdir / entry.name)
        elif entry.suffix != ".xml":
            os.symlink(entry, tmpdir / entry.name)
    dst = tmpdir / src_xml_path.name
    dst.write_text(modified)
    return dst
