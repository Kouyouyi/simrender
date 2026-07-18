"""Build a kinematic MuJoCo G1 model from URDF joints and optional meshes."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np


def _numbers(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.fromstring(text, sep=" ", dtype=np.float64)


def _format(values: np.ndarray | tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _quat_wxyz_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def _add_geom(body: ET.Element, **attributes: str) -> None:
    attributes.setdefault("contype", "0")
    attributes.setdefault("conaffinity", "0")
    attributes.setdefault("group", "1")
    ET.SubElement(body, "geom", attributes)


def _link_color(link_name: str) -> str:
    if "_l_" in link_name or link_name.startswith("gripper_l"):
        return "left"
    if "_r_" in link_name or link_name.startswith("gripper_r"):
        return "right"
    if "head" in link_name:
        return "head"
    return "body"


def _visual_material(link_name: str) -> str:
    """Return the neutral material used by the official G1_120s meshes."""
    if link_name == "body_link1":
        return "visual_dark"
    if link_name == "base_link":
        return "visual_mid"
    if link_name.startswith("gripper_"):
        return "gripper"
    return "visual_white"


def _link_radius(link_name: str) -> float:
    if "gripper" in link_name:
        return 0.009
    if "arm_" in link_name:
        return 0.035
    return 0.028


def _add_link_proxy(
    body: ET.Element,
    link_name: str,
    child_origins: list[np.ndarray],
) -> None:
    material = _link_color(link_name)
    radius = _link_radius(link_name)

    if link_name == "base_link":
        _add_geom(body, type="box", pos="0 0 0.31", size="0.30 0.24 0.25", material="base")
        _add_geom(body, type="capsule", fromto="0 0 0.50 0 0 0.66", size="0.075", material="metal")
        for y in (-0.25, 0.25):
            _add_geom(
                body,
                type="cylinder",
                pos=_format((-0.08, y, 0.11)),
                euler="1.570796327 0 0",
                size="0.105 0.055",
                material="rubber",
            )
    elif link_name == "body_link1":
        _add_geom(body, type="capsule", fromto="0 0 -0.42 0 0 0", size="0.065", material="metal")
        _add_geom(body, type="box", pos="0.06 0 0", size="0.10 0.11 0.08", material="body")
    elif link_name == "body_link2":
        _add_geom(body, type="box", pos="0 -0.15 0", size="0.15 0.20 0.11", material="body")
    elif link_name == "head_link2":
        _add_geom(body, type="sphere", pos="0.02 0.025 0", size="0.105", material="head")
        _add_geom(body, type="box", pos="0.095 0.025 0", size="0.025 0.065 0.045", material="camera")
    elif link_name.endswith("base_link") and "gripper" in link_name:
        _add_geom(body, type="cylinder", pos="0 0 0.045", size="0.055 0.07", material="gripper")
    elif link_name not in {"arm_base_link"}:
        _add_geom(body, type="sphere", size=_format((radius,)), material=material)

    for index, origin in enumerate(child_origins):
        length = float(np.linalg.norm(origin))
        if length < 0.025 or length > 0.70:
            continue
        _add_geom(
            body,
            name=f"{link_name}_branch_{index}",
            type="capsule",
            fromto=_format(np.concatenate((np.zeros(3), origin))),
            size=_format((radius,)),
            material=material,
        )


def build_proxy_mjcf(
    urdf_path: str | Path,
    visual_mesh_dir: str | Path | None = None,
) -> str:
    """Convert URDF joints to MJCF, using link-local OBJ meshes when supplied.

    ``visual_mesh_dir`` must contain one ``<link_name>.obj`` for every URDF
    link that declares a visual. Links without visuals (virtual mount/center
    links) intentionally remain geometry-free.
    """
    urdf_path = Path(urdf_path).expanduser().resolve()
    urdf = ET.parse(urdf_path).getroot()
    link_names = {element.attrib["name"] for element in urdf.findall("link")}
    visual_link_names = {
        element.attrib["name"]
        for element in urdf.findall("link")
        if element.find("visual") is not None
    }
    mesh_files: dict[str, Path] = {}
    if visual_mesh_dir is not None:
        mesh_dir = Path(visual_mesh_dir).expanduser().resolve()
        if not mesh_dir.is_dir():
            raise FileNotFoundError(f"Visual mesh directory does not exist: {mesh_dir}")
        mesh_files = {name: mesh_dir / f"{name}.obj" for name in visual_link_names}
        missing = sorted(name for name, path in mesh_files.items() if not path.is_file())
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} link-local OBJ meshes in {mesh_dir}: {missing}"
            )
    joints: list[dict[str, object]] = []
    children_by_parent: dict[str, list[dict[str, object]]] = defaultdict(list)
    child_links: set[str] = set()

    for element in urdf.findall("joint"):
        parent = element.find("parent")
        child = element.find("child")
        if parent is None or child is None:
            raise ValueError(f"Malformed joint {element.attrib.get('name')}")
        origin_element = element.find("origin")
        origin_xyz = _numbers(
            None if origin_element is None else origin_element.attrib.get("xyz"),
            (0.0, 0.0, 0.0),
        )
        origin_rpy = _numbers(
            None if origin_element is None else origin_element.attrib.get("rpy"),
            (0.0, 0.0, 0.0),
        )
        axis_element = element.find("axis")
        axis = _numbers(
            None if axis_element is None else axis_element.attrib.get("xyz"),
            (1.0, 0.0, 0.0),
        )
        limit_element = element.find("limit")
        limit = None
        if limit_element is not None and "lower" in limit_element.attrib:
            limit = (
                float(limit_element.attrib["lower"]),
                float(limit_element.attrib["upper"]),
            )
        record: dict[str, object] = {
            "name": element.attrib["name"],
            "type": element.attrib["type"],
            "parent": parent.attrib["link"],
            "child": child.attrib["link"],
            "xyz": origin_xyz,
            "rpy": origin_rpy,
            "axis": axis,
            "limit": limit,
        }
        joints.append(record)
        children_by_parent[str(record["parent"])].append(record)
        child_links.add(str(record["child"]))

    roots = link_names - child_links
    if len(roots) != 1:
        raise ValueError(f"Expected one URDF root link, found {sorted(roots)}")
    root_link = next(iter(roots))

    model_name = "agibot_g1_visual_mesh" if mesh_files else "agibot_g1_proxy"
    mjcf = ET.Element("mujoco", {"model": model_name})
    ET.SubElement(mjcf, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(mjcf, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})
    ET.SubElement(mjcf, "statistic", {"center": "0.6 0 0.9", "extent": "2.2"})
    visual = ET.SubElement(mjcf, "visual")
    ET.SubElement(visual, "headlight", {"ambient": "0.45 0.45 0.45", "diffuse": "0.7 0.7 0.7", "specular": "0.2 0.2 0.2"})
    ET.SubElement(visual, "rgba", {"haze": "0.82 0.88 0.92 1"})
    ET.SubElement(visual, "global", {"azimuth": "135", "elevation": "-20"})

    asset = ET.SubElement(mjcf, "asset")
    materials = {
        "base": "0.12 0.15 0.18 1",
        "rubber": "0.025 0.03 0.035 1",
        "metal": "0.70 0.74 0.78 1",
        "body": "0.86 0.88 0.90 1",
        "head": "0.94 0.95 0.96 1",
        "camera": "0.04 0.07 0.08 1",
        "left": "0.08 0.47 0.73 1",
        "right": "0.91 0.38 0.12 1",
        "gripper": "0.10 0.12 0.13 1",
        "visual_white": "0.92 0.92 0.94 1",
        "visual_mid": "0.82 0.82 0.84 1",
        "visual_dark": "0.12 0.13 0.15 1",
    }
    for name, rgba in materials.items():
        ET.SubElement(asset, "material", {"name": name, "rgba": rgba, "specular": "0.25", "shininess": "0.3"})
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "floor_grid",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.30 0.34 0.36",
            "rgb2": "0.20 0.23 0.25",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "floor",
            "texture": "floor_grid",
            "texrepeat": "12 12",
            "reflectance": "0.02" if mesh_files else "0.1",
        },
    )
    for link_name, mesh_path in sorted(mesh_files.items()):
        ET.SubElement(
            asset,
            "mesh",
            {"name": f"{link_name}_visual_mesh", "file": str(mesh_path)},
        )

    world = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(world, "light", {"pos": "-1 -3 5", "dir": "0.3 0.4 -1", "diffuse": "0.9 0.9 0.9"})
    ET.SubElement(world, "light", {"pos": "4 2 3", "dir": "-0.8 -0.2 -0.6", "diffuse": "0.5 0.5 0.5"})
    floor_attributes = {
        "name": "floor",
        "type": "plane",
        "size": "5 5 0.1",
        "material": "floor",
    }
    if mesh_files:
        floor_attributes["pos"] = "0 0 -0.23"
    ET.SubElement(world, "geom", floor_attributes)

    def add_link(parent_xml: ET.Element, link_name: str, incoming: dict[str, object] | None) -> None:
        attributes = {"name": link_name}
        if incoming is not None:
            attributes["pos"] = _format(incoming["xyz"])
            attributes["quat"] = _format(_quat_wxyz_from_rpy(incoming["rpy"]))
        body = ET.SubElement(parent_xml, "body", attributes)
        if incoming is None:
            ET.SubElement(body, "freejoint", {"name": "base_free_joint"})
        elif incoming["type"] != "fixed":
            joint_attributes = {
                "name": str(incoming["name"]),
                "type": "slide" if incoming["type"] == "prismatic" else "hinge",
                "axis": _format(incoming["axis"]),
                "damping": "0.2",
                "armature": "0.001",
            }
            if incoming["limit"] is not None:
                joint_attributes["range"] = _format(incoming["limit"])
                joint_attributes["limited"] = "true"
            ET.SubElement(body, "joint", joint_attributes)

        child_joints = children_by_parent.get(link_name, [])
        if link_name in mesh_files:
            _add_geom(
                body,
                type="mesh",
                mesh=f"{link_name}_visual_mesh",
                material=_visual_material(link_name),
            )
        elif not mesh_files:
            _add_link_proxy(body, link_name, [joint["xyz"] for joint in child_joints])
        ET.SubElement(body, "site", {"name": f"{link_name}_site", "size": "0.006", "rgba": "1 1 1 0.35"})
        for joint in child_joints:
            add_link(body, str(joint["child"]), joint)

    add_link(world, root_link, None)
    return ET.tostring(mjcf, encoding="unicode")


def build_proxy_model(
    urdf_path: str | Path,
    visual_mesh_dir: str | Path | None = None,
) -> mujoco.MjModel:
    """Compile the proxy or mesh-backed MJCF into a MuJoCo model."""
    return mujoco.MjModel.from_xml_string(build_proxy_mjcf(urdf_path, visual_mesh_dir))
