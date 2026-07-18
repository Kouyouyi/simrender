from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JointInfo:
    name: str
    joint_type: str
    lower: float
    upper: float
    mimic_joint: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0


def _joint_limit(joint: ET.Element) -> tuple[float, float]:
    limit = joint.find("limit")
    if limit is None:
        return -3.141592653589793, 3.141592653589793
    lower = float(limit.get("lower", "-3.141592653589793"))
    upper = float(limit.get("upper", "3.141592653589793"))
    return lower, upper


def parse_movable_joints(urdf_path: str | Path) -> list[JointInfo]:
    root = ET.parse(urdf_path).getroot()
    joints: list[JointInfo] = []
    for joint in root.findall("joint"):
        joint_type = str(joint.get("type", "fixed"))
        if joint_type == "fixed":
            continue
        lower, upper = _joint_limit(joint)
        mimic = joint.find("mimic")
        mimic_joint = mimic.get("joint") if mimic is not None else None
        mimic_multiplier = float(mimic.get("multiplier", "1.0")) if mimic is not None else 1.0
        mimic_offset = float(mimic.get("offset", "0.0")) if mimic is not None else 0.0
        joints.append(
            JointInfo(
                name=str(joint.get("name")),
                joint_type=joint_type,
                lower=lower,
                upper=upper,
                mimic_joint=mimic_joint,
                mimic_multiplier=mimic_multiplier,
                mimic_offset=mimic_offset,
            )
        )
    return joints


def joint_limit_map(joints: list[JointInfo]) -> dict[str, tuple[float, float]]:
    return {joint.name: (joint.lower, joint.upper) for joint in joints}


def _flat_convex_hull_stem(mesh_uri: str) -> str:
    filename = mesh_uri.removeprefix("package://").split("/")[-1]
    base = Path(filename)
    if base.suffix == ".obj":
        return base.stem
    return base.stem.removesuffix("_convex_hull")


def resolve_sapien_mesh_path(mesh_uri: str, urdf_dir: Path) -> Path:
    stl_path = urdf_dir / f"{_flat_convex_hull_stem(mesh_uri)}_convex_hull.stl"
    if not stl_path.is_file():
        raise FileNotFoundError(
            f"SAPIEN mesh not found for {mesh_uri!r}. Expected flat STL at {stl_path}"
        )
    return stl_path


def prepare_sapien_urdf(
    urdf_path: str | Path,
    cache_dir: str | Path | None = None,
    include_collisions: bool = False,
) -> Path:
    """Prepare a Galbot URDF variant for SAPIEN visualization.

    Galbot's original URDF uses package mesh URIs and STL collision meshes.
    SAPIEN can warn loudly on the collision STLs, while validation playback only
    needs visuals. By default this writes a visual-only cached URDF.
    """
    urdf_path = Path(urdf_path).expanduser().resolve()
    urdf_dir = urdf_path.parent
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "sapien_urdf"
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    collision_tag = "collision" if include_collisions else "visual_only"
    output_path = cache_path / (
        f"{urdf_path.stem}_{collision_tag}_{int(urdf_path.stat().st_mtime_ns)}.urdf"
    )
    if output_path.is_file():
        return output_path

    root = ET.parse(urdf_path).getroot()
    if not include_collisions:
        for link in root.findall("link"):
            for collision in list(link.findall("collision")):
                link.remove(collision)
    for mesh in root.iter("mesh"):
        mesh_uri = mesh.get("filename")
        if not mesh_uri:
            continue
        mesh.set("filename", str(resolve_sapien_mesh_path(mesh_uri, urdf_dir)))
    ET.ElementTree(root).write(output_path, encoding="unicode", xml_declaration=True)
    return output_path
