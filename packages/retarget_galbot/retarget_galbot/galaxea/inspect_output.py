from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a retarget output pickle.")
    parser.add_argument("path")
    args = parser.parse_args()

    path = Path(args.path).expanduser().resolve()
    with path.open("rb") as file:
        payload = pickle.load(file)

    meta = payload.get("meta_data", {})
    data = payload.get("data", [])
    print(f"path: {path}")
    print(f"backend: {meta.get('backend')}")
    print(f"dof: {meta.get('dof')}")
    print(f"frames: {len(data)}")
    print(f"joint_names: {meta.get('joint_names', [])[:8]}...")
    if data:
        first = data[0]
        qpos = first["qpos"] if isinstance(first, dict) else first
        print(f"first_frame_qpos_head: {qpos[:8]}")


if __name__ == "__main__":
    main()
