from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any


def save_pickle_output(
    output_path: str | Path, meta_data: dict[str, Any], frames: list[Any]
) -> None:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta_data": meta_data,
        "data": [asdict(frame) for frame in frames],
    }
    with path.open("wb") as file:
        pickle.dump(payload, file)


def save_jsonl_output(output_path: str | Path, frames: list[Any]) -> None:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for frame in frames:
            file.write(json.dumps(asdict(frame), ensure_ascii=False) + "\n")
