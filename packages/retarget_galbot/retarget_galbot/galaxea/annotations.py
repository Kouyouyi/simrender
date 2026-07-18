from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ActionSegment:
    id: int
    start_frame: int
    end_frame: int
    scene: str
    verbs: tuple[str, ...]
    objects: tuple[str, ...]
    hands: tuple[str, ...]

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def load_action_segments(segment_dir: Path) -> list[ActionSegment]:
    path = segment_dir / "ego_annotation" / "ego_action_annotation.json"
    if not path.exists():
        return []

    raw_items = json.loads(path.read_text())
    segments: list[ActionSegment] = []
    for item in raw_items:
        actions = item.get("atomic_action", []) or []
        verbs = tuple(str(action.get("verb", "")).lower() for action in actions)
        objects = tuple(str(action.get("object", "")).lower() for action in actions)
        hands = tuple(str(action.get("hand", "")).lower() for action in actions)
        segments.append(
            ActionSegment(
                id=_as_int(item.get("id"), len(segments) + 1),
                start_frame=_as_int(item.get("start_frame")),
                end_frame=_as_int(item.get("end_frame")),
                scene=str(item.get("scene", "")),
                verbs=verbs,
                objects=objects,
                hands=hands,
            )
        )
    return segments


def segment_for_frame(
    segments: Iterable[ActionSegment], frame: int
) -> ActionSegment | None:
    for segment in segments:
        if segment.contains(frame):
            return segment
    return None
