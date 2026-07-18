#!/usr/bin/env python3
"""Load an experiment YAML and emit shell-safe argv / env for run_experiment.sh."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install PyYAML") from exc


def _deep_get(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Cannot parse bool from {value!r}")


def _path_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(Path(value).expanduser())


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Experiment config must be a mapping: {path}")
    return data


def apply_cli_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Map flat CLI keys onto nested experiment YAML sections."""
    run = dict(cfg.get("run") or {})
    episode = dict(cfg.get("episode") or {})
    retarget = dict(cfg.get("retarget") or {})
    lerobot = dict(cfg.get("lerobot") or {})
    visualize = dict(cfg.get("visualize") or {})
    env = dict(cfg.get("env") or {})

    mapping = {
        "episode_dir": ("episode", "episode_dir"),
        "data_root": ("episode", "data_root"),
        "max_episodes": ("episode", "max_episodes"),
        "robot": ("retarget", "robot"),
        "config": ("retarget", "config"),
        "output_dir": ("retarget", "output_dir"),
        "max_frames": ("retarget", "max_frames"),
        "stride": ("retarget", "stride"),
        "skip_existing": ("retarget", "skip_existing"),
        "video_start_frame": ("retarget", "video_start_frame"),
        "write_pickle": ("retarget", "write_pickle"),
        "write_jsonl": ("retarget", "write_jsonl"),
        "write_lerobot": ("retarget", "write_lerobot"),
        "overwrite": ("retarget", "overwrite"),
        "task": ("retarget", "task"),
        "lerobot_root": ("lerobot", "lerobot_root"),
        "repo_id": ("lerobot", "repo_id"),
        "episode_index": ("visualize", "episode_index"),
        "rrd": ("visualize", "rrd"),
        "playback_speed": ("visualize", "playback_speed"),
        "viz_max_frames": ("visualize", "max_frames"),
        "do_retarget": ("run", "retarget"),
        "do_visualize": ("run", "visualize"),
        "open_rerun": ("run", "open_rerun"),
        "conda_env": ("env", "conda_env"),
        "mujoco_gl": ("env", "MUJOCO_GL"),
    }
    sections = {
        "run": run,
        "episode": episode,
        "retarget": retarget,
        "lerobot": lerobot,
        "visualize": visualize,
        "env": env,
    }
    for key, value in overrides.items():
        if value is None:
            continue
        if key not in mapping:
            continue
        section_name, field = mapping[key]
        sections[section_name][field] = value

    return {
        "run": run,
        "episode": episode,
        "retarget": retarget,
        "lerobot": lerobot,
        "visualize": visualize,
        "env": env,
    }


def build_plan(cfg: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    run = cfg.get("run") or {}
    episode = cfg.get("episode") or {}
    retarget = cfg.get("retarget") or {}
    lerobot = cfg.get("lerobot") or {}
    visualize = cfg.get("visualize") or {}
    env = cfg.get("env") or {}

    do_retarget = _as_bool(run.get("retarget", True))
    do_visualize = _as_bool(run.get("visualize", True))
    open_rerun = _as_bool(run.get("open_rerun", False))

    episode_dir = _path_str(episode.get("episode_dir"))
    data_root = _path_str(episode.get("data_root"))
    if do_retarget and not episode_dir and not data_root:
        raise ValueError("Set episode.episode_dir or episode.data_root in the YAML (or CLI).")

    output_dir = Path(
        _path_str(retarget.get("output_dir")) or str(repo_root / "output" / "exp")
    )
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()

    lerobot_root = Path(
        _path_str(lerobot.get("lerobot_root")) or str(output_dir / "lerobot")
    )
    if not lerobot_root.is_absolute():
        lerobot_root = (repo_root / lerobot_root).resolve()

    rrd = _path_str(visualize.get("rrd"))
    if rrd is None:
        rrd = str(output_dir / "episode_0.rrd")
    rrd_path = Path(rrd)
    if not rrd_path.is_absolute():
        rrd_path = (repo_root / rrd_path).resolve()

    retarget_config = _path_str(retarget.get("config"))
    if retarget_config and not Path(retarget_config).is_absolute():
        retarget_config = str((repo_root / retarget_config).resolve())

    max_frames = retarget.get("max_frames")
    viz_max_frames = visualize.get("max_frames", max_frames)

    retarget_argv: list[str] = []
    if episode_dir:
        retarget_argv += ["--episode_dir", episode_dir]
    elif data_root:
        retarget_argv += ["--data_root", data_root]
        if episode.get("max_episodes") is not None:
            retarget_argv += ["--max_episodes", str(int(episode["max_episodes"]))]

    retarget_argv += [
        "--robot",
        str(retarget.get("robot", "galbot")),
        "--output_dir",
        str(output_dir),
        "--lerobot_root",
        str(lerobot_root),
        "--repo_id",
        str(lerobot.get("repo_id", "aoe/galbot_retarget")),
        "--stride",
        str(int(retarget.get("stride", 1))),
        "--playback_speed",
        str(float(visualize.get("playback_speed", 1.0))),
    ]
    if retarget_config:
        retarget_argv += ["--config", retarget_config]
    if max_frames is not None:
        retarget_argv += ["--max_frames", str(int(max_frames))]
    if retarget.get("video_start_frame") is not None:
        retarget_argv += ["--video_start_frame", str(int(retarget["video_start_frame"]))]
    if retarget.get("task"):
        retarget_argv += ["--task", str(retarget["task"])]
    if _as_bool(retarget.get("skip_existing", False)):
        retarget_argv.append("--skip_existing")
    if _as_bool(retarget.get("write_pickle", False)):
        retarget_argv.append("--write_pickle")
    if _as_bool(retarget.get("write_jsonl", False)):
        retarget_argv.append("--write_jsonl")
    if _as_bool(retarget.get("write_lerobot", True)):
        retarget_argv.append("--write_lerobot")
    if _as_bool(retarget.get("overwrite", False)):
        retarget_argv.append("--overwrite")
    # Visualization is always a separate step (scripts/visualize.py) so camera
    # / Rerun settings can be iterated without re-running IK + egoview.

    visualize_argv: list[str] = [
        "--lerobot_root",
        str(lerobot_root),
        "--repo_id",
        str(lerobot.get("repo_id", "aoe/galbot_retarget")),
        "--robot",
        str(retarget.get("robot", "galbot")),
        "--episode_index",
        str(int(visualize.get("episode_index", 0))),
        "--playback_speed",
        str(float(visualize.get("playback_speed", 1.0))),
        "--rrd",
        str(rrd_path),
    ]
    if viz_max_frames is not None:
        visualize_argv += ["--max_frames", str(int(viz_max_frames))]

    return {
        "do_retarget": do_retarget,
        "do_visualize": do_visualize,
        "open_rerun": open_rerun,
        "conda_env": str(env.get("conda_env") or ""),
        "mujoco_gl": str(env.get("MUJOCO_GL") or "egl"),
        "output_dir": str(output_dir),
        "lerobot_root": str(lerobot_root),
        "rrd": str(rrd_path),
        "retarget_argv": retarget_argv,
        "visualize_argv": visualize_argv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--format", choices=["json", "shell"], default="json")

    # Flat overrides (same names as run_experiment.sh CLI)
    parser.add_argument("--episode_dir", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--robot", type=str, default=None)
    parser.add_argument("--retarget_config", dest="config_override", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true", default=None)
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    parser.add_argument("--video_start_frame", type=int, default=None)
    parser.add_argument("--write_pickle", action="store_true", default=None)
    parser.add_argument("--no_write_pickle", action="store_false", dest="write_pickle")
    parser.add_argument("--write_jsonl", action="store_true", default=None)
    parser.add_argument("--no_write_jsonl", action="store_false", dest="write_jsonl")
    parser.add_argument("--write_lerobot", action="store_true", default=None)
    parser.add_argument("--no_write_lerobot", action="store_false", dest="write_lerobot")
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--no_overwrite", action="store_false", dest="overwrite")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--lerobot_root", type=str, default=None)
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--episode_index", type=int, default=None)
    parser.add_argument("--rrd", type=str, default=None)
    parser.add_argument("--playback_speed", type=float, default=None)
    parser.add_argument("--viz_max_frames", type=int, default=None)
    parser.add_argument("--do_retarget", action="store_true", default=None)
    parser.add_argument("--no_retarget", action="store_false", dest="do_retarget")
    parser.add_argument("--do_visualize", action="store_true", default=None)
    parser.add_argument("--no_visualize", action="store_false", dest="do_visualize")
    parser.add_argument("--open_rerun", action="store_true", default=None)
    parser.add_argument("--no_open_rerun", action="store_false", dest="open_rerun")
    parser.add_argument("--conda_env", type=str, default=None)
    parser.add_argument("--mujoco_gl", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config.resolve())
    overrides = {
        "episode_dir": args.episode_dir,
        "data_root": args.data_root,
        "max_episodes": args.max_episodes,
        "robot": args.robot,
        "config": args.config_override,
        "output_dir": args.output_dir,
        "max_frames": args.max_frames,
        "stride": args.stride,
        "skip_existing": args.skip_existing,
        "video_start_frame": args.video_start_frame,
        "write_pickle": args.write_pickle,
        "write_jsonl": args.write_jsonl,
        "write_lerobot": args.write_lerobot,
        "overwrite": args.overwrite,
        "task": args.task,
        "lerobot_root": args.lerobot_root,
        "repo_id": args.repo_id,
        "episode_index": args.episode_index,
        "rrd": args.rrd,
        "playback_speed": args.playback_speed,
        "viz_max_frames": args.viz_max_frames,
        "do_retarget": args.do_retarget,
        "do_visualize": args.do_visualize,
        "open_rerun": args.open_rerun,
        "conda_env": args.conda_env,
        "mujoco_gl": args.mujoco_gl,
    }
    merged = apply_cli_overrides(cfg, overrides)
    plan = build_plan(merged, args.repo_root.resolve())

    if args.format == "json":
        json.dump(plan, sys.stdout)
        sys.stdout.write("\n")
        return

    # shell: exportable variables
    def q(value: str) -> str:
        return shlex.quote(value)

    print(f"DO_RETARGET={int(plan['do_retarget'])}")
    print(f"DO_VISUALIZE={int(plan['do_visualize'])}")
    print(f"OPEN_RERUN={int(plan['open_rerun'])}")
    print(f"CONDA_ENV={q(plan['conda_env'])}")
    print(f"MUJOCO_GL={q(plan['mujoco_gl'])}")
    print(f"OUTPUT_DIR={q(plan['output_dir'])}")
    print(f"LEROBOT_ROOT={q(plan['lerobot_root'])}")
    print(f"RRD={q(plan['rrd'])}")
    print(f"RETARGET_ARGV={q(' '.join(plan['retarget_argv']))}")
    print(f"VISUALIZE_ARGV={q(' '.join(plan['visualize_argv']))}")


if __name__ == "__main__":
    main()
