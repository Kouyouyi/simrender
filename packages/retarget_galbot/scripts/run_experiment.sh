#!/usr/bin/env bash
# Run Galbot retarget (+ optional Rerun) from an experiment YAML.
#
# Usage:
#   ./scripts/run_experiment.sh configs/experiments/example.yml
#   ./scripts/run_experiment.sh configs/experiments/example.yml --max_frames 50 --overwrite
#   ./scripts/run_experiment.sh configs/experiments/example.yml --no_retarget --do_visualize
#
# YAML holds defaults; any CLI flag below overrides the matching field.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  ./scripts/run_experiment.sh <experiment.yml> [overrides...]

Examples:
  ./scripts/run_experiment.sh configs/experiments/example.yml
  ./scripts/run_experiment.sh configs/experiments/example.yml \
      --episode_dir /data/aoe/raw_xxx_seg_yyy \
      --max_frames 80 \
      --overwrite \
      --open_rerun

Common overrides:
  --episode_dir PATH       --data_root PATH
  --max_frames N           --stride N
  --output_dir PATH        --lerobot_root PATH
  --repo_id ID             --rrd PATH
  --overwrite / --no_overwrite
  --write_lerobot / --no_write_lerobot
  --do_retarget / --no_retarget
  --do_visualize / --no_visualize
  --open_rerun / --no_open_rerun
  --conda_env NAME         --mujoco_gl egl|osmesa|glfw
  --retarget_config PATH   (Galbot IK yaml, e.g. configs/galbot_dex_bimanual.yml)
EOF
  exit 2
fi

CONFIG_YAML="$1"
shift

if [[ ! -f "$CONFIG_YAML" ]]; then
  if [[ -f "$ROOT/$CONFIG_YAML" ]]; then
    CONFIG_YAML="$ROOT/$CONFIG_YAML"
  else
    echo "ERROR: config not found: $CONFIG_YAML" >&2
    exit 1
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

PLAN_JSON="$("$PYTHON_BIN" "$ROOT/scripts/exp_config.py" \
  --config "$CONFIG_YAML" \
  --repo_root "$ROOT" \
  --format json \
  "$@")"

PLAN_ENV_FILE="$(mktemp)"
"$PYTHON_BIN" - "$PLAN_JSON" >"$PLAN_ENV_FILE" <<'PY'
import json
import shlex
import sys

plan = json.loads(sys.argv[1])


def q(value: object) -> str:
    return shlex.quote(str(value))


print(f"DO_RETARGET={int(plan['do_retarget'])}")
print(f"DO_VISUALIZE={int(plan['do_visualize'])}")
print(f"OPEN_RERUN={int(plan['open_rerun'])}")
print(f"CONDA_ENV={q(plan['conda_env'])}")
print(f"export MUJOCO_GL={q(plan['mujoco_gl'])}")
print(f"OUTPUT_DIR={q(plan['output_dir'])}")
print(f"LEROBOT_ROOT={q(plan['lerobot_root'])}")
print(f"RRD={q(plan['rrd'])}")
print("RETARGET_ARGS=(" + " ".join(q(x) for x in plan["retarget_argv"]) + ")")
print("VISUALIZE_ARGS=(" + " ".join(q(x) for x in plan["visualize_argv"]) + ")")
PY
# shellcheck disable=SC1090
source "$PLAN_ENV_FILE"
rm -f "$PLAN_ENV_FILE"

run_py() {
  local script="$1"
  shift
  if [[ -n "${CONDA_ENV}" ]]; then
    conda run -n "$CONDA_ENV" --no-capture-output python "$script" "$@"
  else
    "$PYTHON_BIN" "$script" "$@"
  fi
}

echo "=== experiment ==="
echo "config:        $CONFIG_YAML"
echo "output_dir:    $OUTPUT_DIR"
echo "lerobot_root:  $LEROBOT_ROOT"
echo "rrd:           $RRD"
echo "MUJOCO_GL:     ${MUJOCO_GL}"
echo "conda_env:     ${CONDA_ENV:-<current python>}"
echo "do_retarget:   $DO_RETARGET"
echo "do_visualize:  $DO_VISUALIZE"
echo "open_rerun:    $OPEN_RERUN"
echo "==============="

if [[ "$DO_RETARGET" -eq 1 ]]; then
  echo "[1/2] retarget..."
  mkdir -p "$OUTPUT_DIR"
  run_py "$ROOT/scripts/retarget.py" "${RETARGET_ARGS[@]}"
else
  echo "[1/2] retarget skipped"
fi

if [[ "$DO_VISUALIZE" -eq 1 ]]; then
  echo "[2/2] rerun visualize (write .rrd)..."
  if [[ ! -d "$LEROBOT_ROOT" ]]; then
    echo "ERROR: lerobot_root does not exist: $LEROBOT_ROOT" >&2
    echo "Run with retarget.write_lerobot=true first, or pass --lerobot_root." >&2
    exit 1
  fi
  run_py "$ROOT/scripts/visualize.py" "${VISUALIZE_ARGS[@]}"
else
  echo "[2/2] visualize skipped"
fi

if [[ "$OPEN_RERUN" -eq 1 ]]; then
  if [[ ! -f "$RRD" ]]; then
    echo "ERROR: rrd not found: $RRD" >&2
    exit 1
  fi
  echo "Opening Rerun viewer: $RRD"
  exec rerun "$RRD"
fi

echo "Done."
echo "  actions/lerobot: $OUTPUT_DIR"
echo "  rrd:             $RRD"
echo "Open later with:   rerun \"$RRD\""
