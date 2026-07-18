#!/usr/bin/env bash
# Batch retarget 6 AoE POC segments (1200 frames each) into output/exp/<name>/.
#
# For each episode this writes:
#   output/exp/<name>/
#     <name>_actions.npy
#     lerobot/                 # LeRobot v2.1 dataset
#     episode_0.rrd            # Rerun: egoview + mujoco/front + mujoco/top
#     videos/
#       triple_view.mp4      # egoview | mujoco/front | mujoco/top side-by-side
#     run.log
#
# Usage:
#   ./scripts/run_batch_poc.sh
#   CONDA_ENV=retarget_galbot ./scripts/run_batch_poc.sh
#   DATA_ROOT=/path/to/poc_deliver MAX_FRAMES=1200 ./scripts/run_batch_poc.sh
#   ./scripts/run_batch_poc.sh --only poc_raw_video_20260202_214320_part000
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-}"
if [[ -z "$DATA_ROOT" ]]; then
  echo "ERROR: set DATA_ROOT to the directory containing poc_raw_video_* episodes" >&2
  exit 1
fi
OUT_ROOT="${OUT_ROOT:-$ROOT/output/exp}"
MAX_FRAMES="${MAX_FRAMES:-1200}"
CONDA_ENV="${CONDA_ENV:-}"
BASE_YAML="${BASE_YAML:-$ROOT/configs/experiments/batch_poc1200.yml}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_GL

EPISODES=(
  poc_raw_video_20260202_214320_part000
  poc_raw_video_20260505_210922_part000
  poc_raw_video_20260506_220216_part002
  poc_raw_video_20260507_211235_part000
  poc_raw_video_20260523_193213_part000
  poc_raw_video_20260524_210328_part000
)

ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      ONLY="$2"
      shift 2
      ;;
    --data_root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --out_root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --max_frames)
      MAX_FRAMES="$2"
      shift 2
      ;;
    --conda_env)
      CONDA_ENV="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,25p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "$ONLY" ]]; then
  EPISODES=("$ONLY")
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

run_py() {
  local script="$1"
  shift
  if [[ -n "${CONDA_ENV}" ]]; then
    conda run -n "$CONDA_ENV" --no-capture-output python "$script" "$@"
  else
    "$PYTHON_BIN" "$script" "$@"
  fi
}

mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/batch_summary.txt"
echo "batch start $(date -Is)" | tee "$SUMMARY"
echo "DATA_ROOT=$DATA_ROOT MAX_FRAMES=$MAX_FRAMES OUT_ROOT=$OUT_ROOT CONDA_ENV=${CONDA_ENV:-<none>}" | tee -a "$SUMMARY"

FAILED=()
for name in "${EPISODES[@]}"; do
  ep_dir="$DATA_ROOT/$name"
  exp_dir="$OUT_ROOT/$name"
  lerobot_root="$exp_dir/lerobot"
  rrd="$exp_dir/episode_0.rrd"
  videos_dir="$exp_dir/videos"
  log="$exp_dir/run.log"

  echo "" | tee -a "$SUMMARY"
  echo "======== $name ========" | tee -a "$SUMMARY"

  if [[ ! -d "$ep_dir" ]]; then
    echo "MISSING episode_dir: $ep_dir" | tee -a "$SUMMARY"
    FAILED+=("$name")
    continue
  fi

  mkdir -p "$exp_dir" "$videos_dir"
  {
    echo "=== $name ==="
    echo "episode_dir=$ep_dir"
    echo "exp_dir=$exp_dir"
    echo "max_frames=$MAX_FRAMES"
    date -Is
  } >"$log"

  set +e
  EXTRA_CONDA=()
  if [[ -n "$CONDA_ENV" ]]; then
    EXTRA_CONDA=(--conda_env "$CONDA_ENV")
  fi

  ./scripts/run_experiment.sh "$BASE_YAML" \
    --episode_dir "$ep_dir" \
    --output_dir "$exp_dir" \
    --lerobot_root "$lerobot_root" \
    --rrd "$rrd" \
    --max_frames "$MAX_FRAMES" \
    --repo_id "aoe/galbot_${name}" \
    --overwrite \
    --write_lerobot \
    --do_retarget \
    --do_visualize \
    --no_open_rerun \
    --mujoco_gl "$MUJOCO_GL" \
    "${EXTRA_CONDA[@]}" \
    2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    echo "FAILED retarget/visualize rc=$rc" | tee -a "$SUMMARY" "$log"
    FAILED+=("$name")
    set -e
    continue
  fi

  run_py "$ROOT/scripts/export_mp4.py" \
    --lerobot_root "$lerobot_root" \
    --output_dir "$videos_dir" \
    --episode_index 0 \
    --max_frames "$MAX_FRAMES" \
    2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "FAILED export_mp4 rc=$rc" | tee -a "$SUMMARY" "$log"
    FAILED+=("$name")
    continue
  fi

  # Sanity checklist
  ok=1
  for f in \
    "$exp_dir/${name}_actions.npy" \
    "$lerobot_root/meta/info.json" \
    "$rrd" \
    "$videos_dir/triple_view.mp4"
  do
    if [[ ! -e "$f" ]]; then
      echo "MISSING artifact: $f" | tee -a "$SUMMARY" "$log"
      ok=0
    fi
  done
  if [[ $ok -eq 1 ]]; then
    echo "OK $name -> $exp_dir" | tee -a "$SUMMARY"
  else
    FAILED+=("$name")
  fi
done

echo "" | tee -a "$SUMMARY"
echo "batch end $(date -Is)" | tee -a "$SUMMARY"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAILED episodes: ${FAILED[*]}" | tee -a "$SUMMARY"
  exit 1
fi
echo "All episodes succeeded under $OUT_ROOT" | tee -a "$SUMMARY"
