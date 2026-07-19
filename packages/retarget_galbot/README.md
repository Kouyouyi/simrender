# Retarget Galbot

Human-to-robot retargeting from AoE egocentric MANO reconstructions to
**Galbot / Galaxea**, with MuJoCo egoview synthesis, LeRobot export, and Rerun
visualization.

The bundled Galbot G1 Golf URDF and meshes come from the public Apache-2.0
repository `GalaxyGeneralRobotics/galbot_one_golf_description`. Exact upstream
commit, local compatibility changes, and the copied license are recorded in
`assets/robots/galbot/SOURCE.md`.

The full pipeline (AoE episode → Pinocchio IK → egoview overlay → LeRobot →
Rerun) lives under `retarget_galbot/`.

## Requirements

| Component | Notes |
|---|---|
| Python | ≥ 3.10 (3.11 recommended) |
| GPU | CUDA recommended for SAM2 / ProPainter; CPU possible but slow |
| OS | Linux (MuJoCo EGL headless rendering assumed in examples) |
| Disk | SAM2 + ProPainter weights are large; runtime sources are bundled |

## Environment setup

We recommend a dedicated conda environment named after this package:

```bash
conda create -n retarget_galbot python=3.11 -y
conda activate retarget_galbot
```

All shell examples below assume `retarget_galbot` is active. You may use another
name; keep it consistent when you set `PROPAINTER_PYTHON` / related env vars.

### 1. Install this package

From the `simrender` repository root:

```bash
cd /path/to/simrender
python -m pip install --no-deps -e packages/retarget_galbot
```

This pulls the core dependencies declared in `pyproject.toml`
(`mujoco`, `numpy`, `scipy`, `pandas`, `pyarrow`, `opencv-python`, `imageio`,
`imageio-ffmpeg`, the PyPI Pinocchio distribution `pin`, `torch`, `PyYAML`).

For headless MuJoCo (dataset export / Rerun offline renders):

```bash
export MUJOCO_GL=egl
```

### 2. LeRobot + Rerun (dataset export / visualization)

```bash
pip install av 'datasets==3.6.0' 'huggingface_hub>=0.34.2' jsonlines rerun-sdk
```

Install a LeRobot build that provides `lerobot.datasets.lerobot_dataset`
(official or a vendored checkout), for example:

```bash
pip install -e /path/to/lerobot/src
# or: export PYTHONPATH="/path/to/lerobot/src:${PYTHONPATH}"
# or: export LEROBOT_SRC=/path/to/lerobot/src
```

If `import lerobot` fails, dataset export will raise at runtime.

### 3. SAM2 (egoview hand / arm segmentation)

The pinned, weight-free SAM2 runtime is bundled under `third_party/sam2`. From
the repository root, install it in the **same** environment as this package and
download the checkpoint:

```bash
./scripts/setup_egoview.sh sam2
./scripts/download_egoview_checkpoints.sh
```

The wrapper automatically finds the bundled source. Its default checkpoint is
`checkpoints/sam2/sam2.1_hiera_base_plus.pt`; `SAM2_CHECKPOINT` and
`SAM2_MODEL_CFG` can override the checkpoint and config.

### 4. ProPainter (egoview hand removal; separate process)

Hand inpainting runs in a **subprocess** so ProPainter’s dependency stack does
not collide with MuJoCo / SAM2. Create a second env (name is conventional):

```bash
conda create -n propainter python=3.10 -y
export PROPAINTER_PYTHON="$(conda run -n propainter which python)"
PROPAINTER_PYTHON="$PROPAINTER_PYTHON" ./scripts/setup_egoview.sh propainter
```

The wrapper automatically uses `third_party/ProPainter`; use `PROPAINTER_ROOT`
only to select another checkout. Keep the interpreter set when running Galbot:

```bash
export PROPAINTER_PYTHON="$(conda run -n propainter which python)"
# optional:
# export PROPAINTER_ROOT=/path/to/another/ProPainter
# export PROPAINTER_CONDA_ENV=propainter
# export RETARGET_TMPDIR=/tmp/retarget_galbot
```

**License:** bundled ProPainter uses S-Lab License 1.0 and is restricted to
non-commercial use and redistribution. Commercial use requires upstream
permission. See `third_party/README.md` and `third_party/ProPainter/LICENSE`.

### 5. MANO models (optional sidecar generation)

If an episode already has `hands_keypoints.npz`, you can skip this.

Otherwise MANO FK looks for pickles under, in order:

1. `assets/mano_models/` inside this repo
2. `../../assets/mano/` relative to this repo (Open-AoE shared `assets/mano`)

Place `MANO_RIGHT.pkl` / `MANO_LEFT.pkl` there, or follow the Open-AoE MANO
download script in the parent monorepo.

### 6. Sanity check

```bash
python -m compileall retarget_galbot scripts
python -c "from retarget_galbot.robots import get_spec; print(get_spec('galbot').action_dim)"
# expect: 33
```

## Pipeline overview

```text
AoE segment (hands.npz / video)
        │
        ▼
aoe.mano_sidecar          MANO FK → hands_keypoints.npz (if needed)
        │
        ▼
galaxea.*                 palm-center features + Pinocchio TCP IK → qpos (T, 33)
        │
        ▼
egoview.render            SAM2 → ProPainter → MuJoCo ego overlay
        │
        ├──► dataset.*    LeRobot v2.1 (official LeRobotDataset API)
        └──► viz.rerun    live MuJoCo front/top + egoview in Rerun
```

| Path | Role |
|---|---|
| `retarget_galbot/galaxea/` | Galbot IK, features, config, Sapien validators |
| `retarget_galbot/egoview/` | Egoview synthesis (segment / inpaint / overlay / MuJoCo) |
| `retarget_galbot/dataset/` | LeRobot export |
| `retarget_galbot/viz/` | Rerun visualization |
| `retarget_galbot/aoe/` | MANO sidecar generation |
| `retarget_galbot/robots/` | `RobotSpec` + Galbot MJCF/URDF registration |
| `configs/galbot_dex_bimanual.yml` | Default retarget config |
| `scripts/` | CLI entrypoints |

## What this provides

- `scripts/retarget.py` — single / batch retargeting (+ optional LeRobot)
- `scripts/export_lerobot.py` — egoview overlay + LeRobot + Rerun from actions
- `scripts/run_hand_removal.py` — standalone SAM2 masks + ProPainter completion
- `scripts/visualize.py` — Rerun replay of an existing LeRobot root
- Pinocchio damped least-squares TCP IK on Galbot 33-DoF qpos
- Palm-center IK targets (mean of wrist + MCP 5/9/13/17) with episode arm-length scaling

## Acknowledgements

The following egoview / data-plumbing techniques are adapted from Open-AoE
**Phantom** (reimplemented here under Galbot-specific APIs):

| Technique | Module |
|---|---|
| Edge-fusion robot overlay (LAB Reinhard, feathered alpha, optional contact shadow) | `egoview/overlay.py` |
| SAM2 hand/arm segmentation seeded by MANO keypoints | `egoview/sam2_segment.py`, `egoview/render.py` |
| Video inpainting before robot overlay (ProPainter backend) | `egoview/propainter.py` |
| MANO FK → `hands_keypoints.npz` sidecar | `aoe/mano_sidecar.py` |
| `RobotSpec` registry + MJCF camera inject via temp-dir symlinks | `robots/` |

The exact upstream SAM2/ProPainter commits and copied licenses are recorded in
the repository-level `third_party/README.md`. No checkpoints are versioned.

## Standalone hand removal

Run stages 1 and 2 without IK, MuJoCo, LeRobot, or Rerun:

```bash
export AOE_EPISODE=/path/to/episode
export PROPAINTER_PYTHON="$(conda run -n propainter which python)"

python scripts/run_hand_removal.py \
  --episode_dir "$AOE_EPISODE" \
  --output_dir output/galbot/hand_removal \
  --max_frames 100
```

Outputs:

| File | Content |
|---|---|
| `sam2_hand_arm_mask.npy` | Lossless `(T,H,W)` bool mask for reuse |
| `sam2_hand_arm_mask.mp4` | H.264 High / `avc1` / `yuv420p` mask preview |
| `propainter_inpaint.mp4` | H.264 completed RGB video |

Use `--skip_inpaint` to inspect SAM2 only. Use
`--skip_sam --mask_input /path/to/mask.npy` to rerun ProPainter without SAM2.
The full `export_lerobot.py`/`retarget.py --write_lerobot` path calls the same
backend functions before overlaying the MuJoCo arms.

## Experiment runner (YAML + CLI)

For day-to-day sweeps, use a single YAML for defaults and override any field
from the CLI:

```bash
cp configs/experiments/example.yml configs/experiments/my_exp.yml
# edit episode_dir / output paths / max_frames in my_exp.yml

./scripts/run_experiment.sh configs/experiments/my_exp.yml
./scripts/run_experiment.sh configs/experiments/my_exp.yml --max_frames 50 --overwrite
./scripts/run_experiment.sh configs/experiments/my_exp.yml --no_retarget --do_visualize --open_rerun
```

- Template: [`configs/experiments/example.yml`](configs/experiments/example.yml)
- Runner: [`scripts/run_experiment.sh`](scripts/run_experiment.sh)
- Planner: [`scripts/exp_config.py`](scripts/exp_config.py) (YAML → argv)

Stages: **retarget** (`scripts/retarget.py`) then **visualize** (`scripts/visualize.py`
writes `.rrd`). Set `run.open_rerun: true` or pass `--open_rerun` to spawn the viewer.

### Batch POC (6 episodes × 1200 frames)

```bash
# optional: CONDA_ENV=retarget_galbot
DATA_ROOT=/path/to/poc_deliver ./scripts/run_batch_poc.sh
# single episode:
DATA_ROOT=/path/to/poc_deliver ./scripts/run_batch_poc.sh \
  --only poc_raw_video_20260202_214320_part000
```

Writes under `output/exp/<episode_name>/`:

```text
<episode_name>_actions.npy
lerobot/                 # LeRobot dataset
episode_0.rrd            # Rerun (egoview + front + top)
videos/triple_view.mp4   # egoview | front | top side-by-side
run.log
```

MP4-only from an existing LeRobot root:

```bash
python scripts/export_mp4.py \
  --lerobot_root output/exp/<name>/lerobot \
  --output_dir output/exp/<name>/videos \
  --max_frames 1200
```

## Usage

Set a path to one AoE segment directory (contains `ego_process/` or flat
`ego_hands_reconstruction/` + video):

```bash
export AOE_EPISODE=/path/to/raw_<collector>_seg_<id>
export MUJOCO_GL=egl
conda activate retarget_galbot
```

### Retarget only

```bash
python scripts/retarget.py \
  --episode_dir "$AOE_EPISODE" \
  --output_dir output/galbot \
  --max_frames 100 \
  --write_pickle
```

Outputs under `--output_dir`:

- `<segment>_actions.npy` — `(T, 33)` Galbot qpos
- `<segment>_galbot.pkl` / `.jsonl` — optional frame payloads (`--write_pickle` / `--write_jsonl`)

### LeRobot export + Rerun

Schema: `observation.images.egoview` (overlay frames),
`observation.state = qpos[t]`, `action = qpos[t+1]`.

From an existing actions file:

```bash
python scripts/export_lerobot.py \
  --episode_dir "$AOE_EPISODE" \
  --actions output/galbot/<segment>_actions.npy \
  --lerobot_root output/lerobot_galbot \
  --repo_id aoe/galbot_retarget \
  --max_frames 100 \
  --visualize \
  --rrd output/lerobot_galbot/episode_0.rrd
```

Or retarget and export in one shot:

```bash
python scripts/retarget.py \
  --episode_dir "$AOE_EPISODE" \
  --output_dir output/galbot_lerobot \
  --max_frames 20 \
  --write_lerobot \
  --visualize \
  --rrd output/galbot_lerobot/lerobot/episode_0.rrd
```

Replay an existing dataset:

```bash
python scripts/visualize.py \
  --lerobot_root output/lerobot_galbot \
  --repo_id aoe/galbot_retarget \
  --episode_index 0 \
  --rrd output/lerobot_galbot/episode_0.rrd
```

Rerun entities:

- `observation.images.egoview` — composed egoview overlay
- `mujoco/front` — live MuJoCo front view from `observation.state`
- `mujoco/top` — live MuJoCo top-down view from `observation.state`
- `state/*`, `action/*` — joint scalars

### short clip

```bash
python -m compileall retarget_galbot scripts
python scripts/export_lerobot.py \
  --episode_dir "$AOE_EPISODE" \
  --actions output/galbot/<segment>_actions.npy \
  --lerobot_root output/lerobot_galbot_smoke \
  --repo_id aoe/galbot_retarget_smoke \
  --max_frames 2 \
  --visualize \
  --rrd output/lerobot_galbot_smoke/episode_0.rrd
```

## Configuration tips

| Variable | Purpose |
|---|---|
| `MUJOCO_GL=egl` | Headless MuJoCo |
| `SAM2_CHECKPOINT` / `SAM2_MODEL_CFG` | Optional SAM2 weight / config overrides |
| `PROPAINTER_ROOT` / `PROPAINTER_PYTHON` | Optional ProPainter source override + subprocess interpreter |
| `LEROBOT_SRC` | Optional local LeRobot `src` if not installed |
| `RETARGET_TMPDIR` | Scratch dir for ProPainter frame dumps |
| `RETARGET_SAM2_CHUNK_SIZE` | SAM2 chunk length (default `300`) |
| `DATA_ROOT` | Required by `run_batch_poc.sh` (POC episode parent dir) |

Default retarget YAML: `configs/galbot_dex_bimanual.yml`
(`--config` on the CLI to override).

## RoboCOIN Galbot real-trajectory overlay

Replay the first trajectory from `Galbot_G1_use_dryer` and directly composite
the MuJoCo arms over the original head-camera video:

```bash
export MUJOCO_GL=egl
python scripts/replay_robocoin_overlay.py \
  --episode-index 0 \
  --output /home/cody/simrender/output/robocoin_galbot/episode_000000_arm_overlay.mp4
```

Generate the colored alignment diagnostic (`left=cyan`, `right=orange`, green
silhouette):

```bash
python scripts/replay_robocoin_overlay.py \
  --episode-index 0 \
  --style debug \
  --opacity 0.58 \
  --output /home/cody/simrender/output/robocoin_galbot/episode_000000_alignment_debug.mp4
```

The loader maps RoboCOIN's 21-D observation state to this package's 33-D
Galbot joint layout and checks every frame against `eef_sim_pose_state` before
rendering. Camera parameters are in
`configs/robocoin_galbot_g1_head_camera.json`.

RoboCOIN publishes the RGB resolution and FPS, but not the head-camera
intrinsics, lens distortion, or camera-to-head transform. The supplied camera
configuration is therefore a fixed pinhole calibration fitted from multiple
episode-0 arm joint centres and visually checked against the full rendered
mesh. Small residual edge differences are expected from unmodelled lens
distortion and the collision-hull meshes used by this repository.
