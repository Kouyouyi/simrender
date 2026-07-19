# simrender

`simrender` 收集了兩套可以獨立於 Open-AoE 主倉庫運行的機器人流程：

- **AgiBot G1 120S**：讀取真機 trajectory/state/action，控制 MuJoCo 模型，輸出第三人稱回放、頭部相機 RGB/彩色深度渲染及原視頻疊加。
- **Galbot**：把 AoE ego MANO 手部重建 retarget 成 33 維 Galbot qpos，做 MuJoCo 動作回放、胸前相機直接疊加、舊/新投影比較、RoboCOIN 真機軌跡疊加、LeRobot/Rerun 導出，以及可選的 SAM2 + ProPainter egoview 合成。

數據集、MANO 模型、SAM2/ProPainter 權重不在倉庫內。SAM2 與 ProPainter
推理源碼、G1 120S 和 Galbot 仿真所需的 URDF/MJCF/mesh 已隨倉庫提供。

## 目錄

```text
simrender/
├── assets/robots/G1_v2.3/G1_120s/       # G1 120S URDF、官方 OBJ mesh、導出場景
├── packages/retarget_agibot/            # AgiBot 控制、回放、相機疊加
├── packages/retarget_galbot/            # Galbot retarget、回放、疊加與資產
├── third_party/{sam2,ProPainter}/        # 固定版本的無權重推理源碼
├── scripts/                              # Egview 依賴安裝與權重下載
├── environment.yml
└── output/                               # 運行時生成，Git 忽略
```

Galbot 模型資源位於 `packages/retarget_galbot/assets/robots/galbot/`，包括 `galbot_one_golf.urdf`、`galbot_one_golf_with_sites.xml` 和完整 GLB/STL mesh。模型來自 GalaxyGeneralRobotics 公開的 Apache-2.0 倉庫，commit 與本地兼容修改記錄在該目錄的 `SOURCE.md`。

## 環境配置

### 1. 創建環境並安裝本倉庫

```bash
git clone git@github.com:Kouyouyi/simrender.git
cd simrender

source /home/cody/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate simrender

python -m pip install --no-deps -e packages/retarget_agibot
python -m pip install --no-deps -e packages/retarget_galbot
```

`environment.yml` 使用 Python 3.11，並安裝 MuJoCo、OpenCV、Pinocchio 的 PyPI 發行包 `pin`、PyTorch、FFmpeg 和兩個包的核心依賴。若機器需要特定 CUDA 版 PyTorch，先按 PyTorch 官方命令安裝匹配版本，再用上面的 `--no-deps` editable install，避免 pip 替換已驗證的 CUDA build。

無顯示器服務器使用 EGL：

```bash
export MUJOCO_GL=egl
```

若 EGL 不可用，可在有桌面的機器改為 `MUJOCO_GL=glfw`；軟件渲染可嘗試 `MUJOCO_GL=osmesa`。

### 2. 基礎驗證

```bash
python -c "import mujoco, pinocchio, retarget_agibot, retarget_galbot; print(mujoco.__version__, pinocchio.__version__)"
python -m pytest -q packages/retarget_agibot/tests packages/retarget_galbot/tests
```

不提供外部 AgiBot/RoboCOIN 數據時，數據內容測試會自動跳過；模型加載、投影幾何和深度顏色尺測試仍會執行。

## 外部資源

### MANO

只有在 AoE episode 缺少 `hands_keypoints.npz`、需要從 `hands.npz` 做 MANO FK 時才需要：

```text
packages/retarget_galbot/assets/mano_models/MANO_LEFT.pkl
packages/retarget_galbot/assets/mano_models/MANO_RIGHT.pkl
```

也可放在倉庫根目錄的 `assets/mano/`。MANO 有單獨授權，本倉庫不分發模型文件。如果 sidecar 和 MANO 都不存在，retarget 仍可退化到 wrist translation，但 palm center、方向和尺度會較不準確。

### SAM2 與 ProPainter

它們只用於「先移除真人手臂，再疊加 Galbot」的完整 egoview/LeRobot
流程。直接胸前相機疊加和普通 MuJoCo 回放不需要它們。倉庫已固定 SAM2
與 ProPainter 的推理源碼版本，來源 commit 和授權見 `third_party/README.md`。

在 Galbot 主環境安裝 SAM2 依賴並下載全部權重：

```bash
./scripts/setup_egoview.sh sam2
./scripts/download_egoview_checkpoints.sh
```

默認 checkpoint 是 Git 忽略的
`checkpoints/sam2/sam2.1_hiera_base_plus.pt`，也可用 `SAM2_CHECKPOINT`
覆蓋。包裝器會自動發現 `third_party/sam2`，無需另行 clone。

ProPainter 建議使用獨立環境，由主進程以 subprocess 調用：

```bash
conda create -n propainter python=3.10 -y
export PROPAINTER_PYTHON="$(conda run -n propainter which python)"
PROPAINTER_PYTHON="$PROPAINTER_PYTHON" ./scripts/setup_egoview.sh propainter
```

包裝器默認使用 `third_party/ProPainter`，也可用 `PROPAINTER_ROOT` 覆蓋。
可選調節項包括 `PROPAINTER_CHUNK_SIZE`、`PROPAINTER_SUBVIDEO_LENGTH`、
`PROPAINTER_MASK_DILATION` 和 `RETARGET_TMPDIR`。

**ProPainter 的 S-Lab License 1.0 只允許非商業使用與再分發。** 商業使用
需要先取得上游作者許可；本倉庫的 Apache-2.0 授權不會覆蓋這項限制。

### LeRobot、Rerun 與 SAPIEN

普通 `.npy` retarget 和直接 MP4 疊加不需要這些包。需要數據集導出或 Rerun 時安裝：

```bash
python -m pip install av 'datasets==3.6.0' 'huggingface_hub>=0.34.2' jsonlines rerun-sdk
python -m pip install -e /path/to/lerobot
```

也可以設置 `LEROBOT_SRC=/path/to/lerobot/src`。`retarget_galbot/galaxea/sapien_*` 是可選的交互驗證工具，需要另外安裝與本機 CUDA/圖形棧兼容的 SAPIEN。

## AgiBot G1 120S

### 數據目錄

最小真機回放只需要 HDF5；要並排顯示原視頻則還需要 `head_color.mp4`：

```text
/path/to/agibot/
├── proprio_stats/proprio_stats.h5
└── observations/head_color.mp4
```

頭部相機投影和 overlay 還需要：

```text
/path/to/agibot/
├── parameters/head_intrinsic_params.json
├── parameters/head_extrinsic_params_aligned.json
├── parameters/rs_camera_info.json
├── proprio_stats/proprio_stats.h5
└── observations/head_color.mp4
```

主要 HDF5 映射如下：

| HDF5 字段 | 用途 |
|---|---|
| `state/joint/position[:, 0:7]` | 左臂 7 DoF |
| `state/joint/position[:, 7:14]` | 右臂 7 DoF |
| `state/waist/position` | 升降與俯仰 |
| `state/head/position` | 頭部 2 DoF |
| `state/effector/position` | 實測左右夾爪狀態 |
| `action/effector/position` | 指令夾爪狀態，`0=開`、`1=閉` |
| `state/robot/position` | 底盤世界平移 |
| `state/robot/orientation` | 底盤世界四元數，數據 `xyzw` 轉 MuJoCo `wxyz` |
| `state/end/position` | 模型末端與相機外參坐標系自動對齊 |

### 1. 真機軌跡回放

```bash
export AGIBOT_DATA=/path/to/agibot
export MUJOCO_GL=egl

python packages/retarget_agibot/scripts/replay_real_trajectory.py \
  --trajectory "$AGIBOT_DATA/proprio_stats/proprio_stats.h5" \
  --source-video "$AGIBOT_DATA/observations/head_color.mp4" \
  --joint-source state \
  --visual-mode official \
  --layout source_follow_world \
  --output output/agibot/g1_120s_replay.mp4
```

常用選項：

| 選項 | 說明 |
|---|---|
| `--joint-source state|action` | 使用實測 state 或指令 action；默認 state |
| `--base-mode world|relative|fixed` | 保留世界底盤運動、以首幀為原點，或固定底盤 |
| `--visual-mode official|proxy` | 官方 link-local OBJ 或 primitive 調試模型 |
| `--layout simulation|source_simulation|source_follow_world` | 單仿真、原片+仿真、原片+跟隨+世界視角 |
| `--start-frame/--max-frames/--stride` | 選取回放區間與抽幀 |
| `--show-joints` | 顯示 MuJoCo 關節可視化 |

控制器直接寫 `qpos` 後調用 `mujoco.mj_forward()`，屬於運動學 state replay，不是 actuator、力矩或閉環伺服仿真。所有寫入值會按 URDF range 裁剪並計數。輸出旁會生成同名元數據 JSON 和 `*_limits.json` 限位報告。

### 2. 頭部相機 RGB/彩色深度疊加

彩色深度：

```bash
python packages/retarget_agibot/scripts/render_head_overlay.py \
  --dataset-dir "$AGIBOT_DATA" \
  --joint-source state \
  --render-mode depth \
  --output output/agibot/g1_120s_head_depth_overlay.mp4 \
  --render-only-output output/agibot/g1_120s_head_depth_only.mp4
```

普通 RGB 模型：

```bash
python packages/retarget_agibot/scripts/render_head_overlay.py \
  --dataset-dir "$AGIBOT_DATA" \
  --render-mode rgb \
  --output output/agibot/g1_120s_head_rgb_overlay.mp4 \
  --render-only-output output/agibot/g1_120s_head_rgb_only.mp4
```

流程使用逐幀 `c2w` 相機外參、內參/畸變、視頻尺寸和 `state/end/position` 求模型坐標到數據坐標的剛體對齊。MuJoCo 先用等像素 pinhole 相機渲染，再補償主點、非等比例 resize 和 OpenCV 鏡頭畸變。

深度模式把機械臂 metric depth 映射到 Vision Banana RGB 路徑。重點範圍為約 `0.12-0.80 m`，可用 `retarget_agibot.depth_color.focused_rgb_to_depth()` 做近似反解。H.264 `yuv420p`、8-bit 量化和 overlay alpha 混色均會破壞精確可逆性；若深度是算法輸入，應另外保存 float32/uint16 原始深度，彩色 MP4 只作可視化。

更完整的 G1 控制和相機說明見 [`packages/retarget_agibot/README.md`](packages/retarget_agibot/README.md)。

## Galbot Ego Retarget

### AoE episode 結構

推薦目錄：

```text
/path/to/episode/
├── ego_process/
│   ├── ego_hands_reconstruction/
│   │   ├── hands.npz
│   │   └── hands_keypoints.npz          # 可選；缺少時嘗試 MANO FK
│   └── ego_undistorted_video/
│       ├── raw_video_undistorted.mp4
│       └── undistorted_video_info.json  # fx/fy/cx/cy 或相機信息
└── ego_annotation/
    └── ego_action_annotation.json       # 可選任務標籤
```

也支持直接把 `ego_hands_reconstruction/` 或 `hands.npz` 傳給 retarget loader。`hands.npz` 至少應包含 `pred_valid`、`pred_trans_cam`、`pred_rot_cam`、`pred_hand_pose` 和 `pred_betas`；world-space 路徑另外使用 `pred_trans`、`pred_rot`。

輸出 action 是 `(T, 33)` qpos，順序為：5 個升降/軀幹關節、左臂 7、左夾爪 6、右臂 7、右夾爪 6、頭部 2。默認 pipeline 求解雙臂 TCP IK；底盤不是這個 33 維 action 的一部分。

### 1. 只做 retarget，輸出 action

```bash
export AOE_EPISODE=/path/to/episode
export MUJOCO_GL=egl

python packages/retarget_galbot/scripts/retarget.py \
  --episode_dir "$AOE_EPISODE" \
  --output_dir output/galbot \
  --write_pickle \
  --write_jsonl
```

短片測試可加 `--max_frames 30`；批量處理用 `--data_root /path/to/episodes --max_episodes N`。主要輸出為 `<episode>_actions.npy`，可選輸出逐幀 `.pkl`/`.jsonl`。

默認配置是 `packages/retarget_galbot/configs/galbot_dex_bimanual.yml`。可用 `--config /path/to/config.yml` 覆蓋肩膀假設、source scale、TCP、IK 權重、關節 seed 和限位相關參數。

### 2. Retarget 並直接疊加到 ego 視頻

新流程不縮放人體手部距離，從源相機到肩膀中點的 offset 構造胸前相機：

```bash
python packages/retarget_galbot/scripts/render_chest_overlay.py \
  --episode_dir "$AOE_EPISODE" \
  --mode chest_unscaled \
  --actions_output output/galbot/actions_unscaled.npy \
  --output_mp4 output/galbot/chest_unscaled_overlay.mp4
```

原有縮放與 legacy 投影：

```bash
python packages/retarget_galbot/scripts/render_chest_overlay.py \
  --episode_dir "$AOE_EPISODE" \
  --mode legacy_scaled \
  --actions_output output/galbot/actions_legacy.npy \
  --output_mp4 output/galbot/legacy_scaled_overlay.mp4
```

這兩條直接 overlay 不調用 SAM2 或 ProPainter，原視頻中的真人手臂仍然存在，也沒有場景深度遮擋。

### 3. 一次生成舊/新投影比較視頻

```bash
python packages/retarget_galbot/scripts/render_overlay_comparison.py \
  --episode_dir "$AOE_EPISODE" \
  --output_dir output/galbot/comparison \
  --max_frames 100
```

輸出包括 legacy action、新 action、兩個單獨 overlay 和左右拼接的 `legacy_scaled_vs_chest_unscaled.mp4`。

### 4. 完整 egoview、LeRobot 與 Rerun

此流程需要 MANO sidecar 或 MANO 模型，以及可用的 SAM2、ProPainter、LeRobot；流程為真人手/臂分割、inpaint、MuJoCo robot overlay、LeRobot v2.1 寫出：

先只檢查 SAM2 分割與 ProPainter 補全，不做 IK 或機器人渲染：

```bash
export PROPAINTER_PYTHON="$(conda run -n propainter which python)"
python packages/retarget_galbot/scripts/run_hand_removal.py \
  --episode_dir "$AOE_EPISODE" \
  --output_dir output/galbot/hand_removal \
  --max_frames 100
```

輸出包括無損 bool mask `sam2_hand_arm_mask.npy`、H.264 mask 預覽
`sam2_hand_arm_mask.mp4` 和補全視頻 `propainter_inpaint.mp4`。可用
`--skip_sam --mask_input <mask.npy>` 從已有 mask 重跑 ProPainter，或用
`--skip_inpaint` 只生成分割。

完整流程命令：

```bash
python packages/retarget_galbot/scripts/retarget.py \
  --episode_dir "$AOE_EPISODE" \
  --output_dir output/galbot/full \
  --write_lerobot \
  --lerobot_root output/galbot/full/lerobot \
  --repo_id aoe/galbot_retarget \
  --overwrite \
  --visualize \
  --rrd output/galbot/full/episode_0.rrd
```

已有 action 時可跳過 IK：

```bash
python packages/retarget_galbot/scripts/export_lerobot.py \
  --episode_dir "$AOE_EPISODE" \
  --actions output/galbot/actions_unscaled.npy \
  --lerobot_root output/galbot/lerobot \
  --repo_id aoe/galbot_retarget \
  --overwrite \
  --visualize \
  --rrd output/galbot/episode_0.rrd
```

只回放已有 LeRobot 數據：

```bash
python packages/retarget_galbot/scripts/visualize.py \
  --lerobot_root output/galbot/lerobot \
  --repo_id aoe/galbot_retarget \
  --episode_index 0 \
  --rrd output/galbot/replay.rrd
```

導出 `egoview | mujoco/front | mujoco/top` 三聯 MP4：

```bash
python packages/retarget_galbot/scripts/export_mp4.py \
  --lerobot_root output/galbot/lerobot \
  --output_dir output/galbot/videos \
  --episode_index 0
```

### 5. YAML experiment runner

```bash
cd packages/retarget_galbot
cp configs/experiments/example.yml configs/experiments/my_exp.yml
# 修改 episode_dir、output_dir、max_frames 等路徑
./scripts/run_experiment.sh configs/experiments/my_exp.yml
```

CLI 可覆蓋 YAML，例如 `--max_frames 50 --overwrite`。批量模板是 `scripts/run_batch_poc.sh`，通過 `DATA_ROOT=/path/to/episodes` 指定 episode 根目錄。

### 6. RoboCOIN Galbot 真機軌跡疊加

RoboCOIN 數據不隨倉庫提供。數據根目錄應包含 parquet state/EEF pose、episode metadata 和相機視頻；具體文件名由 `retarget_galbot.robocoin.RoboCoinEpisode` 解析。

```bash
python packages/retarget_galbot/scripts/replay_robocoin_overlay.py \
  --dataset-root /path/to/RoboCOIN/Galbot_G1_use_dryer \
  --episode-index 0 \
  --style realistic \
  --output output/robocoin_galbot/episode_000000_overlay.mp4
```

對齊診斷：

```bash
python packages/retarget_galbot/scripts/replay_robocoin_overlay.py \
  --dataset-root /path/to/RoboCOIN/Galbot_G1_use_dryer \
  --episode-index 0 \
  --style debug \
  --opacity 0.58 \
  --outline \
  --output output/robocoin_galbot/episode_000000_debug.mp4
```

程序把 21 維 published observation state 展開成 33 維 Galbot qpos，並在渲染前用 `eef_sim_pose_state` 驗證 FK。RoboCOIN 未發布完整相機內外參，因此 `configs/robocoin_galbot_g1_head_camera.json` 是擬合標定，邊緣存在少量殘差是預期行為。

Galbot 內部流程、座標縮放和肩膀/相機假設的詳細說明見：

- [`packages/retarget_galbot/README.md`](packages/retarget_galbot/README.md)
- [`packages/retarget_galbot/docs/GALBOT_COORDINATE_SCALE_ZH.md`](packages/retarget_galbot/docs/GALBOT_COORDINATE_SCALE_ZH.md)

## 視頻格式與深度精度

所有最終 MP4 writer 使用 H.264/AVC、High profile、`avc1` tag、`yuv420p` 和 faststart，以優先保證播放器、瀏覽器和聊天工具兼容。可用以下命令驗證：

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,profile,codec_tag_string,pix_fmt,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 output/example.mp4
```

H.264 和 AV1 的有損 `yuv420p` 都不適合保存需要精確反解的深度。相同碼率下 AV1 通常失真更低，但精確深度應保存為 float32 `.npy/.npz`、uint16 PNG/TIFF 或 FFV1/無損 4:4:4 文件；彩色 depth MP4 僅作可視化。

## 資產與授權

源代碼沿用 Open-AoE 的 Apache-2.0 授權，見 [`LICENSE`](LICENSE)。Galbot
G1 Golf 資產和 SAM2 源碼使用 Apache-2.0；各自目錄保留上游 LICENSE、
README 和固定 commit。ProPainter 源碼使用只允許非商業用途的 S-Lab
License 1.0。MANO、LeRobot、AgiBot G1 120S 資產和數據集仍可能有各自
授權或使用條款；完整來源清單見 `third_party/README.md`。
