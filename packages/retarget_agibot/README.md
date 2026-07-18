# AgiBot G1 真機軌跡控制與回放

這個目錄先完成 **AgiBot G1 真機軌跡回放**，暫時不把 AoE ego 人體動作 retarget 到 G1。

## 當前資產與模型

`assets/robots/G1_v2.3/G1_120s/G1_120s.urdf` 包含底盤以上的完整運動學樹、關節軸、關節限位、CRT-120S 夾爪結構與 mimic 關係，因此足以完成：

- 真機關節軌跡解析與限位檢查；
- 底盤、腰部、頭部、雙臂和夾爪的逐幀控制；
- FK、後續 IK/retarget 開發；
- MuJoCo 運動回放。

官方 `G1_120s_base.usd` 中的 43 個可見網格已按 link 局部坐標導出到
`assets/robots/G1_v2.3/G1_120s/mujoco_articulated/meshes/`。MuJoCo 使用 URDF
關節樹驅動這些網格，因此手臂、頭部、腰部和左右 120S 夾爪都會逐幀運動。
`--visual-mode proxy` 仍保留 box/capsule/sphere 代理外觀作為調試回退。

## 真機數據映射

數據文件：`datasets/agibot/proprio_stats/proprio_stats.h5`

| HDF5 字段 | 形狀 | G1 控制對象 |
|---|---:|---|
| `state/joint/position[:, 0:7]` | `(T, 7)` | `idx21` 到 `idx27` 左臂 |
| `state/joint/position[:, 7:14]` | `(T, 7)` | `idx61` 到 `idx67` 右臂 |
| `state/waist/position` | `(T, 2)` | 升降關節 `idx01`、俯仰關節 `idx02` |
| `state/head/position` | `(T, 2)` | 頭部 `idx11`、`idx12` |
| `state/effector/position` | `(T, 2)` | CRT-120S 實測角度，逐側穩健歸一化為閉合比例 |
| `action/effector/position` | `(T, 2)` | 左右 CRT-120S，`0=張開`、`1=閉合` |
| `state/robot/position` | `(T, 3)` | MuJoCo free joint 平移 |
| `state/robot/orientation` | `(T, 4)` | 數據 `xyzw` 轉 MuJoCo `wxyz` |

本條數據有 2525 幀，時間戳約 30 Hz，與 `head_color.mp4` 逐幀對齊。
原視頻核對顯示：第 0 幀夾爪張開且 action 為 0；約第 500 幀已抓緊工具且 action 為 1。

## 安裝與測試

```bash
source /home/cody/miniconda3/etc/profile.d/conda.sh
conda activate retarget_galbot
cd /home/cody/simrender
python -m pip install --no-deps -e packages/retarget_agibot
python -m pytest -q packages/retarget_agibot/tests
```

## 回放命令

完整 G1_120s 回放，輸出「真機頭部相機 | 仿真跟隨視角 | 仿真世界視角」：

```bash
export MUJOCO_GL=egl
python scripts/replay_real_trajectory.py
```

短片 smoke test：

```bash
python scripts/replay_real_trajectory.py \
  --max-frames 60 \
  --output /home/cody/simrender/output/agibot/g1_120s_replay_smoke.mp4
```

只看仿真、固定底盤：

```bash
python scripts/replay_real_trajectory.py \
  --layout simulation \
  --base-mode fixed \
  --output /home/cody/simrender/output/agibot/g1_fixed_base.mp4
```

輸出文件旁會生成：

- `*_limits.json`：每個核心關節的觀測範圍、URDF 範圍與越界次數；
- 同名 `.json`：回放參數、幀率、運行時裁剪次數和資產限制說明。

視頻使用 H.264 High、`avc1`、`yuv420p` 與 faststart，便於常用播放器直接打開。

## 頭部相機疊加

本條數據包含完整的頭部 D455 投影信息：

- `parameters/head_intrinsic_params.json`：`fx/fy/cx/cy` 與 OpenCV 五參數畸變；
- `parameters/head_extrinsic_params_aligned.json`：2525 幀相機在數據對齊坐標系中的 `c2w` 位姿；
- `parameters/rs_camera_info.json`：640×480、30 FPS；
- `state/end/position`：用來將 MuJoCo `arm_l/r_end_link` 自動對齊到相機外參坐標系。

直接回放 `controller.py` 的真機 state，把官方 G1_120s mesh 渲染為彩色深度並疊加到原視頻：

```bash
export MUJOCO_GL=egl
python scripts/render_head_overlay.py \
  --dataset-dir /path/to/agibot \
  --joint-source state \
  --render-mode depth \
  --output /home/cody/simrender/output/agibot/g1_120s_head_camera_depth_overlay.mp4 \
  --render-only-output /home/cody/simrender/output/agibot/g1_120s_head_camera_depth.mp4
```

由於視頻由 16:9 相機畫面變換為 640×480，標定中 `fx != fy`。渲染器先在
640×360 等像素畫布渲染，再拉伸至 640×480，補償主點並套用原始鏡頭畸變。
疊加只包含雙臂與 CRT-120S；沒有場景深度，因此工具與仿真手臂之間不能做正確遮擋。
深度顏色沿用 Vision Banana 的 RGB 立方體路徑。实际机械臂稳健范围为
`0.233–0.688 m`，在两端各加入该跨度的 25% 后，重点区间取整为
`0.12–0.80 m`。整把尺仍保持 `0–∞` 和相同像素长度，但把红到品红的 5 个颜色段
集中分配给重点区间；8 个颜色顶点约对应
`0、0.12、0.26、0.39、0.53、0.66、0.80、∞ m`，因此机械臂深度变化更明显。
工具同时提供 RGB 到米制深度的反解，但 H.264 色度抽样、量化与
overlay 混色會破壞精確可逆性；需要精確深度時應保存原始 float/16-bit depth。
在本條純深度 H.264 視頻的 6 幀內部像素抽樣中，顏色反解深度的平均誤差約
`1.06 mm`，`95%` 誤差約 `2.72 mm`，`99%` 誤差約 `5.02 mm`；未压缩
8-bit RGB 在重点区间内的最大往返误差约 `0.27 mm`。因此可作近似
恢復和可視化驗證，但不能代替原始深度數據。Overlay 因為與原 RGB 做了 alpha
混合，若沒有原幀、精確 mask/alpha 和無損中間結果，不能直接反解深度。

## 控制方式

`AgibotG1Controller.set_frame()` 的流程是：

1. 讀取底盤世界位姿，將四元數由 `xyzw` 改排為 MuJoCo `wxyz`；
2. 按關節名稱寫入腰部、頭部、左臂和右臂角度；
3. 依 URDF range 裁剪越界值並累計 `clip_counts`；
4. 將資料的夾爪閉合比例 `0=張開、1=閉合` 反向映射到 CRT-120S 主關節 `1 rad=張開、0 rad=閉合`，並按 URDF mimic 關係更新關節 2/4；四連杆的被動關節 3 保持零位；
5. 調用 `mujoco.mj_forward()` 更新所有 link/site 世界位姿。

這是運動學回放，不是力矩或位置伺服器仿真。後續做 ego retarget 時，可以把 IK 解出的同名關節目標送進同一控制器；若要研究接觸力、閉環伺服或控制穩定性，仍需加入 actuator、接觸參數並校驗碰撞幾何與慣量。
