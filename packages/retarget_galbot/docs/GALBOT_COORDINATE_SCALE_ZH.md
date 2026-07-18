# Galbot 人手坐标映射与臂长缩放详解

本文只解释 Galbot retarget 流程中的两个核心步骤：

1. 如何把人手从相机坐标系映射到机器人坐标系。
2. 如何根据人体与机器人的臂长差异缩放手部轨迹。

对应默认配置：

```yaml
source_space: camera
reference_frame: head
position_mapping: shoulder_scaled
```

配置文件见 [`configs/galbot_dex_bimanual.yml`](../configs/galbot_dex_bimanual.yml)。

## 目录

- [1. 一页概览](#1-一页概览)
- [2. 输入位置究竟是手腕还是手掌](#2-输入位置究竟是手腕还是手掌)
- [3. 第一步坐标变换：相机系到人的虚拟 head 系](#3-第一步坐标变换相机系到人的虚拟-head-系)
- [4. 第二步坐标变换：人的 head 系到机器人 base 系](#4-第二步坐标变换人的-head-系到机器人-base-系)
- [5. 合成人体肩膀](#5-合成人体肩膀)
- [6. 臂长缩放系数如何计算](#6-臂长缩放系数如何计算)
- [7. 缩放后如何生成机器人目标](#7-缩放后如何生成机器人目标)
- [8. 数值例子](#8-数值例子)
- [9. 目标如何进入 IK](#9-目标如何进入-ik)
- [10. 当前实现中最容易误解的地方](#10-当前实现中最容易误解的地方)
- [11. 参数调整建议](#11-参数调整建议)
- [12. 相关代码入口](#12-相关代码入口)

---

## 1. 一页概览

Galbot 默认流程可以概括为：

```text
MANO 相机系手部关键点
        │
        │ 计算 palm center
        ▼
相机系手掌位置 p_C
        │
        │ 只转换坐标轴，不做平移
        ▼
人的虚拟 head 系手掌位置 p_H
        │
        │ 减去虚拟肩膀中点
        ▼
肩膀到手掌的相对向量 r_H
        │
        │ 乘 episode 级统一缩放 s
        ▼
缩放后的相对向量 s · r_H
        │
        │ 用机器人初始 head 朝向旋转到 base 系
        ▼
加到机器人肩膀中点
        │
        ▼
机器人 TCP 目标位置 p_target^B
        │
        ▼
Pinocchio 双臂 IK
```

最核心的最终公式是：

$$
p_{target}^{B}(t)
=
S_{robot,mid}^{B}
+
R_{BH}^{0}
\left[
s\left(p_{palm}^{H}(t)-S_{human,mid}^{H}\right)
\right]
$$

其中：

| 符号 | 含义 |
|---|---|
| $p_{palm}^{H}(t)$ | 第 $t$ 帧人的手掌中心，位于人的虚拟 head 坐标系 |
| $S_{human,mid}^{H}$ | 合成的人体肩膀中点 |
| $s$ | 人体到机器人的 episode 级臂长缩放系数 |
| $R_{BH}^{0}$ | Galbot 初始姿态下，head 坐标系到 base 坐标系的旋转 |
| $S_{robot,mid}^{B}$ | Galbot 左右肩膀位置的中点 |
| $p_{target}^{B}(t)$ | 交给机械臂 IK 的 TCP 目标位置 |

注意：默认流程不是直接令“机器人眼手距离等于人眼手距离”。它实际保留的是“肩膀到手掌”的方向和相对轨迹，再根据机器人臂长统一缩放。

---

## 2. 输入位置究竟是手腕还是手掌

AoE 的 `hands.npz` 提供：

```text
pred_trans_cam  相机系手腕位置
pred_rot_cam    相机系手腕全局旋转
pred_hand_pose  MANO 手指姿态
pred_valid      左右手逐帧有效标记
```

加载逻辑位于 [`ego_data.py`](../retarget_galbot/galaxea/ego_data.py)。

Galbot IK 默认不直接使用 `pred_trans_cam` 作为末端位置。程序会从 MANO 21 点关键点中选取：

```text
0   wrist
5   index MCP
9   middle MCP
13  ring MCP
17  pinky MCP
```

然后计算手掌中心：

$$
p_{palm}
=
\frac{p_0+p_5+p_9+p_{13}+p_{17}}{5}
$$

实现见 [`features.py`](../retarget_galbot/galaxea/features.py) 中的 `_palm_center_from_keypoints()`。

选择手掌中心而不是手腕有两个原因：

- Galbot 的 IK 目标是左右夹爪的 TCP，语义上更接近手掌中心。
- MANO 手腕点位于手掌根部，直接映射到夹爪中心容易产生固定的前后偏差。

如果 `hands_keypoints.npz` 不存在且 MANO FK 生成失败，程序会退化为：

$$
p_{palm}=p_{wrist}
$$

---

## 3. 第一步坐标变换：相机系到人的虚拟 head 系

### 3.1 两套坐标轴

AoE 使用 OpenCV 相机坐标系：

```text
相机 C：

      +Y 向下
       │
       │
       o──── +X 向右
      /
     /
   +Z 向前
```

Galbot 代码定义的 head 坐标系为：

```text
人的虚拟 head H：

+X 向前
+Y 向下
+Z 向左
```

因此轴对应关系是：

| 相机运动 | head 系结果 |
|---|---|
| 相机向前 `+Zc` | head 向前 `+Xh` |
| 相机向下 `+Yc` | head 向下 `+Yh` |
| 相机向右 `+Xc` | head 向右 `-Zh` |

### 3.2 位置变换矩阵

代码中的矩阵为：

$$
R_{HC}
=
\begin{bmatrix}
0&0&1\\
0&1&0\\
-1&0&0
\end{bmatrix}
$$

因此：

$$
p_H=R_{HC}p_C
$$

展开后就是：

$$
\begin{bmatrix}
x_H\\y_H\\z_H
\end{bmatrix}
=
\begin{bmatrix}
z_C\\y_C\\-x_C
\end{bmatrix}
$$

实现见 [`coordinates.py`](../retarget_galbot/galaxea/coordinates.py) 中的 `opencv_camera_to_head_link_position()`。

### 3.3 这一步没有做相机外参平移

当前实现是纯坐标轴旋转：

```text
p_H = R_HC · p_C
```

而不是完整刚体变换：

```text
p_H = R_HC · p_C + t_HC
```

也就是说，程序把相机光心直接视为“人的虚拟 head 原点”。它没有显式标定：

- 手机相机到双眼中心的距离。
- 手机相机到头部旋转中心的距离。
- 手机安装在额头、胸前或颈部时产生的平移差异。

这些差异目前由后续的虚拟肩膀 offset 间接吸收。

### 3.4 手掌旋转的坐标变换

对于旋转，不能只左乘 $R_{HC}$。代码使用换基公式：

$$
R_{hand}^{H}
=
R_{HC}
R_{hand}^{C}
R_{HC}^{T}
$$

实现见 [`coordinates.py`](../retarget_galbot/galaxea/coordinates.py) 中的 `opencv_camera_to_head_link_rotation()`。

不过默认配置为：

```yaml
orientation_weight: 0.0
```

因此当前默认 IK 只追踪 TCP 位置。手掌旋转目标虽然会被计算，但不会参与 IK 优化。

---

## 4. 第二步坐标变换：人的 head 系到机器人 base 系

### 4.1 机器人初始姿态

程序首先创建 Pinocchio 模型，并生成初始关节位置：

```text
q0 = URDF neutral qpos + YAML 中的双臂 seed_qpos
```

然后通过 FK 获取：

```text
机器人 head link 位姿
机器人左肩位置
机器人右肩位置
机器人左 TCP 位姿
机器人右 TCP 位姿
```

初始化逻辑见 [`bimanual.py`](../retarget_galbot/galaxea/bimanual.py) 中的 `BimanualDexRetargeter.__init__()`。

机器人 head 在 base 中的初始位姿写作：

$$
T_{BH}^{0}
=
\begin{bmatrix}
R_{BH}^{0}&t_{BH}^{0}\\
0&1
\end{bmatrix}
$$

在默认初始姿态下，代码预期的轴关系为：

| robot head 轴 | robot base 轴 |
|---|---|
| head `+X` 向前 | base `+X` 向前 |
| head `+Y` 向下 | base `-Z` 向下 |
| head `+Z` 向左 | base `+Y` 向左 |

程序不直接硬编码第二个旋转矩阵，而是使用 URDF FK 得到的 $R_{BH}^{0}$。

### 4.2 机器人肩膀中点

左右肩膀位置由 FK 获取：

$$
S_{robot,L}^{B},\quad S_{robot,R}^{B}
$$

机器人肩膀中点为：

$$
S_{robot,mid}^{B}
=
\frac{S_{robot,L}^{B}+S_{robot,R}^{B}}{2}
$$

这个点才是默认位置映射的机器人锚点，不是机器人 head 原点。

因此虽然流程名为 `reference_frame: head`，默认 `shoulder_scaled` 模式最终仍然把目标位置锚定在机器人肩膀中点。

---

## 5. 合成人体肩膀

AoE 没有身体或肩膀关键点，所以 Galbot 代码在人的虚拟 head 系中合成固定肩膀。

默认配置：

```yaml
source_shoulder_offset_head: [0.0, -0.05, 0.0]
source_shoulder_half_width: 0.18
```

对应：

$$
S_{human,mid}^{H}=[0,-0.05,0]
$$

因为 head 系的 $+Y$ 向下，所以 $-0.05$ 表示肩膀中点位于相机/head 原点上方 5 cm。

左肩和右肩分别是：

$$
S_{human,L}^{H}=[0,-0.05,+0.18]
$$

$$
S_{human,R}^{H}=[0,-0.05,-0.18]
$$

这里 $+Z$ 向左，因此左肩使用 `+0.18`，右肩使用 `-0.18`。

实现见 [`bimanual.py`](../retarget_galbot/galaxea/bimanual.py) 中的 `_source_shoulders_head()`。

这是一个固定人体先验，不会根据视频中的人自动估计肩宽或相机安装位置。

---

## 6. 臂长缩放系数如何计算

### 6.1 人体端距离

对每个有效帧分别计算左右肩到手掌中心的距离：

$$
d_L(t)
=
\left\|
p_{palm,L}^{H}(t)-S_{human,L}^{H}
\right\|
$$

$$
d_R(t)
=
\left\|
p_{palm,R}^{H}(t)-S_{human,R}^{H}
\right\|
$$

处理规则：

- 只使用 `pred_valid` 为真的帧。
- 左右手距离放入同一个集合。
- 丢弃小于 0.12 m 的异常样本。
- 对剩余距离取第 95 百分位。

所以人体参考臂长为：

$$
L_{human}
=
P_{95}
\left(
\{d_L(t)\}\cup\{d_R(t)\}
\right)
$$

为什么使用第 95 百分位：

- 最大值容易被单帧错误重建污染。
- 平均值不能表示手臂接近伸展时的长度。
- 第 95 百分位通常能保留接近伸直的帧，同时忽略极少数离群点。

### 6.2 机器人端距离

机器人端使用初始姿态下肩膀 link 到夹爪 TCP 的直线距离：

$$
L_{robot,L}
=
\left\|
p_{TCP,L}^{0}-S_{robot,L}^{B}
\right\|
$$

$$
L_{robot,R}
=
\left\|
p_{TCP,R}^{0}-S_{robot,R}^{B}
\right\|
$$

程序取左右两侧较大的一个：

$$
L_{robot}=\max(L_{robot,L},L_{robot,R})
$$

需要注意：这里使用的是初始 seed 姿态下的肩到 TCP 直线距离，不是机械臂各连杆长度之和，也不一定等于理论最大可达距离。

### 6.3 最终比例

统一缩放系数为：

$$
s=\frac{L_{robot}}{L_{human}}
$$

如果没有合格的人体距离样本，则退化为：

$$
s=1
$$

实现见 [`bimanual.py`](../retarget_galbot/galaxea/bimanual.py) 中的 `_episode_source_to_robot_scale()`。

---

## 7. 缩放后如何生成机器人目标

### 7.1 相对人体肩膀中点

每一帧先计算手掌相对人体肩膀中点的向量：

$$
r_H(t)
=
p_{palm}^{H}(t)-S_{human,mid}^{H}
$$

注意这里有一个细节：

- 计算缩放系数时，左手使用左肩，右手使用右肩。
- 生成最终目标时，两只手都相对肩膀中点计算。

这样做可以让左右手共享同一个空间原点，更好地保留双手间的相对位置和操作几何。

### 7.2 应用统一缩放

$$
r_{scaled}^{H}(t)=s\,r_H(t)
$$

这是一个标量缩放：

```text
x、y、z 使用同一个 s
左手、右手使用同一个 s
整个 episode 使用同一个 s
```

这样不会逐帧改变轨迹比例，也不会分别拉伸左右手。

### 7.3 旋转到机器人 base 并重新锚定

将缩放后的向量用机器人初始 head 朝向旋转到 base 系：

$$
r_B(t)=R_{BH}^{0}r_{scaled}^{H}(t)
$$

然后加到机器人肩膀中点：

$$
p_{target}^{B}(t)
=
S_{robot,mid}^{B}+r_B(t)
$$

合并得到：

$$
\boxed{
p_{target}^{B}(t)
=
S_{robot,mid}^{B}
+
R_{BH}^{0}
\left[
s\left(p_{palm}^{H}(t)-S_{human,mid}^{H}\right)
\right]
}
$$

实现见 [`bimanual.py`](../retarget_galbot/galaxea/bimanual.py) 中的 `_target_in_head_shoulder_scaled()` 和 `_target_pose()`。

---

## 8. 数值例子

假设某一帧右手掌在相机系中为：

$$
p_C=[0.20,0.25,0.60]\text{ m}
$$

含义是：

```text
向右 0.20 m
向下 0.25 m
向前 0.60 m
```

转换到人的虚拟 head 系：

$$
p_H=[z_C,y_C,-x_C]=[0.60,0.25,-0.20]
$$

右肩位置为：

$$
S_{human,R}^{H}=[0,-0.05,-0.18]
$$

这一帧的右肩到手掌距离为：

$$
d_R
=
\sqrt{
(0.60-0)^2
+(0.25+0.05)^2
+(-0.20+0.18)^2
}
\approx0.671\text{ m}
$$

假设整段视频统计得到：

```text
人体距离 P95 = 0.70 m
机器人肩到 TCP 距离 = 0.56 m
```

则：

$$
s=0.56/0.70=0.8
$$

手掌相对人体肩膀中点的向量为：

$$
r_H
=
[0.60,0.25,-0.20]-[0,-0.05,0]
=
[0.60,0.30,-0.20]
$$

缩放后：

$$
r_{scaled}^{H}
=
0.8[0.60,0.30,-0.20]
=
[0.48,0.24,-0.16]
$$

如果机器人 head 处于默认朝向，这大致对应 base 系中的：

```text
从机器人肩膀中点向前 0.48 m
向右 0.16 m
向下 0.24 m
```

最终还要加上机器人肩膀中点在 base 系中的绝对位置，得到完整 TCP 目标。

---

## 9. 目标如何进入 IK

左右 TCP 目标生成后，程序依次求解：

```text
上一帧 qpos
   │
   ├── 左臂 IK -> 更新左臂 7 个关节
   │
   └── 右臂 IK -> 在左臂结果基础上更新右臂 7 个关节
```

默认 IK 参数：

| 参数 | 默认值 |
|---|---:|
| 阻尼 `damping` | 0.001 |
| 更新步长 `step_size` | 0.6 |
| 最大迭代次数 | 80 |
| 位置容差 | 0.002 m |
| 姿态权重 | 0.0 |

位置误差为：

$$
e_p=p_{target}^{B}-p_{TCP}^{B}(q)
$$

默认使用阻尼最小二乘更新：

$$
\Delta q
=
J^T
\left(JJ^T+\lambda I\right)^{-1}
e_p
$$

然后：

$$
q\leftarrow q+0.6\Delta q
$$

并裁剪到 URDF 关节限制。实现见 [`pinocchio_ik.py`](../retarget_galbot/galaxea/pinocchio_ik.py)。

---

## 10. 当前实现中最容易误解的地方

### 10.1 不是直接映射眼手距离

默认映射的核心量是：

```text
手掌 - 合成肩膀中点
```

而不是：

```text
手掌 - 真实眼睛位置
```

相机只提供了一个稳定的随头参考系。

### 10.2 机器人 head 的平移不是最终锚点

代码中会把目标暂时表示到机器人 head 系，但默认最终位置锚定在机器人肩膀中点。机器人 head 的旋转用于确定方向，head 的平移在坐标往返过程中会抵消。

### 10.3 默认不会按工作空间裁剪

配置文件中虽然存在：

```yaml
workspace:
  left:
    scale: ...
    min: ...
    max: ...
```

但这些参数只用于旧的：

```text
position_mapping: absolute
position_mapping: delta
```

默认 `position_mapping: shoulder_scaled` 不执行 `workspace min/max` 裁剪。超出机械臂可达范围时，只能由 IK 和关节限制被动处理。

### 10.4 肩膀参数是固定先验

当前 Galbot 流程不会从人体图像估计真实肩膀，也不会自动优化：

```yaml
source_shoulder_offset_head
source_shoulder_half_width
```

而且 `RetargetSession.retarget()` 接收的 `refine_shoulder` 参数当前会被直接忽略。

### 10.5 缩放值没有写入最终 RetargetResult

`source_scale` 在 `BimanualDexRetargeter.retarget()` 内部计算并使用，但当前没有赋值给外层 `RetargetResult.scale`。如果需要调试实际缩放值，应增加日志或把该值加入 frame/meta 输出。

---

## 11. 参数调整建议

当机器人手整体过远或过近时，优先检查：

| 现象 | 优先检查 |
|---|---|
| 双手整体太低 | `source_shoulder_offset_head[1]` 是否合适 |
| 双手左右分得太开 | `source_shoulder_half_width` 是否过大 |
| 整段动作幅度过大 | 人体 P95 是否因动作未伸展而偏小 |
| 整段动作幅度过小 | MANO 深度是否偏大，或 P95 是否被异常远点拉大 |
| 前后方向错误 | `CAMERA_TO_HEAD_LINK` 轴映射是否与相机约定一致 |
| 单手可达、另一手不可达 | 左右手重建尺度、肩宽先验或机器人初始 seed 是否不对称 |
| IK 经常到达关节限位 | 目标超出工作空间，默认模式又没有主动裁剪 |

更严格的工程实现应增加：

- 相机到人体 head/肩膀的外参标定。
- 基于人体体型或视觉关键点的肩宽估计。
- 对 `source_scale` 设置合理上下限。
- 在 `shoulder_scaled` 模式中增加机器人可达空间投影。
- 把每帧 TCP 目标、IK 误差和 episode scale 写入诊断输出。

---

## 12. 相关代码入口

| 功能 | 文件 |
|---|---|
| 加载 `hands.npz` | [`ego_data.py`](../retarget_galbot/galaxea/ego_data.py) |
| 计算手掌中心 | [`features.py`](../retarget_galbot/galaxea/features.py) |
| 相机轴到 head 轴 | [`coordinates.py`](../retarget_galbot/galaxea/coordinates.py) |
| 合成肩膀与计算 scale | [`bimanual.py`](../retarget_galbot/galaxea/bimanual.py) |
| 生成 TCP 目标 | [`bimanual.py`](../retarget_galbot/galaxea/bimanual.py) |
| Pinocchio IK | [`pinocchio_ik.py`](../retarget_galbot/galaxea/pinocchio_ik.py) |
| 默认参数 | [`galbot_dex_bimanual.yml`](../configs/galbot_dex_bimanual.yml) |
