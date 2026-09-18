# Task07 — Hybrid Position / Force Control

## 学习目标

Task06 已经能够在水平面法向建立稳定的 10 N 接触力。

Task07 增加的新概念只有一个：

> **用选择矩阵明确规定：哪些任务空间方向归位置控制，哪些方向归力控制。**

典型场景仍然是末端贴着水平平面运动：

```text
切向 BASE-X / BASE-Y：位置控制
法向 BASE-Z          ：力控制
姿态 Rx / Ry / Rz    ：姿态阻抗
```

## 1. 为什么需要 Hybrid Control

如果对受约束的 Z 方向同时独立要求：

```text
z = 某个固定高度
Fz = 10 N
```

两个目标可能互相冲突。

例如真实表面高度有 2 mm 误差：

- 强位置控制要求 TCP 一定到指定 z；
- 力控制要求 TCP 根据接触反力自行调整压入量。

因此正确思路是让不同方向承担不同任务。

## 2. 选择矩阵

统一 wrench 顺序：

```text
[Fx, Fy, Fz, Mx, My, Mz]
```

Task07 使用：

```text
S_p = diag(1, 1, 0, 1, 1, 1)
S_f = diag(0, 0, 1, 0, 0, 0)
```

其中：

- `S_p`：position/orientation-controlled directions；
- `S_f`：force-controlled directions。

并满足：

```text
S_p + S_f = I
S_p S_f = 0
```

也就是两个子空间互补且不重叠。

最终：

```text
W_cmd = S_p W_pos + S_f W_force
tau   = J^T W_cmd + g(q)
```

## 3. 本实验

场景：

1. 使用 Task06 的柔顺 approach 建立接触；
2. PI 法向力闭环先稳定到约 10 N；
3. 等待约 5.6 s 后，TCP 沿 BASE +X 平滑滑动 40 mm；
4. 滑动过程中：
   - X/Y 继续跟踪位置目标；
   - Z 不追踪位置，只维持 10 N；
   - orientation 继续阻抗保持。

切向轨迹使用 quintic smooth step，因此起点和终点速度均为 0。

为避免摩擦成为本 Task 的主问题，教学接触面使用适中的切向摩擦系数。

## 4. Controller Core

新增：

```text
controllers/hybrid_position_force.py
```

该模块只负责：

```text
W_pos
W_force
  ↓
S_p / S_f
  ↓
W_cmd
```

不依赖 MuJoCo，后续可直接复用于 Isaac Adapter。

## 5. 验收指标

滑动窗口内重点检查：

- X RMS / max tracking error；
- Y RMS tracking error；
- actual X travel；
- mean normal force / force error；
- force STD / ripple；
- contact-loss ratio；
- orientation error；
- peak joint torque。

数学上还自动检查：

```text
S_p + S_f = I
S_p S_f = 0
S_p^2 = S_p
S_f^2 = S_f
```

## 6. 运行

```bash
python3 experiments/task07_hybrid_position_force.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml \
  --viewer
```

## 7. 可视化

Viewer：

```text
绿色球   = 切向期望 TCP 目标
黄色球   = physical TCP probe
红色箭头 = 法向 push command
青色箭头 = measured normal contact force
```

输出：

```text
outputs/task07/hybrid_position_force.png
outputs/task07/hybrid_slide.csv
outputs/task07/task07_hybrid_scene.xml
```

## 8. 教学门禁

Task07 完成后应该能够解释：

> Hybrid position/force control 不是“位置控制和力控制全都一起开”，而是先在任务空间里把受控方向分开，再分别设计位置支路和力支路，最后组合成一个 wrench。

并且明确：

> Task07 解决的是单机械臂任务空间方向解耦；双臂之间如何分配 wrench 要到 Task09。


## 2026-09-18 第一轮实验

第一轮使用：

```text
surface timeconst = 0.015 s
surface friction = 0.20
slide distance = 40 mm
slide duration = 4.0 s
```

选择矩阵验证：

```text
S_p + S_f = I      : PASS
S_p S_f = 0        : PASS
```

但控制质量未通过：

```text
X RMS error        = 14.858 mm
X max error        = 21.028 mm
actual X travel    = 19.275 mm / 40 mm desired

mean normal force  = 10.5855 N
force STD          = 3.3393 N
force ripple       = 15.0329 N
contact loss       = 3.184 %
orientation error  = 2.251 deg
```

结论：

- selection matrices 的数学结构正确；
- 但开始切向滑动后，摩擦/接触约束通过真实机器人动力学重新耦合到法向力；
- raw contact force 出现明显 chatter，切向位置跟踪也严重滞后；
- 因此这轮只能判定“数学结构 PASS、控制质量 NOT YET PASS”。

这说明：

> 选择矩阵实现的是 command-space 的方向分工，不意味着真实机器人/接触系统的动力学完全解耦。

### 第二轮稳定化设置

为了让 Task07 先专注验证 Hybrid Position/Force 的核心概念，而不是让高摩擦 stick-slip 成为主要问题：

```text
surface timeconst : 0.015 -> 0.050 s
surface friction  : 0.20  -> 0.05
slide duration    : 4.0   -> 5.0 s
duration          : 12.0  -> 13.0 s
```

其余核心结构保持不变：

```text
S_p = diag(1,1,0,1,1,1)
S_f = diag(0,0,1,0,0,0)
F_des = 10 N
PI gains = Kp=0.8, Ki=0.8
```

第二轮新增明确的 control-quality gate：

```text
X RMS error       < 5 mm
X max error       < 10 mm
|mean force error|< 0.5 N
force STD         < 0.5 N
force ripple      < 2 N
contact loss      < 0.5 %
```

只有满足这些指标，Task07 才正式 PASS。


### 第三轮：理想法向约束面

第二轮虽然降低了摩擦并放慢滑动，但仍存在：

```text
X RMS error      = 12.974 mm
X max error      = 18.287 mm
actual X travel  = 21.054 mm / 40 mm

force STD        = 1.3079 N
force ripple     = 12.2956 N
contact loss     = 0.572 %
```

说明切向接触/摩擦仍在明显干扰本 Task 的核心教学目标。

因此第三轮将 Task07 的第一阶段进一步理想化：

```text
contact condim = 1
tangential friction = 0
```

即只保留法向接触约束，让 X/Y 运动不再受到接触摩擦影响。

同时提高切向位置支路增益：

```text
Kx = Ky = 800 N/m
Dx = Dy = 60 N*s/m
```

法向力支路保持：

```text
F_des = 10 N
PI: Kp=0.8, Ki=0.8
```

这一轮的目的不是“逃避摩擦”，而是遵守项目教学原则：**每个 Task 只增加一个主要新概念**。

Task07 只验证：

```text
X/Y -> position control
Z   -> force control
```

能否通过互补选择矩阵同时工作。

摩擦、stick-slip、接触鲁棒性将在后续 robustness task 中重新加入。
