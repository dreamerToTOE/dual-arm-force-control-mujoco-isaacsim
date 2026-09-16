# Task04 — Jacobian 与末端 Wrench 映射

## 学习目标

把 Task01 的静态公式真正放进控制循环，理解末端 wrench 如何转成关节 torque。

Task04 不做真正的接触力闭环；真正的接触力控制放到 Task06。这里先把最关键的映射关系学透。

## 核心关系

```text
xdot = J(q) qdot

tau_task = J(q)^T W_des
```

本项目统一约定：

```text
J rows = [vx, vy, vz, wx, wy, wz]
W      = [Fx, Fy, Fz, Mx, My, Mz]
```

Task04 中 `J` 和 `W_des` 都表达在 **BASE frame**，不允许靠“试符号”修正方向。

## 为什么是 J^T

虚功/瞬时功率一致性：

```text
tau^T qdot = W^T xdot
xdot = J qdot
```

代入后：

```text
tau^T qdot = W^T J qdot
```

因此：

```text
tau = J^T W
```

实验中会数值验证：

```text
qdot^T tau == xdot^T W
```

## 当前实现

控制器核心：

```text
controllers/wrench_mapping.py
```

只完成：

```text
tau_task = J.T @ W_des
```

它不直接调用 MuJoCo，因此后续可以复用到 Isaac Sim。

实验：

```text
experiments/task04_jacobian_wrench_mapping.py
```

实验包括四部分：

1. 打印 HOME、POSE_A、POSE_B 三种姿态下完整 `6x7 J`；
2. 对相同 BASE-frame wrench 计算 `J^T W`，比较不同姿态的关节 torque 分配；
3. 在控制循环里让一个纯力在 BASE X-Z 平面缓慢旋转，每一步重新计算 `J(q)^T W_des`；
4. 随机搜索一个较小 `sigma_min(Jv)` 的构型，观察接近平移奇异性时 torque/force transmission 的变化。

动态实验为了防止机器人完全漂走，使用：

```text
tau_total = tau_posture_PD+g + J(q)^T W_des
```

这里的 `PD+g` 只负责保持 HOME 附近姿态；Task04 关注的是额外的 `tau_task = J^T W_des`。

## 运行

```bash
python3 experiments/task04_jacobian_wrench_mapping.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml
```

带 MuJoCo Viewer：

```bash
python3 experiments/task04_jacobian_wrench_mapping.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml \
  --viewer
```

Viewer 中：

```text
黄色球      = TCP
橙色箭头    = BASE frame 下缓慢旋转的期望力
```

注意：箭头长度只用于可视化；真实数值以终端输出为准。

## 输出

```text
outputs/task04/task04_wrench_loop.csv
outputs/task04/wrench_loop.png
outputs/task04/pose_torque_comparison.png
```

`wrench_loop.png` 包括：

- `Fx/Fz` 随时间变化；
- 七个关节的 `tau_task = J^T W`；
- 机器人相对 HOME 的关节偏移。

`pose_torque_comparison.png` 对比同一个 BASE-frame wrench 在不同姿态下对应的七关节 torque 分配。

## 实际实验结果（2026-09-16）

静态测试统一使用：

```text
W_des = [0, 0, 10, 0, 0, 0]^T
```

即 BASE frame 下 +Z 方向 10 N 纯力。

HOME：

```text
tau_task = [0, -5.54499, 0, 4.71999, 0, 0.88, 0] N m
sigma_min(Jv) = 0.236594
```

POSE_A：

```text
tau_task = [0, -4.27251, -0.37676, 4.46525, -0.20556, 1.31962, 0] N m
sigma_min(Jv) = 0.200772
```

POSE_B：

```text
tau_task = [0, -5.24368, -0.31282, 3.82529, 0.13507, -0.00341, 0] N m
sigma_min(Jv) = 0.254781
```

同一个 BASE-frame 末端力在不同姿态下得到不同的关节 torque 分配，验证了 `J(q)` 对构型的依赖。

虚功/瞬时功率验证：

```text
qdot^T tau = 0.237199791 W
xdot^T W   = 0.237199791 W
|difference| = 5.551e-17 W
```

说明 `tau = J^T W` 与速度映射 `xdot = J qdot` 在功率意义下完全一致。

随机搜索得到一个接近平移奇异性的诊断构型：

```text
singular values(Jv) = [0.958289, 0.809054, 0.013787]
sigma_min(Jv) = 1.378704e-02
```

沿最弱平移方向施加 10 N：

```text
|Jv^T F_weak| = 0.137870 N m
```

这验证了：`sigma_min -> 0` 时，直接的 `J^T W` 映射并不会发散；相反，在对应弱方向上，给定末端力映射到关节空间的 torque 可以很小。真正容易产生数值放大的是 `J^-1` 或 pseudo-inverse 形式的反解。

动态循环结果：

```text
peak |tau_task| = [0.00004, 3.77221, 0.00004, 2.83594, 0.00002, 0.83123, 0] N m
peak |tau_total| = [0.00037, 29.32666, 0.03739, 21.86524, 0.83298, 2.19451, 0.0014] N m
max |q-q_home| = [0, 0.03962, 0.00001, 0.02761, 0.00003, 0.01525, 0] rad
```

说明姿态保持 PD+g 能把机器人限制在 HOME 附近，同时额外的 `J^T W` torque 随期望 wrench 连续变化。

## 关于奇异构型

本 Task 用平移 Jacobian：

```text
Jv in R^(3x7)
```

并观察：

```text
sigma_min(Jv)
```

随机搜索得到的 LOW_SIGMA 构型仅用于**运动学诊断**，不保证无自碰撞，也不会作为运动目标。

一个必须避免的误解：

> 接近奇异性并不意味着 `J^T W` 一定会发散。

真正容易出现数值爆炸的是包含 `J^-1` / pseudo-inverse 的速度、位置或某些力反解。Task04 只观察 `J^T W` 的 torque 分布和力传递特性如何随姿态改变。

## 验收

- 输出 `J`、`W_des`、`tau_task`；
- `J.shape == (6,7)`；
- `J` 和 wrench 均明确记录为 BASE frame；
- 虚功/功率关系 `qdot^T tau = xdot^T W` 数值成立；
- 同一个末端 wrench 在不同机器人姿态下得到不同的 joint torque 分布；
- 动态 wrench 循环无 NaN/Inf；
- 能解释为什么机器人姿态改变后 torque 分配也改变；
- 能正确解释奇异构型对 Jacobian 映射的影响，而不是笼统地说“奇异点 torque 一定变无穷大”。

**当前状态：PASS。**

## 教学门禁

完成后应能用自己的话解释：

> Jacobian transpose 是如何把“末端想施加的力/力矩”映射为每个关节需要产生的力矩的。
