# Task03 — 7-DoF Joint-Space Dynamics Control

## 学习目标

从单关节扩展到完整 7-DoF，并第一次在控制器中显式使用机器人动力学模型。

Task03 的重点不是把 7 个 Task02 简单复制一遍，而是开始理解机械臂的**关节耦合与逆动力学控制**。

## 动力学模型

```text
M(q) qddot + C(q,qdot) qdot + g(q) = tau
```

其中：

- `M(q)`：7x7 质量/惯性矩阵；
- `C(q,qdot) qdot`：科里奥利与离心项；
- `g(q)`：重力项；
- `tau`：7 维关节力矩。

MuJoCoAdapter 从 Task03 起提供：

```text
get_mass_matrix()
get_bias()
get_gravity()
get_coriolis_centrifugal()
```

其中：

```text
bias = C(q,qdot) qdot + g(q)
C(q,qdot) qdot = bias - gravity
```

## Controller A：7-DoF PD + gravity compensation

```text
tau = Kp (q_des-q) + Kd (qdot_des-qdot) + g(q)
```

这仍然是 Task02 的思想，但现在 7 个关节同时跟踪一条平滑轨迹。

## Controller B：Computed Torque / inverse dynamics

控制器位于：

```text
controllers/computed_torque.py
```

控制律：

```text
v = qddot_des + Kd (qdot_des-qdot) + Kp (q_des-q)

tau = M(q) v + C(q,qdot)qdot + g(q)
```

也可写成：

```text
tau = M(q) v + bias(q,qdot)
```

其思想是先设计期望的闭环加速度 `v`，再利用机器人动力学模型反推出实现这个加速度所需的关节力矩。

## 平滑 7-DoF reference

实验：

```text
experiments/task03_joint_space_dynamics.py
```

从 HOME 到一个小幅安全 7 轴目标，所有关节都运动。轨迹使用五次多项式 / minimum-jerk 时间律：

```text
s(u) = 10u^3 - 15u^4 + 6u^5
```

因此轨迹开始和结束时：

```text
qdot_des = 0
qddot_des = 0
```

避免直接给 7 个关节位置阶跃造成不必要的大力矩。

## 自动动力学检查

在 HOME 姿态自动打印：

- `M(q)`；
- `M(q)` shape；
- `max |M-M^T|`；
- `M(q)` 特征值范围；
- `g(q)`；
- 静止时 `C(q,qdot)qdot`。

质量矩阵应为：

```text
M(q) in R^(7x7)
```

并应近似对称正定。

## 运行

```bash
python3 experiments/task03_joint_space_dynamics.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml
```

带 MuJoCo Viewer：

```bash
python3 experiments/task03_joint_space_dynamics.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml \
  --viewer
```

Viewer 默认重放 `computed_torque`，7 个关节会沿平滑 reference 往返运动。

## 输出

终端打印：

- 每个关节 RMS tracking error；
- 7 轴整体 RMS error；
- 最大 joint error；
- 最大 commanded torque。

文件：

```text
outputs/task03/pd_gravity.csv
outputs/task03/computed_torque.csv
outputs/task03/tracking_comparison.png
outputs/task03/computed_torque_joint_errors.png
```

## 验收

- `M(q)` 为 `7x7`，在 HOME 近似对称正定；
- 静止时 `C(q,qdot)qdot` 近似为 0；
- 7 个关节均能稳定跟踪；
- PD+gravity 与 computed torque 均无 NaN/Inf；
- 能比较两者 `RMS joint error`、`max joint error`、`max torque`；
- 能解释 `M/C/g` 分别表示什么；
- 能解释为什么 computed torque 会显式考虑关节之间的惯性耦合；
- 能说明 computed torque 为什么比单纯 PD 更依赖模型准确性。

## 教学门禁

完成后必须能够解释：

> PD+gravity 主要根据误差直接产生关节力矩，而 computed torque 会先利用期望加速度构造 `v`，再通过 `M(q)` 和 bias 项把这个期望动力学行为转换为实际 torque，因此模型越准，动力学抵消通常越准确；模型错误也会直接进入控制力矩。
