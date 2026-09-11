# Task01 — Force / Torque / Wrench / Jacobian 基础

## 学习目标

把“力”和“力矩”真正和机械臂关节联系起来，而不是只记公式。

## 核心概念

力：

```text
F = [Fx, Fy, Fz]
```

力矩：

```text
M = [Mx, My, Mz]
```

末端 wrench：

```text
W = [Fx, Fy, Fz, Mx, My, Mz]^T
```

Jacobian：

```text
xdot = J(q) qdot
```

虚功对应关系：

```text
tau_ext = J(q)^T W
```

## 实现任务

1. 获取 FR3 TCP Jacobian；
2. 验证 `J` 维度为 `6 x 7`；
3. 给定一个简单末端 wrench，例如 world/base Z 方向力；
4. 计算 `tau = J^T W`；
5. 检查 torque 的量纲和关节方向；
6. 用有限差分验证 Jacobian 的平移部分。

## 验收

必须能独立解释：

- 为什么 `J` 是 6x7；
- 为什么速度映射使用 `J`，力映射使用 `J^T`；
- 同一个末端力为什么会在多个关节上产生力矩；
- wrench 是在哪个坐标系表达的。

## 常见错误

- 把 `J^-1` 用在 wrench 映射；
- 混淆 world/base/TCP frame；
- 把 Nm 和 N 混在一起比较。
