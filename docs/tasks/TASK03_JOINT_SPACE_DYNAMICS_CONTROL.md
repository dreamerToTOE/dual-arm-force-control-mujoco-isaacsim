# Task03 — 7-DoF Joint-Space Dynamics Control

## 学习目标

从单关节扩展到完整 7-DoF，并第一次使用机器人动力学模型。

## 核心模型

```text
M(q) qddot + C(q,qdot) qdot + g(q) = tau
```

第一阶段做：

```text
tau = Kp (q_des-q) + Kd (qdot_des-qdot) + g(q)
```

第二阶段再做 computed torque：

```text
tau = M(q) v + C(q,qdot)qdot + g(q)
v = qddot_des + Kd e_dot + Kp e
```

## 实现任务

1. 获取 `M(q)`、gravity、bias/coriolis 项；
2. 实现 7-DoF PD + gravity compensation；
3. 实现一条平滑 joint-space reference；
4. 再实现 computed torque；
5. 对比两种控制器的 tracking error 和 torque peak。

## 验收

- 7 个关节都能稳定跟踪；
- 输出 `RMS joint error`、`max joint error`、`max torque`；
- 能解释 `M/C/g` 分别表示什么；
- 能说明为什么 computed torque 比单纯 PD 更依赖模型准确性。
