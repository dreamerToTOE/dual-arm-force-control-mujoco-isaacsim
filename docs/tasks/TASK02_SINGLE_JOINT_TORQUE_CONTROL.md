# Task02 — 单关节 Torque Control

## 学习目标

先把一个关节控制稳，再扩展到 7 个关节。

## 最小模型

先使用 PD：

```text
tau = Kp (q_des - q) + Kd (qdot_des - qdot)
```

## 实现任务

1. 只选一个 FR3 关节做目标角度阶跃；
2. 其他关节保持安全姿态；
3. 从较小 `Kp/Kd` 开始调；
4. 记录 rise time、overshoot、settling time、steady-state error；
5. 比较：只有 P、PD、PD + gravity compensation。

## 验收

- 目标角度稳定收敛；
- 无持续发散或高频振荡；
- 能通过曲线解释增大 `Kp`、`Kd` 后发生了什么；
- 能解释为什么重力会造成 steady-state error，以及重力补偿为什么有帮助。

## 教学门禁

完成后必须能解释：

> torque controller 并不是“直接命令关节到某个角度”，而是通过施加力矩改变系统动力学状态。
