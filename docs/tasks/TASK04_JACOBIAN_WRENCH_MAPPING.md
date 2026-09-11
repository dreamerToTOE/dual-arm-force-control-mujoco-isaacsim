# Task04 — Jacobian 与末端 Wrench 映射

## 学习目标

把 Task01 的静态公式真正放进控制循环，理解末端 wrench 如何转成关节 torque。

## 核心关系

```text
xdot = J(q) qdot

tau_task = J(q)^T W_des
```

## 实现任务

1. 选定 TCP frame；
2. 获取完整 6x7 Jacobian；
3. 给 TCP 一个缓慢变化的期望 wrench；
4. 通过 `J^T W_des` 计算 torque；
5. 对比不同机器人姿态下，相同末端 wrench 所需的关节 torque；
6. 观察接近奇异构型时 torque 分布如何变化。

## 验收

- 输出 `J`、`W_des`、`tau_task`；
- 能解释为什么机器人姿态改变后 torque 分配也改变；
- 明确记录 wrench 的表达坐标系；
- 不允许通过猜符号修正方向，必须用 frame 定义解释。

## 教学门禁

完成后应能用自己的话解释：

> Jacobian transpose 是如何把“末端想施加的力/力矩”映射为每个关节需要产生的力矩的。
