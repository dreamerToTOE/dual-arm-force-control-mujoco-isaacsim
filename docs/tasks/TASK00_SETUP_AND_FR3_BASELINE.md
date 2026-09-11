# Task00 — MuJoCo + FR3 环境与 Torque 基线

## 学习目标

先回答最基础的问题：机械臂在 MuJoCo 里有哪些状态量？怎样读取关节位置/速度？怎样直接施加关节力矩？

## 需要理解

- `q` 与 `qdot`；
- actuator 与 joint 的区别；
- position actuator 与 torque/force actuator 的区别；
- simulation timestep；
- 7 个 FR3 关节的顺序与限制。

## 实现任务

1. 安装 MuJoCo Python；
2. 加载 Franka FR3 模型；
3. 打印 joint / actuator 名称与 index；
4. 读取 7 维 `q`、`qdot`；
5. 对单个关节施加一个很小的 torque pulse；
6. 记录其角度、速度响应；
7. 建立 `MuJoCoAdapter` 最小接口：`get_q/get_qdot/set_joint_torque`。

## 验收

- 模型稳定加载，不出现 NaN；
- 关节顺序记录到文档；
- torque pulse 只作用于指定关节；
- 日志可以画出 `q(t)`、`qdot(t)`、`tau(t)`；
- 能解释为什么施加 torque 后关节不是瞬间跳到目标角度。

## 不做

- 不做双臂；
- 不做末端控制；
- 不做接触；
- 不接 Isaac。
