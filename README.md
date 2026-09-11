# Dual-Arm Force Control with MuJoCo + Isaac Sim

面向 **Franka FR3 双机械臂力控** 的教学型研究项目。

本仓库的目标不是直接从复杂双臂 QP 开始，而是按以下顺序循序渐进：

```text
MuJoCo 基础与 FR3 torque 接口
        ↓
力 / 力矩 / wrench / Jacobian
        ↓
单关节 torque control
        ↓
7-DoF joint-space dynamics control
        ↓
Cartesian impedance
        ↓
接触力控制
        ↓
Hybrid position / force control
        ↓
双 FR3 共同物体
        ↓
Wrench distribution + QP
        ↓
Internal force control
        ↓
鲁棒性实验
        ↓
Isaac Sim / Isaac Lab 跨仿真器复现
```

## 项目定位

本项目与 `dual-arm-embodied-palletizing` 分工如下：

- **码垛项目**：任务规划、MoveIt2、FCL、松/紧协调、随机多尺寸码垛；
- **本项目**：动力学、关节力矩控制、阻抗/力控制、双臂受力协调、wrench / torque allocation。

成熟的力控算法可在后期接回码垛项目的紧协调大件搬运模块，但本项目本身应能够独立完成。

## 教学原则

1. 每个 Task 只增加一个主要新概念；
2. 先单关节，再单臂，再接触，再双臂；
3. 先理解物理意义，再实现公式；
4. MuJoCo 用于算法开发与快速验证；
5. Isaac Sim / Isaac Lab 用于跨仿真器和系统级复现；
6. 基础阶段不引入 MoveIt、FCL、视觉和码垛点规划；
7. 每个 Task 都必须有数值验收，而不是只看动画。

## 推荐机器人模型

优先使用 MuJoCo Menagerie 中的 Franka FR3 7-DoF 模型。第一阶段不研究真空吸盘流体细节，只研究机械臂动力学、末端 wrench、接触与共同物体约束。

## Task 路线

| Task | 主题 | 核心目标 |
|---|---|---|
| 00 | MuJoCo + FR3 基线 | 加载 FR3、读取状态、施加 torque |
| 01 | Force / Torque / Wrench | 学懂力、力矩、wrench、Jacobian transpose |
| 02 | 单关节 torque control | 从 1-DoF 闭环控制入门 |
| 03 | 7-DoF joint-space control | PD+重力补偿、computed torque |
| 04 | Jacobian wrench mapping | 验证 `tau = J^T W` |
| 05 | Cartesian impedance | 末端呈现弹簧-阻尼行为 |
| 06 | Contact force control | 指定法向力压住平面 |
| 07 | Hybrid position / force | 部分方向控位置、部分方向控力 |
| 08 | Dual-arm shared object | 双 FR3 共同抓持同一刚体 |
| 09 | Wrench distribution + QP | 分配左右臂末端 wrench |
| 10 | Internal force | 控制不改变物体运动的内部力 |
| 11 | Robustness experiments | 质量、摩擦、扰动、模型误差 |
| 12 | Isaac Adapter | 同一 Controller Core 移植 Isaac |
| 13 | MuJoCo vs Isaac | 跨仿真器统一实验对比 |
| 14 | Palletizing integration | 可选：接回双臂码垛紧协调 |

## 推荐目录结构

```text
controllers/
robot_model/
sim/
models/
experiments/
tests/
docs/
  ARCHITECTURE.md
  LEARNING_GUIDE.md
  tasks/
```

核心设计原则：**控制器不直接依赖 MuJoCo 或 Isaac API**。

```text
MuJoCoAdapter ─┐
               ├──> Controller Core ───> tau
IsaacAdapter ──┘
```

统一接口建议包含：

```text
get_q()
get_qdot()
get_mass_matrix()
get_gravity()
get_coriolis()
get_jacobian()
get_tcp_pose()
get_tcp_velocity()
get_tcp_wrench()
set_joint_torque(tau)
```

## 当前状态

当前为 **教学路线与工程骨架阶段**。正式实现时从 Task00 开始，按顺序验收，不建议跳过基础 Task 直接进入双臂 QP。
