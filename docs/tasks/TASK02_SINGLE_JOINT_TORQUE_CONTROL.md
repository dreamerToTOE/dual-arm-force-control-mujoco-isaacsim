# Task02 — 单关节 Torque Control

## 学习目标

先把一个关节控制稳，再扩展到 7 个关节。

本 Task 的重点不是“调用位置控制器”，而是自己根据位置/速度误差计算 **joint torque**，通过动力学让关节收敛到目标角度。

## 最小控制律

P 控制：

```text
tau = Kp (q_des - q)
```

PD 控制：

```text
tau = Kp (q_des - q) + Kd (qdot_des - qdot)
```

PD + gravity compensation：

```text
tau = Kp (q_des - q) + Kd (qdot_des - qdot) + g(q)
```

其中：

- `Kp`：位置误差对应的“弹簧”作用；
- `Kd`：速度误差对应的“阻尼”作用；
- `g(q)`：抵消机器人自身重力所需的关节力矩。

## 当前实现

控制器：

```text
controllers/joint_pd.py
```

控制器与 MuJoCo 解耦，只接收：

```text
q, qdot, q_des, qdot_des, tau_ff
```

并输出：

```text
tau
```

MuJoCo 特有的状态读取、重力查询和 torque 写入只存在于 `MuJoCoAdapter`。

实验：

```text
experiments/task02_single_joint_pd.py
```

默认选择 `fr3_joint4`，从 HOME 姿态增加 `+0.20 rad` 的目标角度。其他关节目标保持 HOME，因此只有 joint4 的目标发生阶跃，但整个机械臂仍通过 torque controller 保持安全姿态。

实验自动比较：

1. P；
2. PD；
3. PD + gravity compensation。

## 运行

```bash
python3 experiments/task02_single_joint_pd.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml
```

带 MuJoCo 可视化：

```bash
python3 experiments/task02_single_joint_pd.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml \
  --viewer
```

Viewer 默认重放 `PD + gravity`。机器人先保持 HOME 约 1 秒，再给目标关节角度阶跃，便于肉眼观察运动。

## 自动输出

终端输出：

- rise time；
- overshoot；
- settling time；
- steady-state error；
- peak joint torque。

文件输出：

```text
outputs/task02/p.csv
outputs/task02/pd.csv
outputs/task02/pd_gravity.csv
outputs/task02/single_joint_comparison.png
```

曲线至少包含：

- `q(t)` 与 `q_des`；
- tracking error；
- commanded torque。

## 验收

- 三种控制均无 NaN / Inf 或持续发散；
- PD 能稳定完成 joint4 目标角度阶跃；
- 能根据曲线解释 P 与 PD 的差别；
- 能解释增大 `Kp` 后响应为什么通常更快但更容易振荡/超调；
- 能解释增大 `Kd` 为什么会增加阻尼、抑制振荡，但过大时会让动作迟钝；
- 能观察并解释重力导致的 steady-state error；
- 能解释加入 `g(q)` 后为什么稳态误差明显减小。

## 教学门禁

完成后必须能解释：

> torque controller 并不是“直接命令关节到某个角度”，而是根据状态误差计算力矩；力矩先改变角加速度，再通过系统动力学改变角速度和角度。
