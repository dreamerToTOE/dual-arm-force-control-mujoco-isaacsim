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
experiments/task02_gain_sweep.py
```

默认选择 `fr3_joint4`，从 HOME 姿态增加 `+0.20 rad` 的目标角度。其他关节目标保持 HOME，因此只有 joint4 的目标发生阶跃，但整个机械臂仍通过 torque controller 保持安全姿态。

实验自动比较：

1. P；
2. PD；
3. PD + gravity compensation；
4. `Kp` 参数扫描；
5. `Kd` 参数扫描。

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

参数扫描：

```bash
python3 experiments/task02_gain_sweep.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml
```

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
outputs/task02/kp_sweep.png
outputs/task02/kd_sweep.png
```

## 实际验收结果（2026-09-14）

实验条件：`fr3_joint4`，`q0=-1.570790 rad`，阶跃 `+0.200000 rad`，PD + gravity compensation。

### Kp sweep（Kd = 10.0）

| Kp | rise time [s] | overshoot [%] | settling [s] | steady error [rad] | peak |tau| [Nm] |
|---:|---:|---:|---:|---:|---:|
| 24.8 | nan | 0.000 | nan | 3.554212e-02 | 23.9832 |
| 55.0 | 0.4520 | 0.000 | nan | 5.365815e-03 | 30.0332 |
| 110.0 | 0.1940 | 1.267 | 0.3020 | -5.683768e-04 | 41.0332 |

结论：增大 `Kp` 明显提高闭环刚度和响应速度，并减小稳态误差；但峰值 torque 增大，并开始出现超调。`Kp=110` 在当前测试中进入 2% settling band，但不能据此简单认为 `Kp` 越大越好，后续仍需考虑力矩限制、噪声、未建模动力学和鲁棒性。

### Kd sweep（Kp = 55.0）

| Kd | rise time [s] | overshoot [%] | settling [s] | steady error [rad] | peak |tau| [Nm] |
|---:|---:|---:|---:|---:|---:|
| 2.0 | 0.1680 | 19.495 | nan | -5.926895e-03 | 30.0332 |
| 10.0 | 0.4520 | 0.000 | nan | 5.365815e-03 | 30.0332 |
| 25.0 | 2.6740 | 0.000 | nan | 1.319296e-02 | 30.0332 |

结论：较小 `Kd` 响应快但超调明显；增大 `Kd` 可抑制超调，但过大时响应显著变慢。理论上 D 项在静止时趋近于零，因此它不是消除静态误差的主要手段；本次 `Kd=25` 的较大 tail error 主要反映有限仿真时间内仍在缓慢收敛，以及模型摩擦等非理想因素。

## 验收

- [x] 三种控制均无 NaN / Inf 或持续发散；
- [x] 理解 P、PD 与 PID 中 P/D 的对应关系；
- [x] 能解释重力导致的 steady-state error；
- [x] 能解释加入 `g(q)` 后稳态误差显著减小；
- [x] 能根据 gain sweep 解释 `Kp` 对刚度、速度、稳态误差、峰值力矩和超调的影响；
- [x] 能根据 gain sweep 解释 `Kd` 对阻尼、超调和响应速度的影响；
- [x] 能区分 MuJoCo `damping` 与 `frictionloss`（近似粘性阻尼 vs. 干摩擦/库仑摩擦效应）。

## 教学门禁

完成后必须能解释：

> torque controller 并不是“直接命令关节到某个角度”，而是根据状态误差计算力矩；力矩先改变角加速度，再通过系统动力学改变角速度和角度。

**Task02 status: PASS**
