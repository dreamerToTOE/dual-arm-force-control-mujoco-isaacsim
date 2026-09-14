# Task00 — MuJoCo + FR3 环境与 Torque 基线

## 学习目标

先回答最基础的问题：机械臂在 MuJoCo 里有哪些状态量？怎样读取关节位置/速度？怎样直接施加关节力矩？

## 需要理解

- `q` 与 `qdot`；
- actuator 与 joint 的区别；
- position actuator 与 torque/force actuator 的区别；
- simulation timestep；
- 7 个 FR3 关节的顺序与限制。

## 一个必须先弄清楚的点

MuJoCo Menagerie 当前的 FR3 模型自带的是 **position actuator**。因此：

```python
data.ctrl[i] = 1.0
```

在原模型中表示的是“给第 i 个 position actuator 一个位置参考”，**不是 1 N·m 力矩**。

Task00 为了把概念分清，采取下面的做法：

1. 仍然加载 Menagerie 的 FR3 模型；
2. 运行实验前关闭原模型的 position servo；
3. 通过 MuJoCo 的 `qfrc_applied` 对 7 个关节自由度施加明确的 generalized torque；
4. 从 Task02/Task03 开始再逐步建立正式的 torque-control 模型和控制器。

这样可以避免 position controller 与 torque pulse 同时作用而混淆结果。

## 实现任务

1. 安装 MuJoCo Python；
2. 加载 Franka FR3 模型；
3. 打印 joint / actuator 名称与 index；
4. 读取 7 维 `q`、`qdot`；
5. 对单个关节施加一个很小的 torque pulse；
6. 记录其角度、速度响应；
7. 建立 `MuJoCoAdapter` 最小接口：`get_q/get_qdot/set_joint_torque`。

仓库中对应文件：

```text
sim/mujoco_adapter.py
experiments/task00_torque_pulse.py
requirements.txt
```

## 本机安装

建议在单独 Python 虚拟环境中运行：

```bash
git clone https://github.com/dreamerToTOE/dual-arm-force-control-mujoco-isaacsim.git
cd dual-arm-force-control-mujoco-isaacsim

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

然后获取 MuJoCo Menagerie：

```bash
cd ~
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

FR3 场景文件应位于：

```text
~/mujoco_menagerie/franka_fr3/scene.xml
```

## 第一次运行

回到本仓库：

```bash
cd ~/dual-arm-force-control-mujoco-isaacsim
source .venv/bin/activate

python experiments/task00_torque_pulse.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml \
  --joint 4 \
  --torque 1.0 \
  --pulse-duration 0.05
```

默认实验为了把 torque pulse 的作用看得更清楚，会临时关闭重力。此处只是 Task00 的隔离实验，不代表后续动力学控制会忽略重力。

如果要观察带重力情况：

```bash
python experiments/task00_torque_pulse.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml \
  --joint 4 \
  --torque 1.0 \
  --pulse-duration 0.05 \
  --gravity
```

如果本机有图形界面并希望运行结束后弹出曲线：

```bash
python experiments/task00_torque_pulse.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml \
  --joint 4 \
  --torque 1.0 \
  --pulse-duration 0.05 \
  --show
```

## 预期输出

终端应首先打印：

```text
FR3 joint map
fr3_joint1 ... fr3_joint7
MuJoCo actuators found in model
Initial q
Initial qdot
Timestep
```

随后完成仿真并显示：

```text
Task00 result
Torque pulse target
Peak commanded tau
Max |q-q0|
PASS: simulation completed without NaN/Inf.
```

实验数据保存到：

```text
outputs/task00/torque_pulse.csv
outputs/task00/torque_pulse.png
```

其中 CSV 统一记录：

```text
time
q1 ... q7
qdot1 ... qdot7
tau1 ... tau7
```

## FR3 关节顺序

Task00 固定控制接口中的顺序为：

```text
0 -> fr3_joint1
1 -> fr3_joint2
2 -> fr3_joint3
3 -> fr3_joint4
4 -> fr3_joint5
5 -> fr3_joint6
6 -> fr3_joint7
```

也就是说，Python 的 `tau[3]` 对应 `fr3_joint4`。

## 验收

### A. 模型与状态

- 模型稳定加载，不出现 NaN / Inf；
- 能打印 `fr3_joint1` 到 `fr3_joint7`；
- `get_q()` 返回 shape `(7,)`；
- `get_qdot()` 返回 shape `(7,)`；
- 能读出 simulation timestep。

### B. Torque pulse

运行：

```bash
--joint 4 --torque 1.0
```

需要确认：

- `Peak commanded tau` 只有第 4 个关节约为 `1.0`；
- 其他 6 个关节的 command 为 `0`；
- `fr3_joint4` 的 `q` 和 `qdot` 对 pulse 有连续响应；
- 过程中没有 NaN / Inf。

注意：多关节机械臂存在动力学耦合，因此“只有 joint4 被施加 command torque”并不意味着其他关节的运动量绝对为零。Task00 验收的是 **command torque 只写入指定自由度**。

### C. 日志

确认存在：

```text
outputs/task00/torque_pulse.csv
outputs/task00/torque_pulse.png
```

曲线至少应包含：

- `q(t)`；
- `qdot(t)`；
- `tau(t)`。

### D. 概念问题

完成 Task00 前，你需要能解释：

> 为什么施加 1 N·m torque 后，关节角度不会瞬间跳到一个新的角度？

核心原因：torque 是动力学输入，它首先产生角加速度；速度和位置需要经过时间积分后才逐步变化。机械臂还同时受到惯量、阻尼、摩擦、重力以及关节间耦合的影响。

## 本 Task 不做

- 不做双臂；
- 不做末端控制；
- 不做接触；
- 不接 Isaac；
- 不做阻抗控制；
- 不做 QP。

完成上述验收后，再进入 Task01：Force / Torque / Wrench。
