# Task05 — Cartesian Impedance Control

## 学习目标

让末端呈现“虚拟弹簧 + 阻尼器”的动态行为，并理解阻抗控制和刚性位置控制的区别。

本 Task 第一阶段只做 **3D 平移阻抗**。姿态阻抗暂不混在第一版里，避免同时引入旋转误差表示、四元数/旋转矩阵误差等新概念。

## 核心思想

在 BASE frame 中：

```text
e_p = p_des - p
e_v = v_des - v
F_cmd = Kx e_p + Dx e_v
tau_task = Jv^T F_cmd
```

其中：

- `Kx`：笛卡尔平移刚度，单位 `N/m`；
- `Dx`：笛卡尔平移阻尼，单位 `N*s/m`；
- `Jv`：完整 Jacobian 的前三行，即平移 Jacobian；
- `F_cmd`：虚拟弹簧 + 阻尼器产生的末端恢复力；
- `tau_task`：通过 Jacobian transpose 映射得到的关节力矩。

实际仿真使用：

```text
tau_total = Jv^T F_cmd + g(q) + tau_external
```

其中外部扰动力 `F_ext` 通过：

```text
tau_external = Jv^T F_ext
```

等效施加到 TCP。

这仍然不是接触力闭环控制；Task06 才会加入真实接触和力反馈。

## 当前实现

控制器核心：

```text
controllers/cartesian_impedance.py
```

控制器本身与 MuJoCo 解耦，只完成：

```text
F_cmd = Kx (p_des-p) + Dx (v_des-v)
```

实验：

```text
experiments/task05_cartesian_impedance.py
```

默认 HOME TCP 位置作为 `p_des`，并在：

```text
t = [1.0, 3.0) s
```

给 TCP 施加：

```text
F_ext = [8, 0, 0] N
```

即 BASE +X 方向的外力脉冲。

程序自动完成两组对比：

### 1. Stiffness sweep

```text
soft : Kx = 120 N/m
hard : Kx = 400 N/m
```

观察相同外力下，软弹簧与硬弹簧产生的 TCP 位移差异。

理想静态关系可粗略写为：

```text
|dx| ≈ |F_ext| / Kx
```

因此刚度越高，同样外力下位移越小；但“位移小”不等于任何任务里都更好，阻抗控制恰恰常常需要有意保留柔顺性。

### 2. Damping sweep

```text
low_damping : Kx = 250 N/m, Dx = 5 N*s/m
damped      : Kx = 250 N/m, Dx = 35 N*s/m
```

观察：

- 阻尼太小时的振荡/超调；
- 阻尼增加后更平稳的恢复；
- 阻尼过大时可能造成迟缓。

## 运行

```bash
python3 experiments/task05_cartesian_impedance.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml
```

带 MuJoCo Viewer：

```bash
python3 experiments/task05_cartesian_impedance.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml \
  --viewer
```

Viewer 默认显示 `damped` case：

```text
绿色球   = desired TCP position
黄色球   = actual TCP position
红色箭头 = external disturbance force
橙色箭头 = impedance restoring force F_cmd
```

整个扰动—恢复过程会循环播放。

## 输出

```text
outputs/task05/stiffness_comparison.png
outputs/task05/damping_comparison.png
outputs/task05/soft.csv
outputs/task05/hard.csv
outputs/task05/low_damping.csv
outputs/task05/damped.csv
```

终端自动输出：

- peak Cartesian position error；
- 外力撤去前的 TCP X 位移；
- 理想 `F/K` 位移；
- 外力撤去后的 5 mm recovery time；
- 最大姿态漂移。

## 为什么第一版会记录 orientation drift

第一阶段控制的是：

```text
p in R^3
```

而不是完整 6D pose。

因此：

```text
F_cmd = [Fx,Fy,Fz]
```

没有主动生成末端姿态恢复力矩。机器人具有 7 个关节自由度，保持 TCP 位置并不等价于保持 TCP 姿态，所以姿态可能发生漂移。

这不是第一版控制器的 bug，而是“只控制 3D 平移”的自然结果。后续扩展 6D impedance 时再显式加入姿态误差和旋转刚度/阻尼。

## 验收

- 四组仿真均无 NaN / Inf；
- TCP 在外力作用下产生可见位移，撤去外力后能向目标恢复；
- 相同 `F_ext` 下，高 `Kx` 的位移明显小于低 `Kx`；
- 相同 `Kx` 下，提高 `Dx` 能明显改变振荡和恢复过程；
- 能解释近似静态关系 `F ≈ Kx * dx`；
- 能解释为什么高刚度不等于“更好的阻抗控制”；
- 能解释为什么只做 3D 平移阻抗时，TCP orientation 不受约束；
- 能把 Task04 的 `J^T W` 与本 Task 的虚拟弹簧/阻尼联系起来。

## 教学门禁

完成后应能用自己的话解释完整链路：

```text
TCP position/velocity error
        ↓
virtual Cartesian spring + damper
        ↓
F_cmd
        ↓
Jv^T F_cmd
        ↓
joint torque
```

并能说明阻抗控制的目标不是“无论外力多大都死死保持位置”，而是规定“外力—位移/速度”之间的动态关系。
