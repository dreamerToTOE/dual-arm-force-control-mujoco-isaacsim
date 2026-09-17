# Task05 — Cartesian Impedance Control

## 学习目标

让末端呈现“虚拟弹簧 + 阻尼器”的动态行为，并理解阻抗控制和刚性位置控制的区别。

Task05 分两阶段：

1. 先做 **3D 平移阻抗**，理解笛卡尔刚度 `Kx` 与阻尼 `Dx`；
2. 再扩展为 **6D 位姿阻抗**，加入姿态误差、角速度误差和旋转刚度/阻尼。

---

## Stage 1 — 3D Translational Impedance

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

实际仿真：

```text
tau_total = Jv^T F_cmd + g(q) + tau_external
```

外部扰动力通过：

```text
tau_external = Jv^T F_ext
```

等效施加到 TCP。

### Stage-1 实验

```text
experiments/task05_cartesian_impedance.py
```

默认 HOME TCP 位置作为 `p_des`，在：

```text
t = [1.0, 3.0) s
```

施加：

```text
F_ext = [8, 0, 0] N
```

程序比较：

```text
soft : Kx = 120 N/m
hard : Kx = 400 N/m

low_damping : Kx = 250 N/m, Dx = 5 N*s/m
damped      : Kx = 250 N/m, Dx = 35 N*s/m
```

理想静态关系可粗略写为：

```text
|dx| ≈ |F_ext| / Kx
```

但真实 FR3 还存在 joint frictionloss、damping、耦合、冗余自由度等，因此实际位移不必严格等于 `F/K`。

第一阶段只控制 `p in R^3`，不控制 TCP orientation，因此允许姿态漂移。

---

## Stage 2 — Full 6D Cartesian Impedance

Stage 2 在平移弹簧/阻尼之外增加姿态弹簧/阻尼。

### 姿态误差

本项目不直接对旋转矩阵或四元数做普通减法，而是使用 SO(3) 旋转误差：

```text
e_R = Log(R_des R^T)^vee
```

`e_R in R^3` 是 BASE frame 下的 rotation vector，可理解为“从当前 TCP 姿态旋到目标姿态所需的小旋转轴 × 角度”。

角速度误差：

```text
e_omega = omega_des - omega
```

旋转阻抗：

```text
M_cmd = Kr e_R + Dr e_omega
```

其中：

- `Kr`：旋转刚度，单位 `N*m/rad`；
- `Dr`：旋转阻尼，单位 `N*m*s/rad`；
- `M_cmd`：末端虚拟恢复力矩。

完整 6D wrench：

```text
W_cmd = [F_cmd, M_cmd]
```

再使用 Task04 的完整 Jacobian transpose：

```text
tau_task = J^T W_cmd
```

最终：

```text
tau_total = J^T (W_cmd + W_ext) + g(q)
```

### Stage-2 实验

```text
experiments/task05_cartesian_impedance_6d.py
```

同一个外部 wrench 同时包含：

```text
F_ext = [6, 0, 0] N
M_ext = [0, 1.2, 0] N*m
```

比较：

```text
translation_only
vs
full_6d
```

两组使用相同的平移增益：

```text
Kt = 250 N/m
Dt = 35 N*s/m
```

`full_6d` 额外使用：

```text
Kr = 20 N*m/rad
Dr = 4 N*m*s/rad
```

实验目的不是证明某个参数“最好”，而是验证：

> 在相同外部转矩扰动下，translation-only 不会主动恢复姿态，而 full-6D impedance 会通过 `M_cmd` 抵抗并恢复姿态。

### 运行

```bash
python3 experiments/task05_cartesian_impedance_6d.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml \
  --viewer
```

输出：

```text
outputs/task05/pose_impedance_3d_vs_6d.png
outputs/task05/translation_only_6d_stage.csv
outputs/task05/full_6d_6d_stage.csv
```

Viewer：

```text
绿色球   = desired TCP position
黄色球   = actual TCP position
RGB 箭头 = current TCP axes
红色箭头 = external force
橙色箭头 = impedance restoring force
紫色箭头 = external moment axis
青色箭头 = impedance restoring moment axis
```

---

## Task05 与 Task02 / Task04 的联系

Task02：

```text
joint error -> joint PD -> joint torque
```

Task04：

```text
Cartesian wrench -> J^T -> joint torque
```

Task05：

```text
Cartesian pose/velocity error
        ↓
virtual Cartesian spring + damper
        ↓
W_cmd
        ↓
J^T W_cmd
        ↓
joint torque
```

因此阻抗控制的目标不是“无论外力多大都死死保持位置”，而是规定：

> 外力/外力矩作用下，末端应该表现出怎样的位移、姿态偏转、速度和恢复特性。

---

## 验收

Stage 1：

- TCP 在外力作用下产生可见位移；
- 高 `Kx` 的位移明显小于低 `Kx`；
- 提高 `Dx` 能明显改变振荡与速度峰值；
- 能解释 `F ≈ Kx * dx` 的理想含义与实际偏差；
- 能解释 3D 平移阻抗为什么不约束 orientation。

Stage 2：

- translation-only 与 full-6D 均无 NaN/Inf；
- 在相同外部 moment 下，full-6D 的 orientation drift 明显小于 translation-only；
- 能解释 `e_R` 为什么不能用普通矩阵/四元数直接相减；
- 能解释 `Kr` 与 `Dr` 分别对应“旋转弹簧”和“旋转阻尼”；
- 能把完整链路 `pose error -> W_cmd -> J^T W_cmd -> tau` 说清楚。
