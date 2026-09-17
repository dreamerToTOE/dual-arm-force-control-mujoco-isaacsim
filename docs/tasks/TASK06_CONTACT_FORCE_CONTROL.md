# Task06 — Contact Force Control

## 学习目标

第一次让 FR3 **真正接触环境**，并根据测得的接触法向力形成闭环，而不是继续把“向下压一点位置”当作力控制。

Task06 分两阶段：

1. Stage 1：同一接触面上比较 P 与 PI 法向力控制；
2. Stage 2：在不同接触刚度/阻尼下比较同一力控制器的响应。

当前先完成 Stage 1。

## 场景

```text
FR3 TCP spherical probe
          ↓  -BASE-Z push
------------------------------  contact plane
          ↑  measured normal force
```

脚本会在 FR3 的 `attachment_site` 位置自动加入一个小球形接触 probe，并在 HOME TCP 下方约 15 mm 处生成一块水平接触平面。

为保证实验只研究指定接触，Task06 生成的临时教学模型会关闭原始 link collision geoms，仅保留 TCP probe 与 contact plane 的碰撞。

## 为什么这次才叫真正的 force control

Task05 中虽然有外力，但控制器并没有测量环境反力后去调节力。

Task06 的闭环是：

```text
F_des
  ↓
force error e_F = F_des - F_meas
  ↓
P / PI force controller
  ↓
commanded push force
  ↓
J^T mapping
  ↓
joint torque
  ↓
robot/environment contact
  ↓
MuJoCo measured F_meas
  └──────────────────── feedback
```

也就是说，本 Task 的核心新东西是 **contact-force feedback**。

## 接触力读取

MuJoCo 中遍历 `data.contact`，只筛选：

```text
task06_probe <-> task06_surface
```

然后使用：

```python
mujoco.mj_contactForce(...)
```

读取 contact-frame normal component，并取其正向大小作为：

```text
F_meas >= 0
```

第一阶段只控制法向力大小，不控制切向摩擦力。

## Stage 1：P vs PI

目标法向力：

```text
F_des = 10 N
```

### P

```text
u = Kp (F_des - F_meas)
```

只依赖当前力误差。和位置 P 控制类似，存在稳态误差的可能。

### PI

```text
u = Kp (F_des - F_meas)
  + Ki integral(F_des - F_meas) dt
```

积分项会持续累积残余力误差，因此理论上能明显减小 steady-state error。

当前 Stage 1 使用：

```text
P : Kp=1.5, Ki=0
PI: Kp=1.5, Ki=1.8
```

并带有：

- push-force saturation；
- integral limit；
- simple anti-windup。

控制器文件：

```text
controllers/force_pi.py
```

## 接触前为什么还需要 approach phase

一开始 TCP 与平面之间有约 15 mm 间隙，此时：

```text
F_meas = 0
```

力控制器本身并不知道“平面在哪里”。

因此先通过一个低速/柔顺的 Cartesian impedance approach 把 TCP 送向平面。

当检测到：

```text
F_meas >= 0.5 N
```

才认为 contact established，并切换到法向力闭环。

这也是实际机器人力控中很常见的逻辑：

```text
approach -> contact detection -> force regulation
```

## 接触后的控制结构

进入 force-control 阶段后：

- BASE X/Y 方向继续用 Cartesian impedance 稳住 TCP；
- TCP orientation 继续用 rotational impedance 稳定；
- Z 方向不再用 position spring，而改用 force controller；
- force command 沿 `-BASE-Z` 推向表面。

力命令最终通过：

```text
tau_force = J^T W_force
```

转成 joint torque，并加：

```text
+ g(q)
```

做重力补偿。

这里暂时不要把它当成 Task07 的“正式 hybrid position/force control”。Task06 的关注点只有：**接触法向力闭环本身**。Task07 再系统引入选择矩阵和位置/力方向分工。

## 运行

```bash
python3 experiments/task06_contact_force_control.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml
```

带 Viewer：

```bash
python3 experiments/task06_contact_force_control.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml \
  --viewer
```

Viewer 默认重放 PI case：

```text
黄色球 = physical TCP contact probe
红色箭头 = commanded push force (-BASE-Z)
青色箭头 = measured contact normal force (+BASE-Z)
```

## 输出

```text
outputs/task06/p_vs_pi_force_control.png
outputs/task06/p.csv
outputs/task06/pi.csv
outputs/task06/task06_contact_scene.xml
```

终端指标：

- first-contact time；
- 90% force rise time；
- force overshoot；
- steady-state force error；
- peak contact force；
- peak joint torque。

## Stage 1 验收

- TCP 能够建立真实 MuJoCo contact；
- `mj_contactForce` 能读到连续的法向接触力；
- P 和 PI 两种控制均无 NaN / Inf；
- PI 的 steady-state force error 应明显小于 P；
- 不出现持续高频接触振荡或过大的峰值力；
- 能解释 P 与 PI force loop 的区别；
- 能解释为什么“给一个更低的位置目标继续往下压”不等于真正的 force control。

## Stage 2（Stage 1 完成后）

在相同 PI controller 下改变 MuJoCo contact `solref/solimp`，比较：

- softer surface；
- firmer surface。

重点观察：

- force rise time；
- overshoot；
- oscillation；
- controller gains 为什么不能脱离环境接触特性单独调。

## 教学门禁

完成 Task06 后应能用自己的话解释：

> position control 关注“末端在哪里”；force control 关注“环境实际反作用力是多少”。真正的 force control 必须测量/估计接触力，并把它反馈到控制器中形成闭环。
