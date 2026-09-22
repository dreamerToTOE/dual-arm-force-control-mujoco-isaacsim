# Task12：Isaac Sim / Isaac Lab Adapter 移植

## 学习目标

证明控制器核心不依赖 MuJoCo。此阶段不修改控制公式，只更换 simulator adapter。

目标架构：

```text
MuJoCoAdapter ─┐
               ├─ Controller Core -> tau
IsaacAdapter ──┘
```

## 统一接口

Isaac 侧实现与 MuJoCo 一致的：

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

## 移植顺序

1. FR3 joint ordering 对齐；
2. base/TCP frame 对齐；
3. torque sign/unit 对齐；
4. Jacobian 对齐；
5. mass matrix / gravity 对齐；
6. 先跑 Task03 joint controller；
7. 再跑 Task05 impedance；
8. 最后跑 Task08--10 双臂控制。

## 禁止事项

- 不为了 Isaac 单独复制一套控制器；
- 不在 controller 内写 PhysX API；
- 不因为结果不同就直接改公式，先检查模型、坐标系、频率与接触参数。

## 验收

同一个 Controller Core 在两边均可运行，只通过 Adapter 提供数据和输出 torque。


## 当前实施顺序

Task11 暂缓后，Task12 成为当前主线。

为了避免直接猜 Isaac USD 中的 joint/body/prim ordering，Task12 分成：

```text
Stage 0: scene inventory
Stage 1: IsaacAdapter state/dynamics readout
Stage 2: torque-output smoke test
Stage 3: migrate Task03 joint controller
Stage 4: migrate Task05 Cartesian impedance
Stage 5: prepare dual-arm Task08--10 migration
```

### Stage 0 为什么必须先做

MuJoCo 里我们明确知道：

```text
joint names : fr3_joint1 .. fr3_joint7
TCP         : attachment_site
base        : base
```

但 Isaac USD 的：

- articulation-root prim path；
- rigid-body ordering；
- Jacobian body-row ordering；
- TCP body/prim；
- finger DOF 是否包含在 articulation；

都必须从实际 USD 读取，不能凭名字猜。

Isaac Sim 4.5 的 fixed-base articulation Jacobian shape 为：

```text
(num_bodies - 1, 6, num_dof)
```

因此 TCP 对应哪一行取决于真实 body ordering。

### Stage 0 脚本

```text
experiments/task12_isaac_inventory.py
```

功能：

1. 打开已有 Isaac USD scene；
2. 列出所有 articulation roots；
3. 选择一个 FR3 root 后打印 DOF ordering；
4. 打印 rigid-body ordering；
5. 检查 fr3_joint1..7 是否存在；
6. 输出 Jacobian / mass-matrix shape；
7. 给出可能的 TCP body 候选。

完成这个 inventory 后，再写正式 IsaacAdapter，避免把错误 prim/body index 固化进接口。
