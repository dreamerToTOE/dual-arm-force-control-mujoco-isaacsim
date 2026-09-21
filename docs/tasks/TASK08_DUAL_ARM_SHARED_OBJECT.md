# Task08 — 双 FR3 共同物体基线

## 目标

Task08 是项目第一次真正进入双机械臂。

这一关**不做 QP、不做最优 wrench distribution、不做 internal-force controller**。核心问题只有：

> 如何由一个共同物体的期望运动，同时生成左右机械臂一致的 TCP 目标，并让两个单臂控制器共同带动物体。

场景：

```text
Left FR3  ==== SharedBox ====  Right FR3
```

其中 `====` 在第一版中使用 MuJoCo weld equality 表示**理想刚性抓持**。

## 1. 为什么先用理想 weld

如果 Task08 一开始同时加入：

- 真实夹持接触；
- 摩擦锥；
- 抓取稳定性；
- wrench allocation；
- internal force；
- QP；

就无法判断问题到底出在哪一层。

因此第一版先假定“已经抓牢”，只研究**双臂共同物体运动学与控制基线**。

## 2. 双 FR3 场景

脚本直接读取用户现有：

```text
~/mujoco_menagerie/franka_fr3/fr3.xml
```

然后运行时自动复制为：

```text
left_fr3_*
right_fr3_*
```

两台机器人放在物体左右两侧，右臂基座绕 WORLD-Z 旋转 180°，形成面对面的布局。

原始 FR3 position actuators 不复制；两臂继续通过：

```text
qfrc_applied
```

接收 torque command。

## 3. SharedBox 与闭链

SharedBox 是一个带 freejoint 的刚体：

```text
mass = 1.0 kg
```

它分别与：

```text
left_fr3_link7
right_fr3_link7
```

建立 weld equality。

因此系统成为真正的闭链：

```text
left arm -> object -> right arm -> world
```

Task08 暂时关闭机器人/物体物理碰撞，避免把 collision/contact grasp 问题混入这一关。

## 4. 一个 object reference 生成两个 TCP reference

初始时记录物体到左右 TCP 的固定变换：

```text
T_object_left
T_object_right
```

当给定：

```text
T_world_object_des
```

则：

```text
T_world_left_des  = T_world_object_des * T_object_left
T_world_right_des = T_world_object_des * T_object_right
```

这是 Task08 最重要的运动学关系。

本实验只让物体：

```text
+WORLD-Z lift 40 mm
```

不旋转，避免一次加入太多概念。

## 5. 控制基线

左右臂分别使用已有的 6D Cartesian impedance：

```text
W_L = impedance(T_L_des - T_L)
W_R = impedance(T_R_des - T_R)
```

然后：

```text
tau_L = J_L^T W_L + g_L
tau_R = J_R^T W_R + g_R
```

SharedBox 重量先手工做最简单的 50/50 nominal load split：

```text
Fz_L += m g / 2
Fz_R += m g / 2
```

这只是 baseline，不是“最优分配”。

Task09 会把这个手工 50/50 换成正式的 wrench allocation / QP。

## 6. 必须理解的区别

Task08 中“两臂都跟踪由同一个 object reference 生成的目标”，仍然不等于完成了双臂受力协调。

因为它没有回答：

- 左臂应该承担多少力？
- 右臂应该承担多少力？
- 某个关节接近 torque limit 时应该怎么重新分配？
- 两臂互相顶住产生的 internal force 怎么控制？

这些分别是 Task09 / Task10 的主题。

## 7. 输出指标

程序记录：

- SharedBox position error；
- SharedBox orientation error；
- 左右 TCP tracking error；
- 左右 TCP relative grasp geometry error；
- 左右 commanded wrench；
- 左右 joint torque。

注意：

> 当前记录的 `W_L/W_R` 是 controller-commanded wrench，不是 weld equality 的实测约束 wrench。

## 8. 运行

```bash
python3 experiments/task08_dual_arm_shared_object.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml \
  --viewer
```

## 9. 可视化

Viewer：

```text
橙色 box  = SharedBox
绿色球    = desired object center
左右两台 FR3 = rigidly welded to SharedBox
```

输出：

```text
outputs/task08/task08_dual_fr3_shared_object.xml
outputs/task08/dual_arm_shared_object.png
outputs/task08/dual_arm_shared_object.csv
```

## 10. Task08 的教学门禁

完成后应该能够解释：

> 双臂共同物体控制首先需要一个 object-level reference，再由固定 grasp transforms 生成左右 TCP reference。

同时明确：

> 两个独立 Cartesian controller 能形成一个 baseline，但它不能自动解决 wrench distribution 和 internal force。

这正是 Task09 与 Task10 存在的原因。
