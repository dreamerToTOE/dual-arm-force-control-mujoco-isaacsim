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


## 2026-09-21 第一轮场景初始化问题

第一轮出现明显异常：

```text
desired object lift  = +40 mm
actual object travel = -275.207 mm
object RMS error     = 314.077 mm
relative grasp RMS   = 929.999 mm
```

Viewer 中左右末端也明显没有保持在 SharedBox 两侧。

该结果不是控制器性能问题，而是第一版 weld equality 初始化错误，因此本轮数据作废。

### 根因

第一版使用 body-based weld：

```text
shared_box <-> left_fr3_link7
shared_box <-> right_fr3_link7
```

但没有显式给出 `relpose`。

MuJoCo 此时会按模型参考构型 `qpos0` 记录 body-body 相对位姿；而双 FR3 在模型加载后又被程序设置到：

```text
HOME_Q = [0, 0, 0, -pi/2, 0, pi/2, -pi/4]
```

于是仿真开始时，实际 HOME 构型与 weld 在 `qpos0` 保存的参考关系严重不一致，constraint solver 会立即强行拉动两臂和 SharedBox，造成场景“炸开”。

### 修复

当前版本先生成一个无 weld 的临时双臂模型：

```text
load dual FR3
-> set both arms to HOME_Q
-> forward kinematics
-> compute shared_box -> left_link7 relative pose
-> compute shared_box -> right_link7 relative pose
-> write both explicit weld relpose
-> reload final scene
```

因此最终 weld 约束在 HOME 构型下从一开始就是满足的，不再依赖错误的 `qpos0` 相对关系。

控制参数没有因为这一问题而调节。


## 2026-09-21 修复后实验结果

修复 weld relpose 初始化后：

```text
Object RMS pos error   = 11.504 mm
Object max pos error   = 14.985 mm
Actual object Z travel = 26.940 mm / 40 mm desired

Left TCP RMS error     = 11.498 mm
Right TCP RMS error    = 11.498 mm

Relative grasp RMS err = 0.2601 mm
Relative grasp max err = 0.3335 mm
Object max ori error   = 0.0001 deg

Left peak |F_cmd|      = 11.657 N
Right peak |F_cmd|     = 11.657 N
Left peak |tau|        = 32.721 N m
Right peak |tau|       = 32.721 N m
```

结论：

- weld 初始化问题已修复；
- 双臂与 SharedBox 的相对抓持几何能够稳定保持；
- 左右臂响应高度对称；
- object-level reference 能够生成一致的左右 TCP reference；
- 简单的独立 Cartesian impedance baseline 仍存在约 11.5 mm 量级的 object tracking error，且 40 mm 目标抬升只实现约 26.9 mm；
- Task08 不继续以反复调参为目标，后续重点转向 Task09 wrench distribution。

运行结束后若只在 MuJoCo Viewer 关闭阶段出现 native segmentation fault，但 CSV/plot 和数值仿真已经完整输出，应先用不带 `--viewer` 的同一命令隔离确认。若无 Viewer 时不再崩溃，则该问题单独归类为 Viewer/GLFW shutdown 问题，不把它混同为控制仿真失败。
