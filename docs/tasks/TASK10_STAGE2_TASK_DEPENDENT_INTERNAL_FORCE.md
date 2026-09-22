# Task10 Stage 2 — Task-Dependent Internal Force Scheduling

## 为什么不固定指定 20 N

真实任务通常不是直接指定 internal force，而是要求物体不能滑、机械臂不能过载、接触不能超限，并保留安全裕度。

因此 Stage 2 根据当前 object wrench 自动生成 internal-compression reference。

## 调度公式

对于当前左右侧面对称抓取，接触法向为 ±X，切向方向为 Y/Z。

使用 Task09 的线性 friction pyramid：

~~~text
|Fy| + |Fz| <= mu Fn
~~~

若暂时假设左右接触平均承担物体切向载荷，则单侧最低需要：

~~~text
Fn,min = (|Fy,obj| + |Fz,obj|) / (2 mu)
~~~

加入安全系数：

~~~text
Fn,req = gamma * Fn,min
~~~

其中：

- Fn,min：刚好满足摩擦可行性的单侧最小法向力，单位 N；
- Fn,req：希望的单侧 internal-compression reference，单位 N；
- Fy,obj、Fz,obj：物体当前任务需要的 Y/Z 方向总力，单位 N；
- mu：摩擦系数；
- gamma >= 1：安全系数。

本实验使用：

~~~text
mu = 0.8
gamma = 1.5
Fn lower bound = 6.5 N
Fn upper bound = 30 N
~~~

因此理想对称情况下，摩擦利用率目标约为：

~~~text
1/gamma = 66.7 %
~~~

而不是 Task09 baseline 的 100%。

## 物体任务如何变化

物体质量 m=1 kg。

如果 WORLD-Y/Z 方向的期望加速度分别为 ay、az，则教学模型使用：

~~~text
Fy,obj = m ay
Fz,obj = m (g + az)
~~~

其中 ay、az 的单位为 m/s^2，g=9.81 m/s^2。

因此静止时只抗重力；向上加速时 Fz,obj 增大；向下加速时 Fz,obj 减小；横向 Y 加速也会提高切向载荷和夹紧需求。

## QP 如何使用 scheduler

每个任务状态先计算 f_task，再根据 scheduler 得到 Fn,req，构造：

~~~text
f_internal_des(Fn,req)
f_ref = f_task + f_internal_des
~~~

QP 继续满足硬约束：

~~~text
G f = W_obj_des
joint torque limits
unilateral contact
friction pyramid
contact moment limits
normal-force max
~~~

所以 scheduler 负责“根据任务建议夹多紧”，QP 负责“在真实约束下最终左右 wrench 应该是多少”。

## 实验场景

程序依次测试：

~~~text
downward_accel
static_hold
upward_accel
lateral_move
combined_accel
aggressive_combined
~~~

每个 phase 改变 ay/az，所以 object wrench 和 internal-force reference 自动变化。

## 这一步和动态反馈的区别

Stage 2 是：

~~~text
task state -> reference scheduling -> QP
~~~

它根据任务需求和模型参数 mu 计算 reference，不使用实测 internal force。

因此它属于 task-dependent feed-forward/reference scheduling，而不是 measured-force feedback closed loop。

## 完整运行命令

~~~bash
cd /home/ubuntu2004/lmy/dual-arm-force-control-mujoco-isaacsim/dual-arm-force-control-mujoco-isaacsim

git pull

source .venv/bin/activate

python3 -m pip install -r requirements.txt

python3 experiments/task10_task_dependent_internal_force.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml
~~~

输出：

~~~text
outputs/task10/stage2/task_dependent_internal_force.png
outputs/task10/stage2/task_dependent_internal_force.csv
outputs/task10/stage2/task10_stage2_geometry.xml
~~~


## 2026-09-22 实验结果

实验参数：

~~~text
m = 1.0 kg
mu = 0.8
gamma = 1.5
Fn allowed = 6.5 .. 30 N/contact
~~~

结果：

~~~text
scenario              Fy_obj  Fz_obj  Fn_min  Fn_sched  Fn_actual  friction[%]  torque[%]
downward_accel          0.00    7.81    4.88      7.32       7.32       66.7       29.9
static_hold             0.00    9.81    6.13      9.20       9.20       66.7       29.9
upward_accel            0.00   11.81    7.38     11.07      11.07       66.7       29.9
lateral_move            2.00    9.81    7.38     11.07      11.07       66.7       29.3
combined_accel          2.00   11.81    8.63     12.95      12.95       66.7       29.7
aggressive_combined     6.00   13.81   12.38     18.57      18.57       66.7       35.5
~~~

结论：

1. internal-force reference 已从固定人工设定改为 task-dependent scheduling；
2. 任务切向载荷增大时，所需 normal compression 自动增大；
3. QP realized compression 与 scheduler reference 一致，说明当前场景下 reference 可行；
4. friction utilization 基本保持 1/gamma = 66.7%，说明 safety factor 对摩擦裕度的含义得到直接验证；
5. aggressive task 下 joint-torque utilization 上升到约 35.5%，体现了更高抓持裕度需要更高 actuator effort；
6. reconstructed object force 与 task demand 一致，说明 internal-force scheduling 没有破坏 object-level wrench task。

Stage 2 状态：

~~~text
task-dependent scheduler : PASS
QP reference tracking    : PASS
friction margin shaping  : PASS
object wrench preservation: PASS
~~~

当前仍属于 feed-forward/reference scheduling，不包含真实 internal-force measurement feedback，也不处理 mu 估计误差、外部扰动、吸盘泄漏等不确定性；这些属于后续 robustness layer。
