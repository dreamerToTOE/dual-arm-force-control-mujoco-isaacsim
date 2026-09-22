# Task11：力控鲁棒性与扰动实验

## 当前项目决策

2026-09-22：本阶段暂缓，不删除、不判失败。当前优先完成 Task12--14：Isaac Sim Adapter、跨仿真验证和码垛整合。鲁棒性批量实验保留为后续论文/工程增强项。

## Stage 1：Nominal Model vs True Physics

Task10 在已知质量、已知摩擦系数的情况下，根据 task wrench 自动调节 internal force。

Task11 开始故意让“控制器认为的模型”和“真实系统”不同。

本阶段首先区分两类鲁棒性问题：

~~~text
grasp robustness
    -> 摩擦系数估错会不会打滑？

object-task robustness
    -> 质量估错 / 外部扰动会不会导致物体实际运动偏离目标？
~~~

这两类问题不能靠同一种办法解决。

## 1. Nominal model 与 true model

控制器内部使用：

~~~text
m_model = 1.0 kg
~~~

并比较两种 internal-force scheduling policy：

~~~text
nominal_mu:
    mu_plan = 0.80

conservative_mu:
    mu_plan = 0.50
~~~

其中：

- m_model：控制器认为的物体质量；
- mu_plan：控制器规划/约束时使用的摩擦系数。

真实测试场景则独立指定：

- true mass；
- true friction coefficient；
- unmodeled external force。

因此 controller model 与 true physics 可以不一致。

## 2. 真正需要的 object wrench

对于真实质量 m_true，期望 WORLD-Y/Z 加速度 ay、az，以及未知外力 Fext,y、Fext,z，机器人真正应该提供：

~~~text
Frobot,y,true = m_true ay - Fext,y

Frobot,z,true = m_true (g + az) - Fext,z
~~~

其中：

- m_true：物体真实质量，单位 kg；
- ay、az：希望实现的物体加速度，单位 m/s²；
- g：重力加速度，9.81 m/s²；
- Fext：环境施加在物体上的外力，单位 N。

如果控制器仍然按照错误的 m_model 或忽略 Fext，则：

~~~text
W_robot,cmd != W_robot,true_required
~~~

即使 internal force 很大，也不能让这个 object-level wrench error 自动消失。

## 3. True friction utilization

假设左右两接触平均承担真实任务切向载荷，则按照真实摩擦系数评估：

~~~text
rho_true =
(|Fy,true| + |Fz,true|)
/
(2 mu_true Fn_actual)
~~~

其中：

- rho_true：真实摩擦利用率；
- Fy,true、Fz,true：真实任务需要机器人提供的切向力；
- mu_true：真实摩擦系数；
- Fn_actual：QP 实际生成的单侧 internal compression。

判定：

~~~text
rho_true <= 1
    -> 当前夹紧能力足够

rho_true > 1
    -> slip risk
~~~

注意：QP 自己检查的是 mu_plan；Task11 额外用 mu_true 做“事后真值审计”。

## 4. 为什么 conservative friction policy 有意义

Task10 scheduler：

~~~text
Fn_req =
gamma (|Fy_model| + |Fz_model|)
/
(2 mu_plan)
~~~

其中 gamma 是安全系数。

如果使用更小、更保守的 mu_plan：

~~~text
mu_plan down
    -> Fn_req up
    -> friction capacity up
~~~

因此它能够提高低摩擦环境下的抓取鲁棒性。

代价是：

~~~text
internal force up
    -> joint torque / contact pressure up
~~~

## 5. 为什么它解决不了质量误差

internal wrench 满足：

~~~text
G f_internal = 0
~~~

所以增加 internal force 本身不会增加物体净支撑 wrench。

如果实际物体从：

~~~text
1.0 kg -> 1.3 kg
~~~

但控制器仍只给出 1.0 kg 所需的 object wrench，那么：

~~~text
support force is insufficient
~~~

即使两只手夹得非常紧，箱子整体仍可能下坠。

因此：

> grip robustness 和 object-wrench tracking robustness 是两个不同问题。

前者可以通过安全裕度、保守 friction model、contact constraints 改善。

后者最终需要 object pose/velocity/force feedback、payload estimation 或 disturbance observer 等机制。

## 6. 批量场景

Stage 1 一次运行：

~~~text
nominal
low_friction
heavy_payload
heavy_low_mu
lateral_push
combined_mismatch
~~~

每个场景同时由两种 policy 测试。

记录：

- true friction utilization；
- actual internal compression；
- implied object acceleration error；
- maximum joint-torque utilization；
- grip OK / slip risk；
- task tracking OK / task error。

## 7. 完整运行命令

~~~bash
cd /home/ubuntu2004/lmy/dual-arm-force-control-mujoco-isaacsim/dual-arm-force-control-mujoco-isaacsim

git pull

source .venv/bin/activate

python3 -m pip install -r requirements.txt

python3 experiments/task11_robustness_sweep.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml
~~~

输出：

~~~text
outputs/task11/stage1/robustness_model_mismatch.png
outputs/task11/stage1/robustness_model_mismatch.csv
outputs/task11/stage1/task11_stage1_geometry.xml
~~~

## 8. Stage 1 教学目标

完成后应能解释：

~~~text
摩擦模型误差
    -> 主要影响抓取可行性 / slip margin

质量误差 / 未知外力
    -> 主要影响 object wrench correctness
~~~

并理解：

> “夹得更紧”只能提高抓取裕度，不能代替物体层的反馈与扰动补偿。

后续 Stage 2 再扩展 sensor noise、control rate、contact stiffness / damping 等动态鲁棒性因素。
