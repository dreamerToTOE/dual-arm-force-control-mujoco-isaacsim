# Learning Guide — 力控学习门禁

本项目按“理解概念 → 最小实验 → 数值验收 → 再进入下一层”的方式推进。代码能跑不等于完成 Task；如果不能解释变量和公式的物理意义，应先回到当前阶段继续学习。

## Phase A：基础动力学与力矩控制（Task00–04）

必须能解释：

- `q`：关节位置；
- `qdot`：关节速度；
- `tau`：关节广义力 / 力矩；
- `M(q)`：质量矩阵；
- `C(q,qdot)qdot`：科氏/离心项；
- `g(q)`：重力项；
- `J(q)`：关节速度到末端 twist 的映射；
- 为什么末端 wrench 通过 `J^T` 映射成关节力矩。

完成本阶段后，应能独立说明：

```text
xdot = J(q) qdot

tau_ext = J(q)^T W
```

分别在描述什么。

## Phase B：末端阻抗与接触（Task05–07）

必须能区分：

- 位置控制：强调“到哪里”；
- 力控制：强调“施加多大接触力”；
- 阻抗控制：规定位置误差和作用力之间的动态关系；
- Hybrid position/force：在不同任务方向上分别控制位置和力。

进入 Task08 前，至少应能回答：

1. 为什么刚性位置控制在接触任务中可能产生很大接触力？
2. `K` 大和小分别意味着什么？
3. `D` 的作用是什么？
4. 为什么位置和力不能在同一受约束方向上完全独立指定？

## Phase C：双臂共同物体（Task08–10）

这里第一次引入双机械臂协作。

需要理解：

```text
Left wrench  ─┐
              ├─> object resultant wrench
Right wrench ─┘
```

以及为什么同一个物体目标 wrench 可能对应多组左右臂分力。

进入 QP 前应能说明：

- object wrench 是什么；
- grasp matrix `G` 的作用；
- 为什么 `G f = W_object` 一般不是唯一解；
- 为什么要引入力/力矩上限；
- 为什么还要控制 internal force。

## Phase D：鲁棒性与跨仿真器（Task11–13）

这一阶段不再只验证“理想模型能不能跑”。需要主动改变：

- payload mass；
- friction；
- contact parameters；
- controller gains；
- external disturbance；
- model error。

随后保持 Controller Core 不变，只替换 `MuJoCoAdapter` 为 `IsaacAdapter`。

跨仿真器不要求逐采样点结果完全一致；重点比较：

- 稳定性；
- tracking error；
- force error；
- torque peak；
- settling time；
- overshoot；
- internal-force behavior。

## 学习原则

如果某个 Task 出现问题，优先问：

```text
我是否理解这个量是什么？
↓
公式的输入输出单位是否正确？
↓
坐标系是否一致？
↓
仿真接口是否正确？
↓
最后才调参数
```

不要通过无限增大增益来掩盖模型、坐标系或公式错误。
