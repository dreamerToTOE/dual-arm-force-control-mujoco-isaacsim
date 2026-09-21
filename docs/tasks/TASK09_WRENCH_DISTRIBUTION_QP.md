# Task09 — 双臂 Wrench Distribution 与 QP

## 学习目标

Task08 已经建立：

```text
一个 object reference
        ↓
左右 TCP reference
        ↓
两臂共同带动物体
```

但 Task08 的 SharedBox 重量只是手工规定：

```text
左臂 50 %
右臂 50 %
```

Task09 第一次正式回答：

> 给定物体所需的总 wrench，左右机械臂末端各应该承担多少 wrench？

Stage 1 先研究**静态分配问题**，不把 QP 立即塞进动态搬运控制器。

---

## 1. Wrench 与决策变量

单臂末端 wrench：

```text
W_i = [Fx, Fy, Fz, Mx, My, Mz]^T
```

其中：

- `F_x,F_y,F_z`：末端施加的三维力，单位 N；
- `M_x,M_y,M_z`：末端施加的三维力矩，单位 N·m；
- 下标 `i=L,R`：分别表示左臂和右臂。

QP 的决策变量定义为：

```text
f = [W_L; W_R] in R^12
```

也就是一次同时求出左右两个 6D wrench。

---

## 2. 单个抓取点如何对物体产生 wrench

定义：

```text
r_i = p_i - p_O
```

其中：

- `p_i in R^3`：第 i 个抓取点的位置；
- `p_O in R^3`：物体中心位置；
- `r_i in R^3`：从物体中心指向抓取点的位置向量，单位 m。

如果抓取点施加：

```text
W_i = [F_i; M_i]
```

那么对物体中心产生：

```text
F_O = F_i
M_O = r_i x F_i + M_i
```

因此：

```text
W_O = G_i W_i
```

其中：

```text
G_i = [ I        0
        [r_i]x   I ]
```

- `G_i in R^(6x6)`：单个抓取点的 wrench 映射；
- `[r_i]x`：r_i 的反对称叉乘矩阵，满足 `[r_i]x F = r_i x F`；
- `W_O`：该抓取点对物体中心产生的 6D wrench。

---

## 3. 双臂 Grasp Matrix

把左右两个抓取映射拼起来：

```text
G = [G_L  G_R]
```

其中：

- `G in R^(6x12)`：双臂 grasp matrix；
- `G_L,G_R in R^(6x6)`：左右抓取点各自的 wrench map。

于是：

```text
G f = W_O_des
```

其中：

- `f in R^12`：待求的左右末端 wrench；
- `W_O_des in R^6`：物体期望得到的总 wrench。

这是 Task09 最核心的等式约束。

由于：

```text
6 equations
12 unknown wrench components
```

通常存在很多组不同的左右 wrench 都能产生同一个物体总 wrench。

---

## 4. 为什么需要优化

最简单的分配可以写为：

```text
min_f  1/2 f^T H f
s.t.   G f = W_O_des
```

其中：

- `H in R^(12x12)`：正定权重矩阵；
- 目标函数用于在所有满足物体任务的解里选择“代价较小”的一组；
- 当前实验对 force 与 moment 做尺度归一化，避免直接把 N 和 N·m 当成完全同量纲数值比较。

对称抓取 + 对称权重 + 仅静止承重时，最小代价解应该接近左右平均承担重量。

---

## 5. 加入关节力矩约束

第 i 条机械臂的关节力矩近似写成：

```text
tau_i = J_i^T W_i + g_i
```

其中：

- `tau_i in R^7`：第 i 条 FR3 的七个关节力矩，单位 N·m；
- `J_i in R^(6x7)`：第 i 条机械臂 TCP Jacobian；
- `W_i in R^6`：该臂分配到的末端 wrench；
- `g_i in R^7`：当前构型下的重力补偿 torque。

关节限制：

```text
-tau_i,max <= tau_i <= tau_i,max
```

其中 `tau_i,max` 是每个关节允许的力矩上限。

所以完整 Stage-1 QP：

```text
min_f  1/2 f^T H f

s.t.
       G f = W_O_des

       -tau_L,max <= J_L^T W_L + g_L <= tau_L,max
       -tau_R,max <= J_R^T W_R + g_R <= tau_R,max
```

---

## 6. 三个实验 Case

### Case A — 正常限制

SharedBox：

```text
mass = 1 kg
```

静止承重所需物体 wrench：

```text
W_O_des = [0, 0, mg, 0, 0, 0]
```

其中：

- `m=1 kg`：物体质量；
- `g=9.81 m/s^2`：重力加速度；
- 因此期望向上的总支撑力约为 `9.81 N`。

观察对称系统是否产生可解释的左右分配。

### Case B — 人为收紧一个左臂关节

程序自动找到 baseline 中利用率较高的左臂关节，然后把它的 torque limit 收紧。

QP 必须仍满足：

```text
G f = W_O_des
```

同时避免该关节超限。

重点观察左右 `Fz` 是否发生重新分配，以及其它 wrench component 是否参与补偿。

### Case C — 故意制造不可行问题

两条机械臂的 torque limits 都压得极低。

如果不存在同时满足物体 wrench 和 torque limits 的解：

```text
QP infeasible
```

程序必须明确报告失败，而不是输出一个偷偷违反约束的 wrench。

---

## 7. Solver

Stage 1 使用 SciPy SLSQP 作为数值 backend。

需要明确：

> SLSQP 是通用约束优化器；本实验交给它的问题本身是凸二次目标 + 线性等式/不等式约束，因此数学问题就是 QP。

后续如果需要实时控制，再替换为 OSQP / qpOASES 一类专门 QP solver，而 Controller Core 的数学模型不需要因此改变。

---

## 8. 运行

先更新依赖：

```bash
pip install -r requirements.txt
```

运行：

```bash
python3 experiments/task09_wrench_distribution_qp.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml
```

输出：

```text
outputs/task09/wrench_distribution_qp.png
outputs/task09/wrench_distribution_cases.csv
outputs/task09/task09_dual_fr3_geometry.xml
```

---

## 9. 本阶段教学门禁

完成 Stage 1 后应该能够解释：

```text
W_O_des
    ↓
G f = W_O_des
    ↓
many feasible f
    ↓
QP objective + constraints
    ↓
choose W_L and W_R
    ↓
J_L^T W_L / J_R^T W_R
    ↓
joint torque
```

并明确：

> QP 不负责决定物体“想做什么”；物体期望 wrench 已经由上层任务给定。QP 负责在满足这个物体任务的前提下，选择一组更合适的左右臂 wrench。


## 2026-09-21 Stage 1 实验结果

### 几何与目标

```text
W_obj_des = [0, 0, 9.81, 0, 0, 0]

r_L = [-0.070501, 0, 0.00000243] m
r_R = [ 0.070501, 0, 0.00000243] m

G shape = (6, 12)
rank(G) = 6
```

这意味着 12 个左右末端 wrench 分量只受到 6 个物体 wrench 等式约束，因此存在非唯一分配与 6 维 null-space / internal-wrench 自由度。

### Case A — 正常 torque limits

```text
W_L = [0, 0, 4.905, 0, 0, 0]
W_R = [0, 0, 4.905, 0, 0, 0]

Gf = [0, 0, 9.81, 0, 0, 0]
equality residual = 0
```

对称抓取、对称权重与对称关节限制下，QP 自动得到 50/50 静态承重分配。

### Case B — 左臂 J2 torque limit 收紧

```text
baseline |tau_L,J2| = 28.7195 N m
new limit           = 15.7957 N m

W_L =
[ 10.585, 0, -10.255, 0, 1.4319, 0 ]

W_R =
[-10.585, 0,  20.065, 0, 0.7057, 0 ]

Gf residual          = 1.776e-15
max torque violation = 7.745e-13 N m
```

QP 成功满足物体 wrench 与 torque limit，但分配出现：

- 左臂向下的 `Fz`；
- 右臂承担超过物体重量的向上 `Fz`；
- 左右相反的 `Fx`；
- 配套的 `My`。

这些分量在物体层面互相抵消，因此不改变目标物体 wrench，但会改变两臂各自的关节负担。

这说明当前 Stage 1 只有 object-wrench equality + joint-torque limits，还没有加入真实抓取的 unilateral/contact admissibility constraints。由于 Task08 使用 ideal weld，weld 可以数学上承受拉、压、剪切和力矩，因此这种分配在当前模型中允许，但对真实吸盘/夹持接触未必物理可行。

### Case C — 故意不可行

```text
success              = False
equality residual    = 8.882e-16
max torque violation = 24.08 N m
```

优化器返回的候选点仍能满足 `Gf=W_des`，但严重违反 torque inequality，因此程序正确将该结果拒绝为 infeasible / failure，而没有把它当成合法 wrench allocation。

### Stage 1 结论

```text
grasp matrix construction : PASS
symmetric allocation       : PASS
torque-aware redistribution: PASS
infeasibility detection    : PASS
physical contact realism   : NOT YET INCLUDED
```

下一阶段需要给末端 wrench 增加物理可行域约束，例如法向力方向、力/力矩上限以及后续的 friction-cone / contact-wrench constraints，避免 QP 利用 ideal weld 的“任意拉压能力”得到数学可行但真实抓取不可实现的解。
