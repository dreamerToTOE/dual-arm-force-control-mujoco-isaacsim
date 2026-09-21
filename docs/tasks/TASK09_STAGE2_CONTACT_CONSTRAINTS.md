# Task09 Stage 2 — Contact / Wrench Feasibility Constraints

## 目标

Stage 1 已经验证：

```text
G f = W_obj_des
+ joint torque limits
```

可以进行双臂 wrench redistribution。

但 Stage 1 的 torque-limited Case B 出现：

```text
Fz_L < 0
Fz_R > mg
左右还有相反的 Fx
```

它在 ideal weld 模型下数学可行，但未必符合真实侧面抓取。

Stage 2 增加一个简化的**物理接触可行域**。

---

## 1. 左右侧抓取的法向方向

本实验把末端 wrench 定义为“机械臂施加在 SharedBox 上的 wrench”。

左抓取点位于物体左侧，因此 inward normal 是：

```text
+WORLD-X
```

定义：

```text
F_n,L = F_Lx
```

右抓取点位于物体右侧，inward normal 是：

```text
-WORLD-X
```

定义：

```text
F_n,R = -F_Rx
```

其中：

- `F_n,L,F_n,R`：左右接触法向压紧力，单位 N；
- `F_Lx,F_Rx`：左右 wrench 的 X 向力分量。

加入 unilateral constraint：

```text
F_n,i >= 0
```

意思是普通接触允许“压”，不允许无条件“拉”。

同时：

```text
F_n,i <= F_n,max
```

避免 QP 为了获得更大摩擦力而使用无限大的夹紧力。

本实验：

```text
F_n,max = 30 N
```

---

## 2. 摩擦约束

理想 Coulomb 摩擦圆锥：

```text
sqrt(F_y^2 + F_z^2) <= mu F_n
```

其中：

- `F_y,F_z`：接触面的两个切向力分量，单位 N；
- `F_n`：法向压紧力，单位 N；
- `mu`：摩擦系数，无量纲。

为了保持线性 inequality，本阶段采用保守 friction pyramid：

```text
|F_y| + |F_z| <= mu F_n
```

因为：

```text
sqrt(F_y^2+F_z^2) <= |F_y|+|F_z|
```

所以满足这个菱形约束一定也不会超过 Coulomb 圆锥，只是比真实圆锥更保守。

本实验：

```text
mu = 0.8
```

对于只需要竖直承重的情况：

```text
|F_z| <= mu F_n
```

因此左右每只手若各承担：

```text
F_z = mg/2 = 4.905 N
```

至少需要：

```text
F_n >= 4.905 / 0.8 = 6.13125 N
```

的内向夹紧力。

这会自然产生一对互相抵消的 internal compression force。

---

## 3. 接触力矩限制

每个接触 wrench 仍然写成：

```text
W_i = [Fx,Fy,Fz,Mx,My,Mz]
```

Stage 2 另外限制：

```text
|M_i,k| <= M_k,max
```

其中：

- `M_i,k`：第 i 个接触绕 k 轴传递的力矩，单位 N·m；
- `M_k,max`：对应允许的最大接触力矩。

当前教学值：

```text
|Mx|, |My|, |Mz| <= 2 N*m
```

这只是教学级 wrench bound，不代表特定真实吸盘或夹具的额定值。

---

## 4. 完整 Stage-2 QP

决策变量：

```text
f = [W_L; W_R] in R^12
```

目标：

```text
min_f  1/2 f^T H f
```

其中：

- `H`：正定权重矩阵；
- `f`：左右两个 6D wrench 拼成的 12 维向量。

物体任务：

```text
G f = W_obj_des
```

其中：

- `G in R^(6x12)`：双臂 grasp matrix；
- `W_obj_des in R^6`：物体期望总 wrench。

关节限制：

```text
-tau_i,max <= J_i^T W_i + g_i <= tau_i,max
```

其中：

- `J_i in R^(6x7)`：第 i 条 FR3 的 TCP Jacobian；
- `W_i in R^6`：该臂末端 wrench；
- `g_i in R^7`：重力补偿 torque；
- `tau_i,max in R^7`：各关节 torque limit。

接触限制：

```text
0 <= F_n,i <= F_n,max
|F_y| + |F_z| <= mu F_n,i
|M_k| <= M_k,max
```

这些都可以写成：

```text
A_contact f <= b_contact
```

其中：

- `A_contact`：线性接触约束矩阵；
- `b_contact`：对应上界向量。

因此整个问题仍然是：

> quadratic objective + linear equality + linear inequality

也就是标准凸 QP。

---

## 5. 三组对比

程序会运行：

### Physical baseline

正常 torque limits + 接触约束。

预期：

- 左右竖直载荷接近对称；
- 两侧产生正的 inward normal compression；
- friction utilization 不超过 100%。

### Same torque bottleneck：无接触约束 vs 有接触约束

程序自动寻找一个“比 baseline 更严格、但加入接触约束后仍然可行”的左臂 torque bottleneck。

随后用**完全相同的 torque limit**求两次：

```text
B1: no contact constraints
B2: with contact constraints
```

用来直接观察：

> contact constraints 如何阻止 QP 利用 ideal weld 的任意拉压能力。

### Insufficient contact capacity

故意令：

```text
F_n,max = 2 N
mu = 0.8
```

每只手最多只能提供：

```text
mu F_n,max = 1.6 N
```

竖直摩擦支撑。

双手合计最多：

```text
3.2 N
```

但 1 kg SharedBox 需要：

```text
mg = 9.81 N
```

因此该问题应该明确 infeasible。

---

## 6. 运行

```bash
python3 experiments/task09_contact_constrained_qp.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml
```

输出：

```text
outputs/task09/stage2/contact_constrained_wrench_qp.png
outputs/task09/stage2/contact_constrained_wrench_cases.csv
outputs/task09/stage2/task09_stage2_geometry.xml
```

---

## 7. 教学门禁

完成 Stage 2 后应该能够区分：

```text
object wrench feasibility
joint torque feasibility
contact wrench feasibility
```

并理解：

> QP 本身不会自动知道“什么叫物理合理”。只有把真实系统的约束写进优化问题，它才会在物理可行域里寻找最优解。
