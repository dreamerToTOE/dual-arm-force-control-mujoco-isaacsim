# Task10 — Internal Force / Internal Wrench Control

## 学习目标

Task09 已经能够在满足物体 wrench、关节 torque limits 和接触约束的前提下分配左右末端 wrench。

Task10 研究剩下的自由度：

> 为什么两只机械臂可以互相夹得非常用力，但 SharedBox 的整体合力与合力矩完全不变？

核心分解：

```text
f = f_task + f_internal
```

其中：

- `f in R^12`：左右两个 6D contact wrench 拼成的总分配；
- `f_task in R^12`：负责产生物体目标 wrench 的部分；
- `f_internal in R^12`：internal wrench。

如果：

```text
G f_internal = 0
```

则 `f_internal` 位于 grasp matrix `G` 的零空间，不改变物体净 wrench。

---

## 1. Task component

物体任务仍然是静止承重：

```text
W_obj_des = [0, 0, mg, 0, 0, 0]^T
```

其中：

- `W_obj_des in R^6`：SharedBox 的期望物体 wrench；
- `m = 1 kg`：物体质量；
- `g = 9.81 m/s^2`：重力加速度。

Stage 1 使用一个 Euclidean minimum-norm task component：

```text
f_task = G^T (G G^T)^(-1) W_obj_des
```

其中：

- `G in R^(6x12)`：双臂 grasp matrix；
- `G^T`：G 的转置；
- `f_task in R^12`：在 Euclidean 2-norm 意义下最小的 task-producing wrench allocation。

因为 `rank(G)=6`，所以：

```text
dim null(G) = 12 - 6 = 6
```

说明还有 6 维 internal-wrench 自由度。

---

## 2. Null-space projector

定义：

```text
N = I - G^+ G
```

其中：

- `N in R^(12x12)`：投影到 `null(G)` 的矩阵；
- `I`：12x12 单位矩阵；
- `G^+`：G 的 Moore-Penrose pseudoinverse。

对于 full-row-rank G：

```text
G^+ = G^T (G G^T)^(-1)
```

理论上：

```text
G N = 0
```

也就是说，被 N 投影后的 wrench 不会对物体产生净 wrench。

---

## 3. 本实验的 internal compression

左右侧面对称抓取：

```text
left inward normal  = +X
right inward normal = -X
```

定义目标 internal compression：

```text
f_internal_des(F_int)
=
[ +F_int, 0,0,0,0,0,
  -F_int, 0,0,0,0,0 ]^T
```

其中：

- `F_int >= 0`：希望的双侧内部夹紧力，单位 N；
- 左臂对物体沿 +X 压；
- 右臂对物体沿 -X 压。

因为两力大小相同、方向相反，并且作用线穿过物体中心：

```text
G f_internal_des = 0
```

所以改变 `F_int` 不应该改变 SharedBox 的净 wrench。

---

## 4. QP 如何控制 internal force

Task09 的 QP 已经支持 reference wrench：

```text
min_f  1/2 (f - f_ref)^T H (f - f_ref)
```

Task10 令：

```text
f_ref = f_task + f_internal_des
```

其中：

- `f_ref in R^12`：希望 QP 接近的完整左右 wrench reference；
- `H in R^(12x12)`：正定权重矩阵。

仍然必须满足：

```text
G f = W_obj_des
```

以及 Task09 已有的：

```text
joint torque limits
unilateral normal constraints
friction-pyramid constraints
contact moment limits
```

所以 internal-force reference 是**优化目标**，而不是允许违反安全约束的强制命令。

---

## 5. 为什么需要 internal compression

侧面摩擦承重约束：

```text
|Fy| + |Fz| <= mu Fn
```

其中：

- `Fy,Fz`：接触切向力，单位 N；
- `Fn`：法向夹紧力，单位 N；
- `mu`：摩擦系数。

在本实验只有竖直承重：

```text
|Fz| <= mu Fn
```

因此增加 internal compression `Fn` 会降低：

```text
friction utilization = |Fz| / (mu Fn)
```

也就是增加摩擦安全裕度。

但 internal compression 太大也不是免费午餐：

- 增加机械臂 joint torque；
- 增加接触压力；
- 真实吸盘/夹具可能损伤物体；
- 可能接近 actuator / contact limits。

因此 Task10 的本质不是“夹得越紧越好”，而是：

> 在足够稳定抓持和不过载之间选择合理的 internal force。

---

## 6. Stage 1 Sweep

固定：

```text
W_obj_des = [0,0,9.81,0,0,0]
mu = 0.8
Fn,max = 30 N/contact
```

扫描：

```text
F_int,des =
6.5, 10, 15, 20, 25, 40 N
```

其中 40 N 故意超过：

```text
Fn,max = 30 N
```

用于验证：

> reference 超过物理约束时，QP 应停在可行边界，而不是违反约束。

---

## 7. 观察指标

每个 internal-force reference 记录：

- desired / actual internal compression；
- left / right friction utilization；
- left / right maximum joint-torque utilization；
- `||G f_internal||_inf`；
- reconstructed object `Fx` / `Fz`；
- contact / torque constraint violation。

理想现象：

```text
F_int increase
    ↓
friction utilization decrease
    ↓
grasp margin increase

but

F_int increase
    ↓
joint torque loading increase
```

同时：

```text
object Fx ≈ 0
object Fz ≈ 9.81 N
```

始终保持不变。

---

## 8. 运行

```bash
python3 experiments/task10_internal_force_sweep.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml
```

输出：

```text
outputs/task10/internal_force_sweep.png
outputs/task10/internal_force_sweep.csv
outputs/task10/task10_dual_fr3_geometry.xml
```

---

## 9. 教学门禁

完成后应该能够解释：

```text
object wrench
    !=
internal wrench
```

物体任务由：

```text
G f = W_obj_des
```

保证。

抓持内部状态由：

```text
null(G)
```

中的自由度调节。

所以：

> 两只手可以同时增加一对大小相同、方向相反的夹紧力，箱子仍然完全不获得额外净力；但机械臂自己的关节负担与接触安全裕度会发生明显变化。


## 2026-09-22 Stage 1 实验结果

固定物体任务：

```text
W_obj_des = [0, 0, 9.81, 0, 0, 0]
mu = 0.8
Fn,max = 30 N/contact
```

internal compression sweep：

```text
ref[N]  actual[N]  friction L/R[%]  torque L/R[%]  ||G f_int||inf  object Fx/Fz[N]
  6.5      6.500      94.33/94.33      30.83/30.83      0.000e+00      0.000/9.810
 10.0     10.000      61.31/61.31      29.66/29.66      0.000e+00      0.000/9.810
 15.0     15.000      40.87/40.88      30.83/30.83      1.776e-15      0.000/9.810
 20.0     20.000      30.66/30.66      35.29/35.29      0.000e+00      0.000/9.810
 25.0     25.000      24.52/24.53      39.75/39.75      3.553e-15      0.000/9.810
 40.0     30.000      20.44/20.44      44.21/44.21      ~0             0.000/9.810
```

主要结论：

1. `G f_internal = 0` 数值验证通过，null-space residual 约为 1e-15 或更小；
2. internal compression 增大时，object wrench 保持不变；
3. internal compression 增大时，friction utilization 明显下降，即摩擦裕度增加；
4. joint torque utilization 整体随 internal compression 增大而上升，但不是严格单调，因为增加某个方向的 internal force 可能在某些构型下先部分抵消已有的重力/任务关节力矩；
5. 当 desired compression=40 N 超过 contact normal max=30 N 时，QP 将 realized compression 截止在 30 N，而不违反约束。

Stage 1 状态：

```text
null-space decomposition : PASS
internal-force reference : PASS
friction-margin tradeoff : PASS
torque-loading tradeoff  : PASS
constraint saturation    : PASS
```

注意：当前 Stage 1 是静态 wrench allocation / reference shaping，还不是基于实测 internal-force feedback 的动态闭环。若继续做 Task10 Stage 2，可进一步研究动态 closed-chain 中的 internal-force feedback control。
