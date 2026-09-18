# Task06 Stage 2 — 不同接触环境下的同一 PI 力控制器

## 目标

Stage 1B 已经得到一个稳定的法向力 PI 闭环：

```text
Kp = 0.8
Ki = 0.8
F_des = 10 N
force-filter tau = 0.040 s
push-command slew rate = 20 N/s
```

Stage 2 **不再修改控制器**，只改变环境接触动力学，观察同一组控制参数在不同表面上的响应差异。

核心问题：

> 力控制器的参数能否脱离环境刚度/阻尼单独讨论？

答案应该通过实验得到，而不是先假定。

## MuJoCo 接触参数

本实验使用 MuJoCo 正值形式：

```text
solref = [timeconst, damping_ratio]
```

这一阶段固定：

```text
damping_ratio = 1.0
```

只改变 `timeconst`：

```text
soft     : 0.050 s
baseline : 0.015 s
firm     : 0.008 s
```

教学上可把它理解为：

```text
timeconst 大 -> 接触约束更软、更慢
timeconst 小 -> 接触约束更硬、更快
```

注意：这里的 `timeconst` 不是材料的 SI 刚度 `N/m`，因此不要把它直接写成“表面刚度 = 某某 N/m”。

为了让差异真正来自指定接触参数，脚本对：

```text
task06_probe
task06_surface
```

都写入相同的 `solref/solimp`。

## 为什么控制器必须完全一样

如果 soft surface 用一组 PI 参数、firm surface 又重新调一组，就无法判断结果变化到底来自：

- 环境变化；
- 还是 controller gain 变化。

因此 Stage 2 固定：

```text
PI gains
force filter
command slew rate
target force
approach strategy
simulation duration
robot model
```

只改变 contact time constant。

## 观察指标

继续使用 Stage 1B 的指标：

- `rise90`
- sustained settling time
- overshoot
- steady-state error
- force STD
- force ripple
- contact-loss ratio

另外新增：

```text
mean penetration [mm]
```

其中 penetration 来自负的 probe-plane clearance。

它用于观察：

> 为了建立同样的法向接触力，不同“软硬”接触模型需要多少几何压入量。

通常更软的接触会需要更大的 penetration 才建立同量级的接触反力。

## 运行

```bash
python3 experiments/task06_surface_contact_sweep.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml
```

带 Viewer：

```bash
python3 experiments/task06_surface_contact_sweep.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml \
  --viewer
```

默认 Viewer 重放 baseline surface。

也可以：

```bash
python3 experiments/task06_surface_contact_sweep.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml \
  --viewer \
  --viewer-surface soft
```

可选：

```text
soft
baseline
firm
```

## 输出

```text
outputs/task06/stage2/surface_contact_comparison.png
outputs/task06/stage2/surface_steady_zoom.png

outputs/task06/stage2/surface_soft.csv
outputs/task06/stage2/surface_baseline.csv
outputs/task06/stage2/surface_firm.csv
```

## 验收重点

Stage 2 不要求三种表面都得到完全相同的曲线。

真正的学习目标是能够解释：

1. 为什么相同 `F_des` 和相同 PI gains，在不同 contact dynamics 下会出现不同 transient；
2. 为什么软接触和硬接触达到同样法向力时的 penetration 不同；
3. 为什么更硬的环境更容易让高增益力环产生 overshoot / oscillation；
4. 为什么真实机器人做 force control 时，controller gains 必须结合被接触环境一起调试。

完成这一步后，Task06 的“接触力闭环基础”就可以结束，下一步进入 Task07 Hybrid Position/Force Control。

## 2026-09-18 实验结果

相同稳定 PI 控制器：

```text
Kp = 0.8
Ki = 0.8
F_des = 10 N
filter tau = 0.040 s
command slew rate = 20 N/s
```

三种接触条件：

```text
soft     : timeconst = 0.050 s
baseline : timeconst = 0.015 s
firm     : timeconst = 0.008 s
```

实验结果：

```text
surface    rise90[s]  settle[s]  overshoot[%]  steady err[N]  STD[N]  ripple[N]  loss[%]  penetration[mm]
soft          3.9300     5.3520        0.000         0.1225      0.0215    0.0744    0.000      0.18271
baseline      3.9920     5.4420        0.000         0.1345      0.0222    0.0769    0.000      0.01712
firm          4.0700     5.5340        0.000         0.1449      0.0230    0.0798    0.000      0.00487
```

主要结论：

1. 同一 PI 控制器在三种接触动力学下均稳定，无 overshoot、无 contact loss；
2. soft contact 需要显著更大的 penetration 才建立约 10 N 法向力；firm contact penetration 最小；
3. 在当前保守 PI + 低通滤波 + command slew limit 下，三种表面的 force tracking 差异被明显压缩，rise/settle 仅有小幅变化；
4. firm case 初始接触冲击更明显，但随后受到滤波和 command 限速约束，长期响应仍然稳定；
5. 本实验说明 controller gains 不能与 contact dynamics 完全割裂，但一个足够保守且带工程化稳定措施的力环可以覆盖一定范围的接触条件。

Stage 2 状态：

```text
soft: PASS
baseline: PASS
firm: PASS
Task06 Stage 2: PASS
Task06 overall: PASS
```

下一步进入 Task07 Hybrid Position/Force Control：显式使用位置/力选择矩阵，把切向位置控制与法向力控制写成统一结构。
