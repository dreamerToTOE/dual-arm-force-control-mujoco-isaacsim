# Task06 Stage 1B — 接触力闭环稳定化

## 为什么需要这一阶段

Stage 1 已经证明真实 MuJoCo contact、`mj_contactForce` 读取、P/PI force loop、`J^T` 映射和重力补偿能够完整运行。

但是第一次实验中，PI 虽然把最后 0.6 s 的平均稳态误差从约 `3.43 N` 降到了约 `1.91 N`，却出现了明显高频力波动和较大 overshoot。因此不能只凭平均误差判断控制质量。

Stage 1B 的目标不是换一种控制方法，而是把同一个 P/PI 力控制闭环做得更符合真实接触控制的工程常识。

## 稳定化改动

新实验：

```text
experiments/task06_contact_force_control_stable.py
```

相对 Stage 1 增加：

1. 一阶低通滤波：

```text
F_meas_raw -> low-pass -> F_meas_filtered -> P/PI
```

时间常数：

```text
tau_filter = 0.030 s
```

2. push-force command slew-rate limit：

```text
|dF_cmd/dt| <= 40 N/s
```

避免 force command 在相邻仿真步之间近似瞬间跳变。

3. 更低的力环增益：

```text
P : Kp = 0.8, Ki = 0
PI: Kp = 0.8, Ki = 0.8
```

4. 更柔和的接触基线：

```text
solref = 0.030 1.0
```

Stage 2 再系统比较不同接触刚度/阻尼。

## 新增指标

### 1. filtered rise90

从第一次 contact 开始，到低通后的法向力第一次达到：

```text
0.9 * F_des
```

所需时间。

它仍然只是速度指标，不能单独代表“已经稳定”。

### 2. sustained settling time

要求：

```text
|F_filtered - F_des| <= 0.5 N
```

并连续保持至少：

```text
0.25 s
```

才认为进入稳定目标带。

### 3. steady-state error

仍然使用最后一段实际 raw contact force 的平均值：

```text
e_ss = F_des - mean(F_meas_raw)
```

### 4. force STD

最后 1 s raw contact force 的标准差。

STD 越小，说明接触力围绕平均值的波动越小。

### 5. force ripple

最后 1 s raw contact force 的 peak-to-peak：

```text
ripple = max(F) - min(F)
```

### 6. contact-loss ratio

第一次接触后跳过 0.2 s 过渡阶段，然后统计：

- probe-plane contact pair 不存在；或
- `F_meas <= 0.05 N`

所占的采样比例。

理想稳定接触应接近：

```text
0 %
```

## 新可视化

主图：

```text
outputs/task06/p_vs_pi_force_control_stable.png
```

P 和 PI 的 force response 分开显示，不再让后绘制的 PI 高频曲线遮住 P。

每个 force plot 同时显示：

- raw `F_meas`；
- filtered `F_meas`；
- `F_des`。

另生成稳态局部放大：

```text
outputs/task06/steady_force_zoom.png
```

用于直接观察最后 1 s 的力波动。

## 运行

```bash
python3 experiments/task06_contact_force_control_stable.py \
  --model ~/mujoco_menagerie/franka_fr3/scene.xml \
  --viewer
```

## 验收原则

Stage 1B 不再接受“平均值看起来接近 10 N”作为充分条件。

至少同时检查：

```text
steady error
overshoot
force STD
force ripple
contact-loss ratio
sustained settling time
```

目标是：PI 相比 P 明显减小 steady error，同时不以持续高频振荡或频繁失去接触为代价。
