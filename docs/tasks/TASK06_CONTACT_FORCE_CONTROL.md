# Task06 — Contact Force Control

## 学习目标

第一次让 FR3 主动接触环境，并控制接触法向力，而不是只控制末端位置。

## 场景

```text
FR3 TCP
   ↓
刚性/可调接触平面
```

第一版只研究一个法向方向，例如 Z 方向。

## 实现任务

1. 在 MuJoCo 中创建平面与 TCP 接触；
2. 读取接触/力传感量；
3. 建立期望法向力 `F_des`；
4. 从最简单的 PI/PID force loop 入门；
5. 通过 `J^T` 转成关节 torque；
6. 比较不同表面刚度、阻尼下的力响应。

## 验收

- 接触后法向力能稳定在目标附近；
- 输出 force rise time、overshoot、steady-state error；
- 不出现明显穿透、发散或持续高频振荡；
- 能解释为什么“目标位置继续向下压”不等同于真正的 force control。

## 注意

这一 Task 的重点是理解接触闭环。不要提前做双臂，也不要为了追求力误差而使用危险的大增益。
