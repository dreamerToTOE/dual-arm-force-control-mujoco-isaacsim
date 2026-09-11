# controllers

这里存放**与仿真器无关的 Controller Core**。

规划中的模块：

```text
joint_pd.py
computed_torque.py
cartesian_impedance.py
force_controller.py
hybrid_position_force.py
wrench_allocator.py
internal_force_controller.py
```

约束：

- 不直接调用 MuJoCo API；
- 不直接调用 Isaac/PhysX API；
- 输入来自统一 RobotModelInterface；
- 输出统一为关节力矩 `tau` 或双臂 `tau_L / tau_R`。

正式代码从 Task02 起逐步加入，不提前一次性实现全部控制器。
