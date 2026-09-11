# tests

这里存放控制器与适配器的最小自动化测试。

优先测试：

```text
维度 / joint order
单位与坐标系
Jacobian 数值检查
gravity / mass matrix consistency
controller output finite
QP feasible / infeasible handling
MuJoCoAdapter 与 IsaacAdapter 接口一致性
```

测试目标不是替代仿真实验，而是尽早发现“公式没错但接口错了”的问题，特别是关节顺序、坐标系、符号和单位错误。
