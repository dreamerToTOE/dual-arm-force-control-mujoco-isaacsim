# robot_model

这里定义控制器看到的统一机器人动力学接口与公共数据结构。

建议后续包含：

```text
robot_model_interface.py
frames.py
state.py
```

核心目的：把 `q / qdot / M / g / C / J / TCP pose / wrench` 的含义、维度、单位和坐标系固定下来。

任何仿真器特有索引、joint name 映射和 API 调用都不应泄漏到 Controller Core。
