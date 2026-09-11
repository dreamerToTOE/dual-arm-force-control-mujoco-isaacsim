# sim

这里存放仿真器适配层，而不是控制算法。

目标结构：

```text
mujoco_adapter.py
isaac_adapter.py
```

两者都实现统一接口，例如：

```text
get_q()
get_qdot()
get_mass_matrix()
get_gravity()
get_coriolis()
get_jacobian()
get_tcp_pose()
get_tcp_velocity()
get_tcp_wrench()
set_joint_torque(tau)
```

Task00--11 先实现和使用 MuJoCoAdapter；Task12 才实现 IsaacAdapter。
