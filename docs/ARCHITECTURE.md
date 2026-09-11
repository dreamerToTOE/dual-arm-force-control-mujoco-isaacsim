# Architecture — Controller Core 与 Simulator Adapter 解耦

## 1. 目标

本项目从第一天开始就要求控制算法与仿真器 API 解耦，避免后期从 MuJoCo 迁移到 Isaac Sim 时重写控制器。

```text
                    ┌──────────────────────┐
                    │   Controller Core    │
                    │  impedance / force   │
                    │  QP / internal force│
                    └──────────┬───────────┘
                               │ tau
                 ┌─────────────┴─────────────┐
                 │                           │
          MuJoCoAdapter                IsaacAdapter
                 │                           │
              MuJoCo                  Isaac Sim/Lab
```

## 2. RobotModelInterface

建议统一接口：

```python
class RobotModelInterface:
    def get_q(self): ...
    def get_qdot(self): ...
    def get_mass_matrix(self): ...
    def get_gravity(self): ...
    def get_coriolis(self): ...
    def get_jacobian(self, frame): ...
    def get_tcp_pose(self, frame): ...
    def get_tcp_velocity(self, frame): ...
    def get_tcp_wrench(self, frame): ...
    def set_joint_torque(self, tau): ...
```

控制器禁止直接读取：

```text
mujoco.MjData
isaac articulation.data
PhysX API
```

这些只能出现在 Adapter 内。

## 3. 统一坐标系约定

建议从 Task00 起固定：

- 机器人动力学量：base frame；
- 末端位姿：base frame；
- 接触 wrench：明确注明 TCP / contact frame；
- 双臂共同物体：object frame；
- 所有 6D wrench 顺序统一为 `[Fx,Fy,Fz,Mx,My,Mz]`。

任何跨 frame 计算必须显式变换，禁止依赖“看起来方向一样”。

## 4. 单臂控制器层

```text
JointPDController
ComputedTorqueController
CartesianImpedanceController
ForceController
HybridPositionForceController
```

输入统一状态，输出 7-DoF `tau`。

## 5. 双臂控制器层

双臂状态：

```text
q = [qL, qR]
qdot = [qdotL, qdotR]
```

双臂控制器建议分层：

```text
Object-level desired motion / wrench
        ↓
WrenchAllocator
        ↓
WL, WR
        ↓
Left / Right arm torque mapping
        ↓
tauL, tauR
```

其中 Task09 才正式引入 QP。

## 6. 日志与实验接口

所有实验至少统一保存：

```text
time
q, qdot
tau_cmd
TCP pose / velocity
TCP wrench
object pose / velocity
controller reference
```

后续 Task13 跨仿真器对照使用同一指标脚本，不分别写两套评价方法。

## 7. 迁移原则

MuJoCo -> Isaac 时允许变化：

- model loading；
- state query；
- Jacobian/mass-matrix API；
- force sensor API；
- torque command API；
- contact parameters。

原则上不变化：

- 控制公式；
- Controller Core；
- QP 目标与约束定义；
- 实验指标定义。

如果迁移时必须大量修改控制器公式，优先检查是否发生了坐标系、单位或接口耦合错误。
