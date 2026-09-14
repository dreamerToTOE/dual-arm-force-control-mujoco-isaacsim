"""Minimal MuJoCo adapter used by Task00.

The controller/experiment code should access robot state through this class
instead of reading ``mujoco.MjData`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np


FR3_JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))


@dataclass(frozen=True)
class JointMap:
    name: str
    joint_id: int
    qpos_adr: int
    dof_adr: int


class MuJoCoAdapter:
    """Task00 subset of the simulator-independent robot interface.

    Torque commands are written to ``qfrc_applied``.  This is intentional:
    MuJoCo Menagerie's FR3 model currently ships with position actuators, whose
    ``ctrl`` values are position references rather than torques.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_names: Sequence[str] = FR3_JOINT_NAMES,
    ) -> None:
        self.model = model
        self.data = data
        self.joint_names = tuple(joint_names)
        self._joint_map = tuple(self._build_joint_map(name) for name in self.joint_names)

    @classmethod
    def from_xml_path(
        cls,
        xml_path: str,
        joint_names: Sequence[str] = FR3_JOINT_NAMES,
    ) -> "MuJoCoAdapter":
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        return cls(model, data, joint_names)

    def _build_joint_map(self, name: str) -> JointMap:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint not found in MuJoCo model: {name}")
        return JointMap(
            name=name,
            joint_id=joint_id,
            qpos_adr=int(self.model.jnt_qposadr[joint_id]),
            dof_adr=int(self.model.jnt_dofadr[joint_id]),
        )

    @property
    def joint_map(self) -> tuple[JointMap, ...]:
        return self._joint_map

    def get_q(self) -> np.ndarray:
        return np.asarray([self.data.qpos[j.qpos_adr] for j in self._joint_map], dtype=float)

    def get_qdot(self) -> np.ndarray:
        return np.asarray([self.data.qvel[j.dof_adr] for j in self._joint_map], dtype=float)

    def set_joint_torque(self, tau: Sequence[float]) -> None:
        tau_array = np.asarray(tau, dtype=float)
        if tau_array.shape != (len(self._joint_map),):
            raise ValueError(
                f"Expected torque shape {(len(self._joint_map),)}, got {tau_array.shape}"
            )

        # Clear only this robot's commanded generalized forces, then apply the
        # requested 7-DoF torque vector.
        for joint in self._joint_map:
            self.data.qfrc_applied[joint.dof_adr] = 0.0
        for joint, torque in zip(self._joint_map, tau_array):
            self.data.qfrc_applied[joint.dof_adr] = float(torque)

    def clear_joint_torque(self) -> None:
        self.set_joint_torque(np.zeros(len(self._joint_map)))

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)
