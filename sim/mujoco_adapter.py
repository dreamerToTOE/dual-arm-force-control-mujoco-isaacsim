"""MuJoCo adapter for the FR3 force-control learning tasks.

Controller/experiment code should access robot state and model quantities through
this class instead of reading ``mujoco.MjData`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np


FR3_JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))
DEFAULT_TCP_SITE = "attachment_site"
DEFAULT_BASE_BODY = "base"


@dataclass(frozen=True)
class JointMap:
    name: str
    joint_id: int
    qpos_adr: int
    dof_adr: int


class MuJoCoAdapter:
    """Simulator adapter exposing the FR3 quantities needed by the tasks.

    Torque commands are written to ``qfrc_applied``. This is intentional:
    MuJoCo Menagerie's FR3 model ships with position actuators, whose ``ctrl``
    values are position references rather than torques.

    Task01 adds TCP pose and geometric Jacobian queries. 6D vectors use the
    project convention ``[linear, angular]``; wrench vectors therefore use
    ``[Fx, Fy, Fz, Mx, My, Mz]``.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_names: Sequence[str] = FR3_JOINT_NAMES,
        tcp_site_name: str = DEFAULT_TCP_SITE,
        base_body_name: str = DEFAULT_BASE_BODY,
    ) -> None:
        self.model = model
        self.data = data
        self.joint_names = tuple(joint_names)
        self.tcp_site_name = tcp_site_name
        self.base_body_name = base_body_name
        self._joint_map = tuple(self._build_joint_map(name) for name in self.joint_names)

        self._tcp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.tcp_site_name
        )
        if self._tcp_site_id < 0:
            raise ValueError(f"TCP site not found in MuJoCo model: {self.tcp_site_name}")

        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_body_name
        )
        if self._base_body_id < 0:
            raise ValueError(f"Base body not found in MuJoCo model: {self.base_body_name}")

    @classmethod
    def from_xml_path(
        cls,
        xml_path: str,
        joint_names: Sequence[str] = FR3_JOINT_NAMES,
        tcp_site_name: str = DEFAULT_TCP_SITE,
        base_body_name: str = DEFAULT_BASE_BODY,
    ) -> "MuJoCoAdapter":
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        return cls(model, data, joint_names, tcp_site_name, base_body_name)

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

    def _world_to_base_rotation(self) -> np.ndarray:
        """Rotation that maps vector coordinates from world to base frame."""
        r_world_base = np.asarray(self.data.xmat[self._base_body_id], dtype=float).reshape(3, 3)
        return r_world_base.T

    def get_tcp_pose(self, frame: str = "base") -> tuple[np.ndarray, np.ndarray]:
        """Return TCP position and 3x3 rotation matrix in world or base frame."""
        p_world = np.asarray(self.data.site_xpos[self._tcp_site_id], dtype=float).copy()
        r_world_tcp = np.asarray(
            self.data.site_xmat[self._tcp_site_id], dtype=float
        ).reshape(3, 3).copy()

        if frame == "world":
            return p_world, r_world_tcp
        if frame != "base":
            raise ValueError("frame must be 'world' or 'base'")

        p_world_base = np.asarray(self.data.xpos[self._base_body_id], dtype=float)
        r_base_world = self._world_to_base_rotation()
        p_base = r_base_world @ (p_world - p_world_base)
        r_base_tcp = r_base_world @ r_world_tcp
        return p_base, r_base_tcp

    def get_jacobian(self, frame: str = "base") -> np.ndarray:
        """Return the TCP geometric Jacobian with shape (6, 7).

        Row order is ``[vx, vy, vz, wx, wy, wz]``. MuJoCo computes the site
        Jacobian in world coordinates; for the fixed-base FR3 we rotate its
        linear and angular components into the base frame when requested.
        """
        jacp_world = np.zeros((3, self.model.nv), dtype=float)
        jacr_world = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(
            self.model,
            self.data,
            jacp_world,
            jacr_world,
            self._tcp_site_id,
        )

        dof_indices = [joint.dof_adr for joint in self._joint_map]
        jacp = jacp_world[:, dof_indices]
        jacr = jacr_world[:, dof_indices]

        if frame == "base":
            r_base_world = self._world_to_base_rotation()
            jacp = r_base_world @ jacp
            jacr = r_base_world @ jacr
        elif frame != "world":
            raise ValueError("frame must be 'world' or 'base'")

        return np.vstack((jacp, jacr))

    def set_joint_torque(self, tau: Sequence[float]) -> None:
        tau_array = np.asarray(tau, dtype=float)
        if tau_array.shape != (len(self._joint_map),):
            raise ValueError(
                f"Expected torque shape {(len(self._joint_map),)}, got {tau_array.shape}"
            )

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
