"""Simulator-independent joint-space computed-torque controller.

Control law:
    v   = qddot_des + Kd (qdot_des - qdot) + Kp (q_des - q)
    tau = M(q) v + h(q,qdot)

where h = C(q,qdot) qdot + g(q).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _as_vector(value: float | Sequence[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(size, float(array), dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must be scalar or shape {(size,)}, got {array.shape}")
    return array.copy()


@dataclass
class ComputedTorqueController:
    """Joint-space inverse-dynamics / computed-torque controller."""

    kp: np.ndarray
    kd: np.ndarray
    torque_limits: np.ndarray | None = None

    @classmethod
    def from_gains(
        cls,
        kp: float | Sequence[float],
        kd: float | Sequence[float],
        dof: int = 7,
        torque_limits: float | Sequence[float] | None = None,
    ) -> "ComputedTorqueController":
        kp_vec = _as_vector(kp, dof, "kp")
        kd_vec = _as_vector(kd, dof, "kd")
        limits = None
        if torque_limits is not None:
            limits = np.abs(_as_vector(torque_limits, dof, "torque_limits"))
        return cls(kp=kp_vec, kd=kd_vec, torque_limits=limits)

    def compute(
        self,
        q: Sequence[float],
        qdot: Sequence[float],
        q_des: Sequence[float],
        qdot_des: Sequence[float],
        qddot_des: Sequence[float],
        mass_matrix: np.ndarray,
        bias: Sequence[float],
    ) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        q_des = np.asarray(q_des, dtype=float)
        qdot_des = np.asarray(qdot_des, dtype=float)
        qddot_des = np.asarray(qddot_des, dtype=float)
        mass_matrix = np.asarray(mass_matrix, dtype=float)
        bias = np.asarray(bias, dtype=float)

        n = self.kp.size
        expected = (n,)
        for name, value in (
            ("q", q),
            ("qdot", qdot),
            ("q_des", q_des),
            ("qdot_des", qdot_des),
            ("qddot_des", qddot_des),
            ("bias", bias),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
        if mass_matrix.shape != (n, n):
            raise ValueError(f"mass_matrix must have shape {(n, n)}, got {mass_matrix.shape}")

        e = q_des - q
        edot = qdot_des - qdot
        v = qddot_des + self.kd * edot + self.kp * e
        tau = mass_matrix @ v + bias

        if self.torque_limits is not None:
            tau = np.clip(tau, -self.torque_limits, self.torque_limits)
        return tau
