"""Simulator-independent joint-space P/PD torque controller.

The controller deliberately knows nothing about MuJoCo or Isaac Sim. It maps
joint state/reference vectors to commanded joint torque.
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
class JointPDController:
    """Joint-space torque controller.

    Control law:
        tau = Kp (q_des - q) + Kd (qdot_des - qdot) + tau_ff

    ``tau_ff`` is optional. Task02 uses it for gravity compensation.
    """

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
    ) -> "JointPDController":
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
        qdot_des: Sequence[float] | None = None,
        tau_ff: Sequence[float] | None = None,
    ) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        q_des = np.asarray(q_des, dtype=float)
        if q.shape != self.kp.shape or qdot.shape != self.kp.shape or q_des.shape != self.kp.shape:
            raise ValueError("q, qdot, and q_des must match controller DOF")

        if qdot_des is None:
            qdot_des_vec = np.zeros_like(q)
        else:
            qdot_des_vec = np.asarray(qdot_des, dtype=float)
            if qdot_des_vec.shape != q.shape:
                raise ValueError("qdot_des must match q shape")

        if tau_ff is None:
            tau_ff_vec = np.zeros_like(q)
        else:
            tau_ff_vec = np.asarray(tau_ff, dtype=float)
            if tau_ff_vec.shape != q.shape:
                raise ValueError("tau_ff must match q shape")

        tau = self.kp * (q_des - q) + self.kd * (qdot_des_vec - qdot) + tau_ff_vec

        if self.torque_limits is not None:
            tau = np.clip(tau, -self.torque_limits, self.torque_limits)
        return tau
