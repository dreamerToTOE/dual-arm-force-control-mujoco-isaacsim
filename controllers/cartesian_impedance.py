"""Simulator-independent Cartesian impedance controllers.

Task05 starts with translational impedance only:

    F_cmd = Kx (p_des - p) + Dx (v_des - v)

The force is expressed in the same Cartesian frame as the position and velocity
errors. Mapping from Cartesian force to joint torque remains a separate step:

    tau_task = Jv.T @ F_cmd

Keeping these pieces separate makes the controller reusable with MuJoCo and
Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _vec3(value: float | Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(3, float(arr), dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be scalar or shape (3,), got {arr.shape}")
    return arr.copy()


@dataclass
class TranslationalCartesianImpedance:
    """3D translational virtual spring-damper controller."""

    stiffness: np.ndarray
    damping: np.ndarray
    force_limits: np.ndarray | None = None

    @classmethod
    def from_gains(
        cls,
        stiffness: float | Sequence[float],
        damping: float | Sequence[float],
        force_limits: float | Sequence[float] | None = None,
    ) -> "TranslationalCartesianImpedance":
        k = _vec3(stiffness, "stiffness")
        d = _vec3(damping, "damping")
        limits = None if force_limits is None else np.abs(_vec3(force_limits, "force_limits"))
        if np.any(k < 0.0) or np.any(d < 0.0):
            raise ValueError("stiffness and damping must be non-negative")
        return cls(stiffness=k, damping=d, force_limits=limits)

    def compute(
        self,
        position: Sequence[float],
        velocity: Sequence[float],
        desired_position: Sequence[float],
        desired_velocity: Sequence[float] | None = None,
    ) -> np.ndarray:
        p = _vec3(position, "position")
        v = _vec3(velocity, "velocity")
        p_des = _vec3(desired_position, "desired_position")
        v_des = np.zeros(3) if desired_velocity is None else _vec3(desired_velocity, "desired_velocity")

        force = self.stiffness * (p_des - p) + self.damping * (v_des - v)
        if self.force_limits is not None:
            force = np.clip(force, -self.force_limits, self.force_limits)
        return force
