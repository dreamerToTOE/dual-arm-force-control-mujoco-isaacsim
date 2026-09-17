"""Simulator-independent Cartesian impedance controllers.

Task05 begins with translational impedance:

    F_cmd = Kx (p_des - p) + Dx (v_des - v)

and then extends to full 6D pose impedance:

    M_cmd = Kr e_R + Dr (omega_des - omega)
    W_cmd = [F_cmd, M_cmd]

where ``e_R`` is the rotation-vector error, expressed in the same BASE frame as
angular velocity and the rotational Jacobian. Mapping from Cartesian wrench to
joint torque remains a separate step:

    tau_task = J.T @ W_cmd

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


def rotation_error_base(current_rotation: np.ndarray, desired_rotation: np.ndarray) -> np.ndarray:
    """Return orientation error as a BASE-frame rotation vector.

    ``current_rotation`` and ``desired_rotation`` are 3x3 matrices mapping TCP
    coordinates into BASE coordinates. The error rotation is chosen so that

        R_des ~= Exp([e_R]x) R

    therefore positive ``e_R`` describes the small BASE-frame rotation that
    moves the current TCP orientation toward the desired orientation.

    The implementation uses the SO(3) logarithm. Task05 stays far from the
    pi-radian singularity, so the standard axis-angle expression is sufficient.
    """
    r = np.asarray(current_rotation, dtype=float)
    r_des = np.asarray(desired_rotation, dtype=float)
    if r.shape != (3, 3) or r_des.shape != (3, 3):
        raise ValueError("current_rotation and desired_rotation must have shape (3, 3)")

    r_err = r_des @ r.T
    cosine = float(np.clip((np.trace(r_err) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew_vec = np.array(
        [
            r_err[2, 1] - r_err[1, 2],
            r_err[0, 2] - r_err[2, 0],
            r_err[1, 0] - r_err[0, 1],
        ],
        dtype=float,
    )

    if angle < 1e-8:
        return 0.5 * skew_vec

    sine = float(np.sin(angle))
    if abs(sine) < 1e-8:
        raise ValueError("orientation error is too close to pi radians for this Task05 representation")

    return (angle / (2.0 * sine)) * skew_vec


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


@dataclass
class CartesianImpedance6D:
    """6D Cartesian virtual spring-damper controller.

    All translational and rotational quantities must be expressed in the same
    frame. Task05 uses BASE frame throughout.
    """

    translational_stiffness: np.ndarray
    translational_damping: np.ndarray
    rotational_stiffness: np.ndarray
    rotational_damping: np.ndarray
    force_limits: np.ndarray | None = None
    moment_limits: np.ndarray | None = None

    @classmethod
    def from_gains(
        cls,
        translational_stiffness: float | Sequence[float],
        translational_damping: float | Sequence[float],
        rotational_stiffness: float | Sequence[float],
        rotational_damping: float | Sequence[float],
        force_limits: float | Sequence[float] | None = None,
        moment_limits: float | Sequence[float] | None = None,
    ) -> "CartesianImpedance6D":
        kt = _vec3(translational_stiffness, "translational_stiffness")
        dt = _vec3(translational_damping, "translational_damping")
        kr = _vec3(rotational_stiffness, "rotational_stiffness")
        dr = _vec3(rotational_damping, "rotational_damping")
        if np.any(kt < 0.0) or np.any(dt < 0.0) or np.any(kr < 0.0) or np.any(dr < 0.0):
            raise ValueError("all stiffness and damping gains must be non-negative")
        f_lim = None if force_limits is None else np.abs(_vec3(force_limits, "force_limits"))
        m_lim = None if moment_limits is None else np.abs(_vec3(moment_limits, "moment_limits"))
        return cls(kt, dt, kr, dr, f_lim, m_lim)

    def compute(
        self,
        position: Sequence[float],
        rotation: np.ndarray,
        linear_velocity: Sequence[float],
        angular_velocity: Sequence[float],
        desired_position: Sequence[float],
        desired_rotation: np.ndarray,
        desired_linear_velocity: Sequence[float] | None = None,
        desired_angular_velocity: Sequence[float] | None = None,
    ) -> np.ndarray:
        p = _vec3(position, "position")
        v = _vec3(linear_velocity, "linear_velocity")
        omega = _vec3(angular_velocity, "angular_velocity")
        p_des = _vec3(desired_position, "desired_position")
        v_des = np.zeros(3) if desired_linear_velocity is None else _vec3(desired_linear_velocity, "desired_linear_velocity")
        omega_des = np.zeros(3) if desired_angular_velocity is None else _vec3(desired_angular_velocity, "desired_angular_velocity")

        e_p = p_des - p
        e_v = v_des - v
        e_r = rotation_error_base(rotation, desired_rotation)
        e_omega = omega_des - omega

        force = self.translational_stiffness * e_p + self.translational_damping * e_v
        moment = self.rotational_stiffness * e_r + self.rotational_damping * e_omega

        if self.force_limits is not None:
            force = np.clip(force, -self.force_limits, self.force_limits)
        if self.moment_limits is not None:
            moment = np.clip(moment, -self.moment_limits, self.moment_limits)

        return np.concatenate((force, moment))
