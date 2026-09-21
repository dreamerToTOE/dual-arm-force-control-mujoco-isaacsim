"""Simulator-independent dual-arm wrench allocation with a small convex QP.

Wrench convention:
    W = [Fx, Fy, Fz, Mx, My, Mz]

For contact point r (object center -> contact point), expressed in the same
frame as the wrench,

    W_object = G_i W_i

with

    G_i = [[I, 0],
           [[r]x, I]]

because the contact force contributes moment r x F about the object center.

The dual-arm grasp map is

    G = [G_L  G_R]

and the allocator solves

    min_f  0.5 (f-f_ref)^T H (f-f_ref)
    s.t.   G f = W_object_des
           -tau_max <= A_tau f + tau_bias <= tau_max

where f = [W_L; W_R].

SciPy SLSQP is used as the numerical backend.  The optimization problem itself
is a convex quadratic program with linear equality/inequality constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


def skew(v: Sequence[float]) -> np.ndarray:
    """Return the 3x3 skew matrix [v]x such that [v]x a = v x a."""
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )


def grasp_map(contact_offset: Sequence[float]) -> np.ndarray:
    """Map a 6D contact wrench to the object-center wrench.

    contact_offset is r = p_contact - p_object, expressed in the object/task
    frame.  Wrench order is [F, M].
    """
    r = np.asarray(contact_offset, dtype=float)
    if r.shape != (3,):
        raise ValueError(f"contact_offset must have shape (3,), got {r.shape}")

    g = np.zeros((6, 6), dtype=float)
    g[:3, :3] = np.eye(3)
    g[3:, :3] = skew(r)
    g[3:, 3:] = np.eye(3)
    return g


def dual_grasp_matrix(
    left_contact_offset: Sequence[float],
    right_contact_offset: Sequence[float],
) -> np.ndarray:
    """Return G=[G_L G_R] with shape (6,12)."""
    return np.hstack(
        (grasp_map(left_contact_offset), grasp_map(right_contact_offset))
    )


def side_grasp_contact_constraints(
    friction_coefficient: float,
    normal_force_max: float,
    moment_limits: Sequence[float] = (2.0, 2.0, 2.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return A,b for A f <= b for symmetric side contacts.

    Wrench convention for f=[W_L;W_R]:
        W_i = [Fx, Fy, Fz, Mx, My, Mz]

    The forces are the forces applied ON THE OBJECT.

    Contact normals point inward:
        left  normal: +X, so F_n,L = +Fx_L
        right normal: -X, so F_n,R = -Fx_R

    Constraints:
        F_n >= 0
        F_n <= normal_force_max
        |Fy| + |Fz| <= mu F_n      (linear friction pyramid)
        |Mx|, |My|, |Mz| <= moment_limits
    """
    mu = float(friction_coefficient)
    nmax = float(normal_force_max)
    mlim = np.asarray(moment_limits, dtype=float)

    if mu <= 0.0:
        raise ValueError("friction_coefficient must be positive")
    if nmax <= 0.0:
        raise ValueError("normal_force_max must be positive")
    if mlim.shape != (3,) or np.any(mlim < 0.0):
        raise ValueError("moment_limits must contain three non-negative values")

    rows: list[np.ndarray] = []
    bounds: list[float] = []

    def add(coeffs: dict[int, float], bound: float) -> None:
        row = np.zeros(12, dtype=float)
        for index, value in coeffs.items():
            row[index] = float(value)
        rows.append(row)
        bounds.append(float(bound))

    # Unilateral normal force and maximum normal force.
    # Left:  0 <= Fx_L <= nmax.
    add({0: -1.0}, 0.0)
    add({0: 1.0}, nmax)

    # Right: 0 <= -Fx_R <= nmax  ->  Fx_R <= 0 and -Fx_R <= nmax.
    add({6: 1.0}, 0.0)
    add({6: -1.0}, nmax)

    # Conservative linear friction pyramid:
    # |Fy| + |Fz| <= mu * F_n.
    for sy in (-1.0, 1.0):
        for sz in (-1.0, 1.0):
            add({1: sy, 2: sz, 0: -mu}, 0.0)
            add({7: sy, 8: sz, 6: mu}, 0.0)

    # Independent contact-moment bounds.
    for local_index, limit in zip((3, 4, 5), mlim):
        add({local_index: 1.0}, limit)
        add({local_index: -1.0}, limit)
        add({local_index + 6: 1.0}, limit)
        add({local_index + 6: -1.0}, limit)

    return np.vstack(rows), np.asarray(bounds, dtype=float)


@dataclass
class WrenchAllocationResult:
    success: bool
    status: str
    wrench_left: np.ndarray
    wrench_right: np.ndarray
    object_wrench_reconstructed: np.ndarray
    equality_residual: float
    tau_left: np.ndarray | None
    tau_right: np.ndarray | None
    max_torque_violation: float
    max_linear_inequality_violation: float
    objective: float


class DualArmWrenchQP:
    """Convex dual-arm wrench allocator with optional joint-torque limits."""

    def __init__(
        self,
        grasp_matrix: np.ndarray,
        weight_diag: Sequence[float] | None = None,
    ) -> None:
        g = np.asarray(grasp_matrix, dtype=float)
        if g.shape != (6, 12):
            raise ValueError(f"grasp_matrix must have shape (6,12), got {g.shape}")
        if np.linalg.matrix_rank(g) < 6:
            raise ValueError("grasp_matrix must have full row rank for this Task09 baseline")
        self.G = g

        if weight_diag is None:
            # Forces are measured in N and moments in N*m.  A 0.1 m reference
            # length makes 1 N*m comparable to roughly 10 N in the quadratic cost.
            arm_weights = np.array([1.0, 1.0, 1.0, 100.0, 100.0, 100.0])
            w = np.concatenate((arm_weights, arm_weights))
        else:
            w = np.asarray(weight_diag, dtype=float)
            if w.shape != (12,):
                raise ValueError(f"weight_diag must have shape (12,), got {w.shape}")
            if np.any(w <= 0.0):
                raise ValueError("all weight_diag entries must be positive")

        self.H = np.diag(w)

    def minimum_norm_equality_solution(
        self,
        desired_object_wrench: Sequence[float],
    ) -> np.ndarray:
        """Weighted equality-only solution, useful as baseline / initial guess."""
        w_des = np.asarray(desired_object_wrench, dtype=float)
        if w_des.shape != (6,):
            raise ValueError("desired_object_wrench must have shape (6,)")

        h_inv = np.diag(1.0 / np.diag(self.H))
        middle = self.G @ h_inv @ self.G.T
        return h_inv @ self.G.T @ np.linalg.solve(middle, w_des)

    @staticmethod
    def _torque_map(
        jacobian_left: np.ndarray,
        jacobian_right: np.ndarray,
    ) -> np.ndarray:
        jl = np.asarray(jacobian_left, dtype=float)
        jr = np.asarray(jacobian_right, dtype=float)
        if jl.shape != (6, 7) or jr.shape != (6, 7):
            raise ValueError("left/right Jacobians must have shape (6,7)")

        a = np.zeros((14, 12), dtype=float)
        a[:7, :6] = jl.T
        a[7:, 6:] = jr.T
        return a

    def solve(
        self,
        desired_object_wrench: Sequence[float],
        *,
        reference_wrench: Sequence[float] | None = None,
        jacobian_left: np.ndarray | None = None,
        jacobian_right: np.ndarray | None = None,
        gravity_left: Sequence[float] | None = None,
        gravity_right: Sequence[float] | None = None,
        torque_limits_left: Sequence[float] | None = None,
        torque_limits_right: Sequence[float] | None = None,
        linear_inequality_A: np.ndarray | None = None,
        linear_inequality_b: Sequence[float] | None = None,
        max_iterations: int = 500,
    ) -> WrenchAllocationResult:
        w_des = np.asarray(desired_object_wrench, dtype=float)
        if w_des.shape != (6,):
            raise ValueError("desired_object_wrench must have shape (6,)")

        if reference_wrench is None:
            f_ref = np.zeros(12, dtype=float)
        else:
            f_ref = np.asarray(reference_wrench, dtype=float)
            if f_ref.shape != (12,):
                raise ValueError("reference_wrench must have shape (12,)")

        x0 = self.minimum_norm_equality_solution(w_des)

        def objective(x: np.ndarray) -> float:
            d = x - f_ref
            return 0.5 * float(d @ self.H @ d)

        def gradient(x: np.ndarray) -> np.ndarray:
            return self.H @ (x - f_ref)

        constraints: list[dict] = [
            {
                "type": "eq",
                "fun": lambda x: self.G @ x - w_des,
                "jac": lambda x: self.G,
            }
        ]

        torque_map = None
        torque_bias = None
        torque_limits = None

        linear_a = None
        linear_b = None
        if linear_inequality_A is not None or linear_inequality_b is not None:
            if linear_inequality_A is None or linear_inequality_b is None:
                raise ValueError(
                    "linear_inequality_A and linear_inequality_b must be provided together"
                )
            linear_a = np.asarray(linear_inequality_A, dtype=float)
            linear_b = np.asarray(linear_inequality_b, dtype=float)
            if linear_a.ndim != 2 or linear_a.shape[1] != 12:
                raise ValueError(
                    f"linear_inequality_A must have shape (m,12), got {linear_a.shape}"
                )
            if linear_b.shape != (linear_a.shape[0],):
                raise ValueError(
                    "linear_inequality_b must have one bound for each inequality row"
                )

            # A f <= b  ->  b - A f >= 0 for SLSQP.
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, a=linear_a, b=linear_b: b - a @ x,
                    "jac": lambda x, a=linear_a: -a,
                }
            )

        torque_args = (
            jacobian_left,
            jacobian_right,
            gravity_left,
            gravity_right,
            torque_limits_left,
            torque_limits_right,
        )
        if any(arg is not None for arg in torque_args):
            if not all(arg is not None for arg in torque_args):
                raise ValueError(
                    "Jacobian, gravity, and torque-limit arguments must all be provided together"
                )

            torque_map = self._torque_map(jacobian_left, jacobian_right)
            torque_bias = np.concatenate(
                (
                    np.asarray(gravity_left, dtype=float),
                    np.asarray(gravity_right, dtype=float),
                )
            )
            torque_limits = np.concatenate(
                (
                    np.asarray(torque_limits_left, dtype=float),
                    np.asarray(torque_limits_right, dtype=float),
                )
            )
            if torque_bias.shape != (14,) or torque_limits.shape != (14,):
                raise ValueError("gravity and torque limits must each contain 7 values per arm")
            if np.any(torque_limits < 0.0):
                raise ValueError("torque limits must be non-negative")

            # SLSQP inequality convention: fun(x) >= 0.
            constraints.extend(
                [
                    {
                        "type": "ineq",
                        "fun": lambda x, a=torque_map, b=torque_bias, lim=torque_limits:
                            lim - (a @ x + b),
                        "jac": lambda x, a=torque_map: -a,
                    },
                    {
                        "type": "ineq",
                        "fun": lambda x, a=torque_map, b=torque_bias, lim=torque_limits:
                            lim + (a @ x + b),
                        "jac": lambda x, a=torque_map: a,
                    },
                ]
            )

        result = minimize(
            objective,
            x0,
            jac=gradient,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": int(max_iterations), "disp": False},
        )

        x = np.asarray(result.x, dtype=float)
        equality_residual = float(np.linalg.norm(self.G @ x - w_des, ord=np.inf))

        max_linear_inequality_violation = 0.0
        if linear_a is not None and linear_b is not None:
            max_linear_inequality_violation = float(
                max(0.0, np.max(linear_a @ x - linear_b))
            )

        tau_left = None
        tau_right = None
        max_torque_violation = 0.0
        if torque_map is not None and torque_bias is not None and torque_limits is not None:
            tau = torque_map @ x + torque_bias
            tau_left = tau[:7].copy()
            tau_right = tau[7:].copy()
            max_torque_violation = float(
                max(0.0, np.max(np.abs(tau) - torque_limits))
            )

        success = bool(
            result.success
            and equality_residual < 1e-6
            and max_torque_violation < 1e-6
            and max_linear_inequality_violation < 1e-6
        )
        status = str(result.message)
        if not success and result.success:
            status = (
                f"numerical solution rejected: eq_residual={equality_residual:.3e}, "
                f"torque_violation={max_torque_violation:.3e}, "
                f"linear_ineq_violation={max_linear_inequality_violation:.3e}"
            )

        return WrenchAllocationResult(
            success=success,
            status=status,
            wrench_left=x[:6].copy(),
            wrench_right=x[6:].copy(),
            object_wrench_reconstructed=(self.G @ x).copy(),
            equality_residual=equality_residual,
            tau_left=tau_left,
            tau_right=tau_right,
            max_torque_violation=max_torque_violation,
            max_linear_inequality_violation=max_linear_inequality_violation,
            objective=objective(x),
        )
