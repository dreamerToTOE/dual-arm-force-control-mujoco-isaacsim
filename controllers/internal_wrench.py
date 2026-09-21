"""Internal-wrench utilities for dual-arm cooperative manipulation.

For a full-row-rank grasp matrix G and stacked contact wrench

    f = [W_L; W_R],

the object wrench is

    W_obj = G f.

A convenient Euclidean minimum-norm task component is

    f_task = G^T (G G^T)^-1 W_obj_des,

and any other solution can be decomposed as

    f = f_task + f_internal,

where

    G f_internal = 0.

Task10 uses a particularly intuitive null-space direction for symmetric side
grasps: equal inward compression from the left and right contacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def minimum_norm_task_component(
    grasp_matrix: np.ndarray,
    desired_object_wrench: Sequence[float],
) -> np.ndarray:
    """Return the Euclidean minimum-norm f satisfying G f = W_des."""
    g = np.asarray(grasp_matrix, dtype=float)
    w = np.asarray(desired_object_wrench, dtype=float)

    if g.shape != (6, 12):
        raise ValueError(f"grasp_matrix must have shape (6,12), got {g.shape}")
    if w.shape != (6,):
        raise ValueError("desired_object_wrench must have shape (6,)")
    if np.linalg.matrix_rank(g) < 6:
        raise ValueError("grasp_matrix must have full row rank")

    return g.T @ np.linalg.solve(g @ g.T, w)


def nullspace_projector(grasp_matrix: np.ndarray) -> np.ndarray:
    """Return N = I - G^+ G, the Euclidean projector into null(G)."""
    g = np.asarray(grasp_matrix, dtype=float)
    if g.shape != (6, 12):
        raise ValueError(f"grasp_matrix must have shape (6,12), got {g.shape}")
    if np.linalg.matrix_rank(g) < 6:
        raise ValueError("grasp_matrix must have full row rank")

    g_pinv = g.T @ np.linalg.inv(g @ g.T)
    return np.eye(12) - g_pinv @ g


def symmetric_side_compression(compression_force: float) -> np.ndarray:
    """Return a pure internal-compression wrench for symmetric side grasps.

    Wrench convention:
        W_i = [Fx, Fy, Fz, Mx, My, Mz]

    Forces are applied ON THE OBJECT:
        left inward normal  = +X
        right inward normal = -X

    Therefore a positive scalar F_int produces:
        W_Lx = +F_int
        W_Rx = -F_int
    with zero net object wrench when the contact normals pass through the
    object center.
    """
    fn = float(compression_force)
    if fn < 0.0:
        raise ValueError("compression_force must be non-negative")

    f = np.zeros(12, dtype=float)
    f[0] = fn
    f[6] = -fn
    return f


@dataclass
class InternalWrenchDecomposition:
    task_component: np.ndarray
    internal_component: np.ndarray
    reconstructed: np.ndarray
    internal_object_residual: float


def decompose_wrench(
    grasp_matrix: np.ndarray,
    stacked_wrench: Sequence[float],
    desired_object_wrench: Sequence[float],
) -> InternalWrenchDecomposition:
    """Decompose f into one task component plus a null-space component."""
    g = np.asarray(grasp_matrix, dtype=float)
    f = np.asarray(stacked_wrench, dtype=float)

    if f.shape != (12,):
        raise ValueError("stacked_wrench must have shape (12,)")

    task = minimum_norm_task_component(g, desired_object_wrench)
    internal = f - task
    residual = float(np.linalg.norm(g @ internal, ord=np.inf))

    return InternalWrenchDecomposition(
        task_component=task,
        internal_component=internal,
        reconstructed=task + internal,
        internal_object_residual=residual,
    )
