"""Simulator-independent Jacobian-transpose wrench-to-torque mapping.

Task04 convention:
    xdot = J(q) qdot
    tau_task = J(q).T @ W_des

The Jacobian and wrench must be expressed in the same frame and use the same
6D ordering. This project uses [linear, angular] for twists and therefore
[Fx, Fy, Fz, Mx, My, Mz] for wrenches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class JacobianTransposeWrenchMapper:
    """Map a Cartesian wrench to generalized joint torque using ``J.T @ W``."""

    dof: int = 7

    def compute(
        self,
        jacobian: np.ndarray,
        wrench: Sequence[float],
    ) -> np.ndarray:
        jacobian = np.asarray(jacobian, dtype=float)
        wrench = np.asarray(wrench, dtype=float)

        if jacobian.shape != (6, self.dof):
            raise ValueError(
                f"jacobian must have shape {(6, self.dof)}, got {jacobian.shape}"
            )
        if wrench.shape != (6,):
            raise ValueError(f"wrench must have shape (6,), got {wrench.shape}")
        if not np.isfinite(jacobian).all() or not np.isfinite(wrench).all():
            raise ValueError("jacobian and wrench must be finite")

        return jacobian.T @ wrench
