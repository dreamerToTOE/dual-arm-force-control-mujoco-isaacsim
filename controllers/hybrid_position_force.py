"""Simulator-independent hybrid position/force selection logic.

Task07 separates task-space directions with complementary 6D selection
matrices:

    W_cmd = S_p W_pos + S_f W_force
    S_f = I - S_p

For the teaching experiment:
    X, Y, Rx, Ry, Rz -> position/impedance branch
    Z                -> force branch

The class only combines already-computed wrenches.  It does not depend on
MuJoCo or Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _mask6(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (6,):
        raise ValueError(f"{name} must have shape (6,), got {arr.shape}")
    if not np.all(np.logical_or(np.isclose(arr, 0.0), np.isclose(arr, 1.0))):
        raise ValueError(f"{name} entries must be 0 or 1")
    return arr


@dataclass
class HybridSelection6D:
    """Complementary 6D position/force selection matrices."""

    position_mask: np.ndarray

    @classmethod
    def from_position_mask(cls, position_mask: Sequence[float]) -> "HybridSelection6D":
        return cls(position_mask=_mask6(position_mask, "position_mask"))

    @property
    def force_mask(self) -> np.ndarray:
        return 1.0 - self.position_mask

    @property
    def S_position(self) -> np.ndarray:
        return np.diag(self.position_mask)

    @property
    def S_force(self) -> np.ndarray:
        return np.diag(self.force_mask)

    def combine(
        self,
        position_wrench: Sequence[float],
        force_wrench: Sequence[float],
    ) -> np.ndarray:
        w_pos = np.asarray(position_wrench, dtype=float)
        w_force = np.asarray(force_wrench, dtype=float)
        if w_pos.shape != (6,) or w_force.shape != (6,):
            raise ValueError("position_wrench and force_wrench must have shape (6,)")
        return self.S_position @ w_pos + self.S_force @ w_force

    def validate(self) -> dict[str, float]:
        sp = self.S_position
        sf = self.S_force
        eye = np.eye(6)
        return {
            "complement_error": float(np.max(np.abs(sp + sf - eye))),
            "overlap_error": float(np.max(np.abs(sp @ sf))),
            "position_idempotence_error": float(np.max(np.abs(sp @ sp - sp))),
            "force_idempotence_error": float(np.max(np.abs(sf @ sf - sf))),
        }
