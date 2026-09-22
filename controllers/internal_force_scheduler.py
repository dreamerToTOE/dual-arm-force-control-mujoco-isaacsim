"""Task-dependent internal-compression scheduler for symmetric side grasps.

This module converts an object-level tangential load requirement into a desired
per-contact inward normal force:

    F_n,req = gamma * (|F_y,obj| + |F_z,obj|) / (2 * mu)

Assumptions for this teaching baseline:
- two symmetric side contacts;
- left/right share tangential object load approximately equally;
- contact normals are +/- X;
- the linear friction-pyramid model is |Fy| + |Fz| <= mu * Fn;
- object moments are handled by the downstream QP and are not used directly by
  this scalar scheduler.

The scheduler is NOT a force controller. It only produces a reference for the
QP. Hard contact/joint constraints still have priority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class InternalCompressionSchedule:
    requested_normal_force: float
    scheduled_normal_force: float
    minimum_normal_force: float
    maximum_normal_force: float
    safety_factor: float
    friction_coefficient: float
    tangential_object_load_l1: float
    clipped_low: bool
    clipped_high: bool


def schedule_symmetric_side_compression(
    desired_object_wrench: Sequence[float],
    *,
    friction_coefficient: float,
    safety_factor: float,
    normal_force_min: float = 0.0,
    normal_force_max: float = np.inf,
) -> InternalCompressionSchedule:
    """Return per-contact internal compression for a symmetric side grasp.

    desired_object_wrench order:
        [Fx, Fy, Fz, Mx, My, Mz]

    Only object tangential force components Fy and Fz enter this teaching
    scheduler because the side-contact normal axis is WORLD-X.
    """
    w = np.asarray(desired_object_wrench, dtype=float)
    if w.shape != (6,):
        raise ValueError("desired_object_wrench must have shape (6,)")

    mu = float(friction_coefficient)
    gamma = float(safety_factor)
    fn_min = float(normal_force_min)
    fn_max = float(normal_force_max)

    if mu <= 0.0:
        raise ValueError("friction_coefficient must be positive")
    if gamma < 1.0:
        raise ValueError("safety_factor should be >= 1.0")
    if fn_min < 0.0 or fn_max <= 0.0 or fn_min > fn_max:
        raise ValueError("invalid normal-force bounds")

    tangential_l1 = float(abs(w[1]) + abs(w[2]))

    minimum_required = tangential_l1 / (2.0 * mu)
    requested = gamma * minimum_required
    scheduled = float(np.clip(requested, fn_min, fn_max))

    return InternalCompressionSchedule(
        requested_normal_force=requested,
        scheduled_normal_force=scheduled,
        minimum_normal_force=minimum_required,
        maximum_normal_force=fn_max,
        safety_factor=gamma,
        friction_coefficient=mu,
        tangential_object_load_l1=tangential_l1,
        clipped_low=bool(requested < fn_min),
        clipped_high=bool(requested > fn_max),
    )
