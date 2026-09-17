"""Small simulator-independent PI controller for Task06 contact-force control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForcePIController:
    """PI controller that maps scalar force error to a positive push-force command.

    The experiment defines the sign convention separately. This class only
    computes a non-negative command magnitude from

        u = Kp * (F_des - F_meas) + Ki * integral(error dt)

    with simple anti-windup at the command limits.
    """

    kp: float
    ki: float
    command_min: float = 0.0
    command_max: float = 20.0
    integral_limit: float = 20.0
    integral: float = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        error = float(error)
        candidate_integral = float(
            np.clip(
                self.integral + error * dt,
                -abs(self.integral_limit),
                abs(self.integral_limit),
            )
        )
        unsaturated = self.kp * error + self.ki * candidate_integral
        command = float(np.clip(unsaturated, self.command_min, self.command_max))

        # Integrate when unsaturated, or when the current error would move the
        # controller back toward the admissible range.
        if (
            self.command_min < unsaturated < self.command_max
            or (unsaturated >= self.command_max and error < 0.0)
            or (unsaturated <= self.command_min and error > 0.0)
        ):
            self.integral = candidate_integral

        return command
