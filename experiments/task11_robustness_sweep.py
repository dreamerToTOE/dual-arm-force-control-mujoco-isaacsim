"""Task11 Stage 1: robustness audit under model mismatch.

This batch experiment separates two robustness questions:

1) Grip robustness:
   What if the true friction coefficient is lower than the controller assumes?

2) Object-task robustness:
   What if the true payload mass or an external disturbance differs from the
   controller model?

Two internal-force scheduling policies are compared:
- nominal:      mu_plan = 0.80
- conservative: mu_plan = 0.50

Both use the same object-wrench model (mass estimate = 1.0 kg).  This is
intentional: conservative internal compression can improve friction margin, but
it cannot fix an incorrect object-level wrench caused by mass/disturbance
mismatch.

All QPs still enforce the model-side constraints from Tasks 09-10.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.internal_force_scheduler import schedule_symmetric_side_compression
from controllers.internal_wrench import (
    minimum_norm_task_component,
    symmetric_side_compression,
)
from controllers.wrench_allocator_qp import (
    DualArmWrenchQP,
    side_grasp_contact_constraints,
)
from experiments.task08_dual_arm_shared_object import (
    GRAVITY,
    TORQUE_LIMITS,
    build_dual_scene,
    resolve_fr3_xml,
)
from experiments.task09_wrench_distribution_qp import task09_geometry
from experiments.task10_internal_force_sweep import (
    compression_from_wrench,
    torque_utilization,
)


MODEL_MASS = 1.0
SAFETY_FACTOR = 1.50
NORMAL_FORCE_MIN = 6.5
NORMAL_FORCE_MAX = 30.0
MOMENT_LIMITS = np.array([2.0, 2.0, 2.0])

POLICIES = {
    "nominal_mu": 0.80,
    "conservative_mu": 0.50,
}


@dataclass(frozen=True)
class RobustnessScenario:
    name: str
    true_mass: float
    true_mu: float
    desired_ay: float = 0.0
    desired_az: float = 0.0
    external_fy: float = 0.0
    external_fz: float = 0.0


SCENARIOS = [
    RobustnessScenario("nominal", 1.00, 0.80),
    RobustnessScenario("low_friction", 1.00, 0.50),
    RobustnessScenario("heavy_payload", 1.30, 0.80),
    RobustnessScenario("heavy_low_mu", 1.30, 0.50),
    RobustnessScenario("lateral_push", 1.00, 0.80, external_fy=3.0),
    RobustnessScenario(
        "combined_mismatch",
        1.20,
        0.55,
        desired_ay=2.0,
        desired_az=1.0,
        external_fy=2.0,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to MuJoCo Menagerie franka_fr3/scene.xml or fr3.xml",
    )
    return parser.parse_args()


def modeled_object_wrench(ay: float, az: float) -> np.ndarray:
    """Object wrench computed by the controller's nominal mass model.

    ay, az:
        desired WORLD-Y/Z acceleration [m/s^2].

    Output wrench order:
        [Fx, Fy, Fz, Mx, My, Mz].

    Fz includes gravity compensation.
    """
    return np.array(
        [
            0.0,
            MODEL_MASS * ay,
            MODEL_MASS * (GRAVITY + az),
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def true_required_robot_wrench(s: RobustnessScenario) -> np.ndarray:
    """Robot wrench required to realize the desired acceleration in true physics.

    The force balance is:

        m a = F_robot + F_external + F_gravity

    so:

        F_robot,y = m ay - F_external,y
        F_robot,z = m (g + az) - F_external,z

    Parameters:
        m: true object mass [kg]
        ay, az: desired accelerations [m/s^2]
        F_external: unmodeled external force acting on the object [N]
    """
    return np.array(
        [
            0.0,
            s.true_mass * s.desired_ay - s.external_fy,
            s.true_mass * (GRAVITY + s.desired_az) - s.external_fz,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def actual_acceleration_from_command(
    robot_wrench: np.ndarray,
    s: RobustnessScenario,
) -> tuple[float, float]:
    """Return actual ay/az produced by the command under true mass/disturbance."""
    fy = float(robot_wrench[1])
    fz = float(robot_wrench[2])

    ay_actual = (fy + s.external_fy) / s.true_mass
    az_actual = (fz + s.external_fz) / s.true_mass - GRAVITY
    return ay_actual, az_actual


def true_required_friction_utilization(
    true_required_wrench: np.ndarray,
    actual_compression: float,
    true_mu: float,
) -> float:
    """Required friction utilization if both contacts share tangential load equally.

    utilization =
        (|Fy_required| + |Fz_required|)
        / (2 * mu_true * Fn_actual)

    A value <= 1 means the scheduled compression has enough friction capacity
    for the true task under the same linear friction-pyramid model.
    """
    if actual_compression <= 1e-12 or true_mu <= 0.0:
        return float("inf")

    tangential_l1 = float(
        abs(true_required_wrench[1]) + abs(true_required_wrench[2])
    )
    return tangential_l1 / (2.0 * true_mu * actual_compression)


def save_csv(output_dir: Path, rows: list[dict]) -> Path:
    path = output_dir / "robustness_model_mismatch.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_plot(output_dir: Path, rows: list[dict]) -> Path:
    path = output_dir / "robustness_model_mismatch.png"

    scenarios = [s.name for s in SCENARIOS]
    x = np.arange(len(scenarios))
    width = 0.34

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    for offset, (policy_name, _mu_plan) in zip(
        (-width / 2, width / 2),
        POLICIES.items(),
    ):
        policy_rows = [r for r in rows if r["policy"] == policy_name]

        true_friction = 100.0 * np.array(
            [r["true_required_friction_utilization"] for r in policy_rows]
        )
        compression = np.array([r["actual_compression_N"] for r in policy_rows])
        accel_error = np.array([r["accel_error_norm_mps2"] for r in policy_rows])
        torque = 100.0 * np.array(
            [r["max_torque_utilization"] for r in policy_rows]
        )

        axes[0].bar(
            x + offset,
            true_friction,
            width,
            label=policy_name,
        )
        axes[1].bar(
            x + offset,
            compression,
            width,
            label=policy_name,
        )
        axes[2].bar(
            x + offset,
            accel_error,
            width,
            label=policy_name,
        )
        axes[3].bar(
            x + offset,
            torque,
            width,
            label=policy_name,
        )

    axes[0].axhline(100.0, linestyle="--", label="true friction boundary")
    axes[0].set_ylabel("true required friction\nutilization [%]")
    axes[0].set_title("Task11 Stage 1: nominal model vs true physics")
    axes[0].grid(True, axis="y")
    axes[0].legend()

    axes[1].axhline(NORMAL_FORCE_MAX, linestyle="--", label="normal-force max")
    axes[1].set_ylabel("realized internal\ncompression [N]")
    axes[1].grid(True, axis="y")
    axes[1].legend()

    axes[2].set_ylabel("acceleration error\n[m/s^2]")
    axes[2].grid(True, axis="y")
    axes[2].legend()

    axes[3].axhline(100.0, linestyle="--", label="torque limit")
    axes[3].set_ylabel("max joint torque\nutilization [%]")
    axes[3].grid(True, axis="y")
    axes[3].legend()

    axes[3].set_xticks(x, scenarios, rotation=20)
    axes[3].set_xlabel("true scenario")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    output_dir = ROOT / "outputs" / "task11" / "stage1"
    output_dir.mkdir(parents=True, exist_ok=True)

    fr3_xml = resolve_fr3_xml(model_path)
    scene_xml = output_dir / "task11_stage1_geometry.xml"
    build_dual_scene(fr3_xml, scene_xml)

    (
        _model,
        _data,
        _left,
        _right,
        _r_left,
        _r_right,
        grasp_matrix,
        jl,
        jr,
        gl,
        gr,
    ) = task09_geometry(scene_xml)

    print("\n=== Task11 Stage 1: robustness under model mismatch ===")
    print(f"Controller mass model   : {MODEL_MASS:.2f} kg")
    print(f"Safety factor gamma     : {SAFETY_FACTOR:.2f}")
    print(
        f"Normal force bounds     : "
        f"{NORMAL_FORCE_MIN:.2f} .. {NORMAL_FORCE_MAX:.2f} N/contact"
    )
    print("Policies:")
    for name, mu_plan in POLICIES.items():
        print(f"  {name:<18}: planned mu = {mu_plan:.2f}")

    rows: list[dict] = []

    print(
        "\nscenario            policy              m_true  mu_true  "
        "Fn[N]  true friction[%]  accel err[m/s2]  torque[%]  grip/task"
    )
    print("-" * 120)

    for policy_name, mu_plan in POLICIES.items():
        contact_a, contact_b = side_grasp_contact_constraints(
            friction_coefficient=mu_plan,
            normal_force_max=NORMAL_FORCE_MAX,
            moment_limits=MOMENT_LIMITS,
        )
        allocator = DualArmWrenchQP(grasp_matrix)

        for scenario in SCENARIOS:
            w_model = modeled_object_wrench(
                scenario.desired_ay,
                scenario.desired_az,
            )

            schedule = schedule_symmetric_side_compression(
                w_model,
                friction_coefficient=mu_plan,
                safety_factor=SAFETY_FACTOR,
                normal_force_min=NORMAL_FORCE_MIN,
                normal_force_max=NORMAL_FORCE_MAX,
            )

            f_task = minimum_norm_task_component(grasp_matrix, w_model)
            f_ref = (
                f_task
                + symmetric_side_compression(schedule.scheduled_normal_force)
            )

            result = allocator.solve(
                w_model,
                reference_wrench=f_ref,
                jacobian_left=jl,
                jacobian_right=jr,
                gravity_left=gl,
                gravity_right=gr,
                torque_limits_left=TORQUE_LIMITS,
                torque_limits_right=TORQUE_LIMITS,
                linear_inequality_A=contact_a,
                linear_inequality_b=contact_b,
            )

            if not result.success:
                raise RuntimeError(
                    f"Task11 QP failed for {policy_name}/{scenario.name}: "
                    f"{result.status}"
                )

            stacked_wrench = np.concatenate(
                (result.wrench_left, result.wrench_right)
            )
            compression = compression_from_wrench(stacked_wrench)

            w_true_req = true_required_robot_wrench(scenario)
            true_friction_util = true_required_friction_utilization(
                w_true_req,
                compression,
                scenario.true_mu,
            )

            w_robot = result.object_wrench_reconstructed
            ay_actual, az_actual = actual_acceleration_from_command(
                w_robot,
                scenario,
            )
            ay_err = ay_actual - scenario.desired_ay
            az_err = az_actual - scenario.desired_az
            accel_error_norm = float(np.hypot(ay_err, az_err))

            torque_l, torque_r = torque_utilization(result)
            torque_max = max(torque_l, torque_r)

            wrench_error = w_robot - w_true_req
            wrench_force_error_norm = float(
                np.linalg.norm(wrench_error[:3])
            )

            grip_ok = bool(true_friction_util <= 1.0 + 1e-9)
            task_ok = bool(accel_error_norm <= 0.25)

            rows.append(
                {
                    "scenario": scenario.name,
                    "policy": policy_name,
                    "mass_model_kg": MODEL_MASS,
                    "true_mass_kg": scenario.true_mass,
                    "mu_plan": mu_plan,
                    "true_mu": scenario.true_mu,
                    "desired_ay_mps2": scenario.desired_ay,
                    "desired_az_mps2": scenario.desired_az,
                    "external_fy_N": scenario.external_fy,
                    "external_fz_N": scenario.external_fz,
                    "scheduled_compression_N": schedule.scheduled_normal_force,
                    "actual_compression_N": compression,
                    "true_required_friction_utilization": true_friction_util,
                    "true_grip_ok": grip_ok,
                    "robot_force_error_norm_N": wrench_force_error_norm,
                    "ay_actual_mps2": ay_actual,
                    "az_actual_mps2": az_actual,
                    "accel_error_norm_mps2": accel_error_norm,
                    "task_tracking_ok": task_ok,
                    "max_torque_utilization": torque_max,
                    "qp_equality_residual": result.equality_residual,
                    "qp_contact_violation": result.max_linear_inequality_violation,
                    "qp_torque_violation": result.max_torque_violation,
                }
            )

            print(
                f"{scenario.name:<19} "
                f"{policy_name:<19} "
                f"{scenario.true_mass:6.2f}  "
                f"{scenario.true_mu:7.2f}  "
                f"{compression:5.2f}  "
                f"{100*true_friction_util:15.1f}  "
                f"{accel_error_norm:16.3f}  "
                f"{100*torque_max:8.1f}  "
                f"{'OK' if grip_ok else 'SLIP-RISK'}/"
                f"{'OK' if task_ok else 'TASK-ERROR'}"
            )

    plot_path = save_plot(output_dir, rows)
    csv_path = save_csv(output_dir, rows)

    print("\nInterpretation:")
    print(
        "Low true friction mainly threatens grasp feasibility. "
        "A conservative friction model increases internal compression and "
        "can recover margin."
    )
    print(
        "Payload-mass error or unknown external force changes the true object "
        "wrench requirement. More internal compression alone does NOT correct "
        "that object-level force error."
    )
    print(
        "Therefore robustness needs two layers: grasp-margin robustness and "
        "object-wrench feedback/estimation."
    )

    print(f"\nPlot : {plot_path}")
    print(f"CSV  : {csv_path}")
    print(f"Scene: {scene_xml}")


if __name__ == "__main__":
    main()
