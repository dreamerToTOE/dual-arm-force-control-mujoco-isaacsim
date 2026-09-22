"""Task10 Stage 2: task-dependent internal-force scheduling.

Instead of manually choosing a fixed internal-force reference, derive the
desired symmetric side compression from the current object-level task wrench.

For the teaching side-grasp model:

    Fn_req = gamma * (|Fy_obj| + |Fz_obj|) / (2 * mu)

Then:

    f_ref = f_task + f_internal_des(Fn_req)

and the Task09 QP enforces:
- G f = W_obj_des,
- joint torque limits,
- unilateral side-contact constraints,
- friction-pyramid constraints,
- contact moment limits.

This is reference scheduling, not yet measured-force feedback control.
"""

from __future__ import annotations

import argparse
import csv
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
    BOX_MASS,
    GRAVITY,
    TORQUE_LIMITS,
    build_dual_scene,
    resolve_fr3_xml,
)
from experiments.task09_wrench_distribution_qp import task09_geometry
from experiments.task10_internal_force_sweep import (
    compression_from_wrench,
    friction_utilization,
    torque_utilization,
)


MU = 0.80
SAFETY_FACTOR = 1.50
NORMAL_FORCE_MIN = 6.5
NORMAL_FORCE_MAX = 30.0
MOMENT_LIMITS = np.array([2.0, 2.0, 2.0])

SCENARIOS = [
    ("downward_accel", 0.0, -2.0),
    ("static_hold", 0.0, 0.0),
    ("upward_accel", 0.0, 2.0),
    ("lateral_move", 2.0, 0.0),
    ("combined_accel", 2.0, 2.0),
    ("aggressive_combined", 6.0, 4.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to MuJoCo Menagerie franka_fr3/scene.xml or fr3.xml",
    )
    return parser.parse_args()


def desired_wrench_from_acceleration(ay: float, az: float) -> np.ndarray:
    """Return object wrench for desired WORLD-Y/Z accelerations.

    ay, az: desired translational acceleration [m/s^2].
    Positive Z is upward, so Fz = m(g + az).
    """
    return np.array(
        [
            0.0,
            BOX_MASS * ay,
            BOX_MASS * (GRAVITY + az),
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def stacked(result) -> np.ndarray:
    return np.concatenate((result.wrench_left, result.wrench_right))


def save_csv(output_dir: Path, rows: list[dict]) -> Path:
    path = output_dir / "task_dependent_internal_force.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_plot(output_dir: Path, rows: list[dict]) -> Path:
    path = output_dir / "task_dependent_internal_force.png"

    labels = [row["scenario"] for row in rows]
    x = np.arange(len(rows))

    minimum = np.array([row["minimum_fn_N"] for row in rows])
    scheduled = np.array([row["scheduled_fn_N"] for row in rows])
    actual = np.array([row["actual_fn_N"] for row in rows])

    friction_l = 100.0 * np.array([row["friction_util_left"] for row in rows])
    friction_r = 100.0 * np.array([row["friction_util_right"] for row in rows])

    torque_l = 100.0 * np.array([row["torque_util_left"] for row in rows])
    torque_r = 100.0 * np.array([row["torque_util_right"] for row in rows])

    object_fy = np.array([row["object_Fy_N"] for row in rows])
    object_fz = np.array([row["object_Fz_N"] for row in rows])

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)

    axes[0].plot(x, minimum, marker="o", label="minimum for friction feasibility")
    axes[0].plot(x, scheduled, marker="o", label="scheduled with safety factor")
    axes[0].plot(x, actual, marker="o", label="QP realized compression")
    axes[0].axhline(NORMAL_FORCE_MAX, linestyle="--", label="contact normal max")
    axes[0].set_ylabel("normal compression [N]")
    axes[0].set_title("Task10 Stage 2: internal force scheduled from task demand")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(x, friction_l, marker="o", label="left friction utilization")
    axes[1].plot(x, friction_r, marker="o", label="right friction utilization")
    axes[1].axhline(
        100.0 / SAFETY_FACTOR,
        linestyle="--",
        label="nominal target utilization",
    )
    axes[1].axhline(100.0, linestyle=":", label="friction boundary")
    axes[1].set_ylabel("friction utilization [%]")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(x, torque_l, marker="o", label="left max torque utilization")
    axes[2].plot(x, torque_r, marker="o", label="right max torque utilization")
    axes[2].axhline(100.0, linestyle="--", label="torque limit")
    axes[2].set_ylabel("joint torque utilization [%]")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(x, object_fy, marker="o", label="reconstructed object Fy")
    axes[3].plot(x, object_fz, marker="o", label="reconstructed object Fz")
    axes[3].set_ylabel("object force [N]")
    axes[3].grid(True)
    axes[3].legend()

    axes[3].set_xticks(x, labels, rotation=20)
    axes[3].set_xlabel("task phase")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    output_dir = ROOT / "outputs" / "task10" / "stage2"
    output_dir.mkdir(parents=True, exist_ok=True)

    fr3_xml = resolve_fr3_xml(model_path)
    scene_xml = output_dir / "task10_stage2_geometry.xml"
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

    allocator = DualArmWrenchQP(grasp_matrix)
    contact_a, contact_b = side_grasp_contact_constraints(
        friction_coefficient=MU,
        normal_force_max=NORMAL_FORCE_MAX,
        moment_limits=MOMENT_LIMITS,
    )

    print("\n=== Task10 Stage 2: task-dependent internal-force scheduling ===")
    print(f"Object mass             : {BOX_MASS:.2f} kg")
    print(f"Friction coefficient mu : {MU:.2f}")
    print(f"Safety factor gamma     : {SAFETY_FACTOR:.2f}")
    print(
        f"Normal-force range      : "
        f"{NORMAL_FORCE_MIN:.2f} .. {NORMAL_FORCE_MAX:.2f} N/contact"
    )
    print(
        "Scheduling law         : "
        "Fn=gamma*(|Fy_obj|+|Fz_obj|)/(2*mu), clipped to allowed range"
    )

    rows: list[dict] = []

    print(
        "\nscenario             ay     az    Fy_obj  Fz_obj  "
        "Fn_min  Fn_sched  Fn_actual  friction L/R[%]  torque L/R[%]"
    )
    print("-" * 126)

    for name, ay, az in SCENARIOS:
        w_des = desired_wrench_from_acceleration(ay, az)

        schedule = schedule_symmetric_side_compression(
            w_des,
            friction_coefficient=MU,
            safety_factor=SAFETY_FACTOR,
            normal_force_min=NORMAL_FORCE_MIN,
            normal_force_max=NORMAL_FORCE_MAX,
        )

        f_task = minimum_norm_task_component(grasp_matrix, w_des)
        f_int_des = symmetric_side_compression(schedule.scheduled_normal_force)
        f_ref = f_task + f_int_des

        result = allocator.solve(
            w_des,
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
                f"Task10 Stage 2 QP failed in scenario {name}: {result.status}"
            )

        f = stacked(result)
        actual_fn = compression_from_wrench(f)
        friction_l, friction_r = friction_utilization(f, MU)
        torque_l, torque_r = torque_utilization(result)
        w_obj = result.object_wrench_reconstructed

        rows.append(
            {
                "scenario": name,
                "ay_mps2": ay,
                "az_mps2": az,
                "desired_Fy_N": float(w_des[1]),
                "desired_Fz_N": float(w_des[2]),
                "minimum_fn_N": schedule.minimum_normal_force,
                "requested_fn_N": schedule.requested_normal_force,
                "scheduled_fn_N": schedule.scheduled_normal_force,
                "actual_fn_N": actual_fn,
                "scheduler_clipped_low": schedule.clipped_low,
                "scheduler_clipped_high": schedule.clipped_high,
                "friction_util_left": friction_l,
                "friction_util_right": friction_r,
                "torque_util_left": torque_l,
                "torque_util_right": torque_r,
                "object_Fy_N": float(w_obj[1]),
                "object_Fz_N": float(w_obj[2]),
                "object_wrench_error_inf": result.equality_residual,
                "contact_violation": result.max_linear_inequality_violation,
                "torque_violation_Nm": result.max_torque_violation,
            }
        )

        print(
            f"{name:<20} "
            f"{ay:5.1f}  {az:5.1f}  "
            f"{w_des[1]:6.2f}  {w_des[2]:6.2f}  "
            f"{schedule.minimum_normal_force:6.2f}  "
            f"{schedule.scheduled_normal_force:8.2f}  "
            f"{actual_fn:9.2f}  "
            f"{100*friction_l:6.1f}/{100*friction_r:6.1f}  "
            f"{100*torque_l:6.1f}/{100*torque_r:6.1f}"
        )

    plot_path = save_plot(output_dir, rows)
    csv_path = save_csv(output_dir, rows)

    print("\nInterpretation:")
    print(
        "The internal-force reference is derived from task demand rather than "
        "chosen manually."
    )
    print(
        "With gamma=1.5, symmetric cases should operate near or below about "
        "66.7% friction utilization unless another QP constraint changes the split."
    )
    print(
        "This is still feed-forward/reference scheduling; measured-force feedback "
        "and uncertainty adaptation belong to a later robustness layer."
    )

    print(f"\nPlot : {plot_path}")
    print(f"CSV  : {csv_path}")
    print(f"Scene: {scene_xml}")


if __name__ == "__main__":
    main()
