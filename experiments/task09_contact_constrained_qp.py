"""Task09 Stage 2: physically admissible side-contact wrench constraints.

Stage 1 showed that G f = W_des plus joint-torque limits can produce a
mathematically valid but physically odd wrench split.  Stage 2 adds a simplified
side-grasp contact model:

    left  inward normal: +WORLD-X
    right inward normal: -WORLD-X

For each contact:
    F_n >= 0
    F_n <= F_n,max
    |F_y| + |F_z| <= mu F_n
    |M_k| <= M_k,max

The friction relation is a conservative linear friction-pyramid approximation,
so the optimization remains a QP with linear constraints.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


MU = 0.80
NORMAL_FORCE_MAX = 30.0
MOMENT_LIMITS = np.array([2.0, 2.0, 2.0])

# Deliberately impossible contact capacity for the final case:
# each contact can provide at most mu*2 = 1.6 N vertical friction,
# so two contacts can provide at most 3.2 N < 9.81 N.
IMPOSSIBLE_NORMAL_FORCE_MAX = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to MuJoCo Menagerie franka_fr3/scene.xml or fr3.xml",
    )
    return parser.parse_args()


def stack_wrench(result) -> np.ndarray:
    return np.concatenate((result.wrench_left, result.wrench_right))


def contact_metrics(wrench_vector: np.ndarray, mu: float) -> dict[str, float]:
    f = np.asarray(wrench_vector, dtype=float)
    wl = f[:6]
    wr = f[6:]

    fn_l = float(wl[0])
    fn_r = float(-wr[0])

    tangential_l = float(abs(wl[1]) + abs(wl[2]))
    tangential_r = float(abs(wr[1]) + abs(wr[2]))

    util_l = (
        tangential_l / (mu * fn_l)
        if fn_l > 1e-9
        else (0.0 if tangential_l < 1e-9 else float("inf"))
    )
    util_r = (
        tangential_r / (mu * fn_r)
        if fn_r > 1e-9
        else (0.0 if tangential_r < 1e-9 else float("inf"))
    )

    return {
        "normal_left": fn_l,
        "normal_right": fn_r,
        "friction_util_left": util_l,
        "friction_util_right": util_r,
        "vertical_left": float(wl[2]),
        "vertical_right": float(wr[2]),
    }


def find_feasible_torque_bottleneck(
    allocator: DualArmWrenchQP,
    desired_wrench: np.ndarray,
    baseline,
    jl: np.ndarray,
    jr: np.ndarray,
    gl: np.ndarray,
    gr: np.ndarray,
    contact_a: np.ndarray,
    contact_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float, object]:
    """Find a deterministic left-arm limit that is tighter but still feasible."""
    if baseline.tau_left is None:
        raise RuntimeError("Physical baseline did not return left joint torques")

    baseline_tau = np.abs(baseline.tau_left)
    utilization = baseline_tau / TORQUE_LIMITS
    candidate_joints = list(np.argsort(-utilization))

    best = None

    for joint_index in candidate_joints:
        baseline_abs = float(baseline_tau[joint_index])
        if baseline_abs < 0.5:
            continue

        # Try meaningful reductions first.  Keep the strongest feasible case.
        for factor in (0.55, 0.65, 0.75, 0.85, 0.92, 0.96):
            left_limits = TORQUE_LIMITS.copy()
            right_limits = TORQUE_LIMITS.copy()
            left_limits[joint_index] = baseline_abs * factor

            trial = allocator.solve(
                desired_wrench,
                jacobian_left=jl,
                jacobian_right=jr,
                gravity_left=gl,
                gravity_right=gr,
                torque_limits_left=left_limits,
                torque_limits_right=right_limits,
                linear_inequality_A=contact_a,
                linear_inequality_b=contact_b,
            )

            if not trial.success:
                continue

            load_change = abs(
                trial.wrench_left[2] - baseline.wrench_left[2]
            )
            candidate = (
                load_change,
                left_limits.copy(),
                right_limits.copy(),
                joint_index,
                baseline_abs,
                trial,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        raise RuntimeError(
            "Could not find any feasible tightened left-arm torque limit "
            "under the contact-wrench constraints."
        )

    _, left_limits, right_limits, joint_index, baseline_abs, trial = best
    return left_limits, right_limits, joint_index, baseline_abs, trial


def save_csv(output_dir: Path, cases: dict[str, object], mu: float) -> Path:
    path = output_dir / "contact_constrained_wrench_cases.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        headers = ["case", "success", "status"]
        headers += [f"WL_{x}" for x in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]
        headers += [f"WR_{x}" for x in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]
        headers += [
            "Fn_left",
            "Fn_right",
            "friction_util_left",
            "friction_util_right",
            "eq_residual",
            "torque_violation",
            "contact_violation",
        ]
        writer.writerow(headers)

        for name, result in cases.items():
            cm = contact_metrics(stack_wrench(result), mu)
            writer.writerow(
                [
                    name,
                    result.success,
                    result.status,
                    *result.wrench_left,
                    *result.wrench_right,
                    cm["normal_left"],
                    cm["normal_right"],
                    cm["friction_util_left"],
                    cm["friction_util_right"],
                    result.equality_residual,
                    result.max_torque_violation,
                    result.max_linear_inequality_violation,
                ]
            )
    return path


def save_plot(
    output_dir: Path,
    physical_baseline,
    unconstrained_limited,
    physical_limited,
    left_limits: np.ndarray,
) -> Path:
    path = output_dir / "contact_constrained_wrench_qp.png"

    names = ["physical baseline", "no contact\nconstraints", "physical limited"]
    results = [physical_baseline, unconstrained_limited, physical_limited]
    x = np.arange(len(results))
    width = 0.34

    fz_left = [r.wrench_left[2] for r in results]
    fz_right = [r.wrench_right[2] for r in results]
    fn_left = [r.wrench_left[0] for r in results]
    fn_right = [-r.wrench_right[0] for r in results]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    axes[0].bar(x - width / 2, fz_left, width, label="Left Fz")
    axes[0].bar(x + width / 2, fz_right, width, label="Right Fz")
    axes[0].axhline(0.0)
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("vertical force [N]")
    axes[0].set_title("Task09 Stage 2: contact constraints remove nonphysical wrench splits")
    axes[0].grid(True, axis="y")
    axes[0].legend()

    axes[1].bar(x - width / 2, fn_left, width, label="Left inward normal")
    axes[1].bar(x + width / 2, fn_right, width, label="Right inward normal")
    axes[1].axhline(0.0)
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("normal compression [N]")
    axes[1].grid(True, axis="y")
    axes[1].legend()

    assert physical_limited.tau_left is not None
    assert physical_limited.tau_right is not None
    joints = np.arange(1, 8)
    axes[2].plot(joints, np.abs(physical_limited.tau_left), marker="o", label="|tau_L|")
    axes[2].plot(joints, left_limits, linestyle="--", label="left torque limits")
    axes[2].plot(joints, np.abs(physical_limited.tau_right), marker="o", label="|tau_R|")
    axes[2].set_xlabel("joint index")
    axes[2].set_ylabel("joint torque [N m]")
    axes[2].grid(True)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def print_case(name: str, result, mu: float) -> None:
    cm = contact_metrics(stack_wrench(result), mu)
    print(f"\n--- {name} ---")
    print("success                 :", result.success)
    print("W_L                     :", np.array2string(result.wrench_left, precision=4))
    print("W_R                     :", np.array2string(result.wrench_right, precision=4))
    print(
        f"vertical load split     : L={cm['vertical_left']:.4f} N, "
        f"R={cm['vertical_right']:.4f} N"
    )
    print(
        f"inward normal force     : L={cm['normal_left']:.4f} N, "
        f"R={cm['normal_right']:.4f} N"
    )
    print(
        f"friction utilization    : L={100*cm['friction_util_left']:.2f} %, "
        f"R={100*cm['friction_util_right']:.2f} %"
    )
    print(f"equality residual       : {result.equality_residual:.3e}")
    print(f"max torque violation    : {result.max_torque_violation:.3e} N m")
    print(
        f"max contact violation   : "
        f"{result.max_linear_inequality_violation:.3e}"
    )
    if not result.success:
        print("solver status           :", result.status)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    output_dir = ROOT / "outputs" / "task09" / "stage2"
    output_dir.mkdir(parents=True, exist_ok=True)

    fr3_xml = resolve_fr3_xml(model_path)
    scene_xml = output_dir / "task09_stage2_geometry.xml"
    build_dual_scene(fr3_xml, scene_xml)

    (
        _model,
        _data,
        _left,
        _right,
        r_left,
        r_right,
        grasp_matrix,
        jl,
        jr,
        gl,
        gr,
    ) = task09_geometry(scene_xml)

    allocator = DualArmWrenchQP(grasp_matrix)
    desired_wrench = np.array(
        [0.0, 0.0, BOX_MASS * GRAVITY, 0.0, 0.0, 0.0],
        dtype=float,
    )

    contact_a, contact_b = side_grasp_contact_constraints(
        friction_coefficient=MU,
        normal_force_max=NORMAL_FORCE_MAX,
        moment_limits=MOMENT_LIMITS,
    )

    physical_baseline = allocator.solve(
        desired_wrench,
        jacobian_left=jl,
        jacobian_right=jr,
        gravity_left=gl,
        gravity_right=gr,
        torque_limits_left=TORQUE_LIMITS,
        torque_limits_right=TORQUE_LIMITS,
        linear_inequality_A=contact_a,
        linear_inequality_b=contact_b,
    )
    if not physical_baseline.success:
        raise RuntimeError(
            f"Physical-contact baseline QP failed: {physical_baseline.status}"
        )

    (
        left_limits,
        right_limits,
        limited_joint,
        baseline_joint_tau,
        physical_limited,
    ) = find_feasible_torque_bottleneck(
        allocator,
        desired_wrench,
        physical_baseline,
        jl,
        jr,
        gl,
        gr,
        contact_a,
        contact_b,
    )

    # Solve the SAME torque-limited problem without contact constraints.  This
    # makes the role of the new physical constraints directly visible.
    unconstrained_limited = allocator.solve(
        desired_wrench,
        jacobian_left=jl,
        jacobian_right=jr,
        gravity_left=gl,
        gravity_right=gr,
        torque_limits_left=left_limits,
        torque_limits_right=right_limits,
    )

    impossible_a, impossible_b = side_grasp_contact_constraints(
        friction_coefficient=MU,
        normal_force_max=IMPOSSIBLE_NORMAL_FORCE_MAX,
        moment_limits=MOMENT_LIMITS,
    )
    impossible_contact = allocator.solve(
        desired_wrench,
        jacobian_left=jl,
        jacobian_right=jr,
        gravity_left=gl,
        gravity_right=gr,
        torque_limits_left=TORQUE_LIMITS,
        torque_limits_right=TORQUE_LIMITS,
        linear_inequality_A=impossible_a,
        linear_inequality_b=impossible_b,
    )

    print("\n=== Task09 Stage 2: physical contact-wrench constraints ===")
    print("Contact geometry        : symmetric side grasp")
    print("Left inward normal      : +WORLD-X")
    print("Right inward normal     : -WORLD-X")
    print(f"Friction coefficient mu : {MU:.2f}")
    print(f"Normal force max        : {NORMAL_FORCE_MAX:.2f} N per contact")
    print("Moment limits           :", MOMENT_LIMITS, "N m")
    print("Friction approximation  : |Fy| + |Fz| <= mu * Fn")
    print("Object desired wrench   :", np.array2string(desired_wrench, precision=4))
    print("Left contact offset     :", np.array2string(r_left, precision=4), "m")
    print("Right contact offset    :", np.array2string(r_right, precision=4), "m")

    print_case("Case A: physical-contact baseline", physical_baseline, MU)

    print("\nTorque bottleneck chosen automatically")
    print(f"limited left joint      : J{limited_joint + 1}")
    print(f"baseline |tau|          : {baseline_joint_tau:.4f} N m")
    print(f"new left torque limit   : {left_limits[limited_joint]:.4f} N m")

    print_case(
        "Case B1: same torque limit WITHOUT contact constraints",
        unconstrained_limited,
        MU,
    )
    print_case(
        "Case B2: same torque limit WITH contact constraints",
        physical_limited,
        MU,
    )

    print(
        "\n--- Case C: deliberately insufficient contact capacity ---"
    )
    print(
        f"normal force max        : {IMPOSSIBLE_NORMAL_FORCE_MAX:.2f} N per contact"
    )
    print(
        f"maximum vertical support: "
        f"{2.0 * MU * IMPOSSIBLE_NORMAL_FORCE_MAX:.2f} N"
    )
    print(f"required vertical force : {BOX_MASS * GRAVITY:.2f} N")
    print("success                 :", impossible_contact.success)
    print("solver status           :", impossible_contact.status)
    print(
        f"max contact violation   : "
        f"{impossible_contact.max_linear_inequality_violation:.3e}"
    )
    if impossible_contact.success:
        print("WARNING: expected this contact-capacity case to be infeasible.")
    else:
        print("Expected result         : infeasible / rejected clearly")

    cases = {
        "physical_baseline": physical_baseline,
        "unconstrained_same_torque_limit": unconstrained_limited,
        "physical_same_torque_limit": physical_limited,
        "insufficient_contact_capacity": impossible_contact,
    }

    plot_path = save_plot(
        output_dir,
        physical_baseline,
        unconstrained_limited,
        physical_limited,
        left_limits,
    )
    csv_path = save_csv(output_dir, cases, MU)

    print(f"\nPlot : {plot_path}")
    print(f"CSV  : {csv_path}")
    print(f"Scene: {scene_xml}")
    print(
        "\nInterpretation target: torque constraints decide how the load may be "
        "redistributed, while contact-wrench constraints decide which of those "
        "mathematical allocations are physically admissible."
    )


if __name__ == "__main__":
    main()
