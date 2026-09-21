"""Task09 Stage 1: static dual-arm wrench distribution with QP.

The experiment reuses the corrected Task08 dual-FR3 HOME geometry but does not
yet run a dynamic carry simulation.  It answers three questions:

1) With only G f = W_des, what allocation does a minimum-norm solution choose?
2) If one left-arm joint is artificially tightened, can the QP redistribute
   wrench while preserving the same object wrench?
3) If both arms are constrained unrealistically tightly, is infeasibility
   reported clearly?

All wrench quantities are expressed in WORLD coordinates and moments are taken
about the SharedBox center.
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

from controllers.wrench_allocator_qp import DualArmWrenchQP, dual_grasp_matrix
from experiments.task08_dual_arm_shared_object import (
    BOX_MASS,
    GRAVITY,
    TORQUE_LIMITS,
    build_dual_scene,
    make_adapters,
    object_pose,
    resolve_fr3_xml,
    set_home,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to MuJoCo Menagerie franka_fr3/scene.xml or fr3.xml",
    )
    return parser.parse_args()


def task09_geometry(scene_xml: Path):
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    left, right = make_adapters(model, data)
    set_home(left, right)

    p_obj, r_obj = object_pose(model, data)
    p_left, _ = left.get_tcp_pose(frame="world")
    p_right, _ = right.get_tcp_pose(frame="world")

    # Express offsets in WORLD.  Task08's initial object orientation is identity,
    # so WORLD and object axes coincide here; keeping WORLD also matches J_world.
    r_left = p_left - p_obj
    r_right = p_right - p_obj

    g = dual_grasp_matrix(r_left, r_right)
    jl = left.get_jacobian(frame="world")
    jr = right.get_jacobian(frame="world")
    gl = left.get_gravity()
    gr = right.get_gravity()

    return model, data, left, right, r_left, r_right, g, jl, jr, gl, gr


def choose_demo_torque_limit(
    allocator: DualArmWrenchQP,
    desired_wrench: np.ndarray,
    jl: np.ndarray,
    jr: np.ndarray,
    gl: np.ndarray,
    gr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Create one meaningful left-arm bottleneck from the baseline allocation."""
    baseline = allocator.solve(
        desired_wrench,
        jacobian_left=jl,
        jacobian_right=jr,
        gravity_left=gl,
        gravity_right=gr,
        torque_limits_left=TORQUE_LIMITS,
        torque_limits_right=TORQUE_LIMITS,
    )
    if not baseline.success:
        raise RuntimeError(f"Baseline torque-constrained QP failed: {baseline.status}")

    tau_left = baseline.tau_left
    assert tau_left is not None

    utilization = np.abs(tau_left) / TORQUE_LIMITS
    joint_index = int(np.argmax(utilization))

    left_limits = TORQUE_LIMITS.copy()
    # Tighten this joint below the baseline demand, but never below the gravity
    # compensation magnitude by a vanishing amount.  A search below will relax
    # it until a feasible redistribution is found.
    baseline_abs = float(abs(tau_left[joint_index]))
    candidate = max(0.20, 0.55 * baseline_abs)

    right_limits = TORQUE_LIMITS.copy()

    # Find the tightest demonstrably feasible limit from a small deterministic
    # sequence.  This avoids hand-tuning to one exact MuJoCo minor version.
    for scale in (1.0, 1.15, 1.35, 1.6, 2.0, 3.0):
        left_limits[joint_index] = candidate * scale
        trial = allocator.solve(
            desired_wrench,
            jacobian_left=jl,
            jacobian_right=jr,
            gravity_left=gl,
            gravity_right=gr,
            torque_limits_left=left_limits,
            torque_limits_right=right_limits,
        )
        if trial.success and left_limits[joint_index] < baseline_abs * 0.98:
            return left_limits.copy(), right_limits, joint_index, baseline_abs

    raise RuntimeError(
        "Could not create a feasible left-arm torque bottleneck below baseline demand"
    )


def save_csv(
    output_dir: Path,
    cases: dict[str, object],
    desired_wrench: np.ndarray,
) -> Path:
    path = output_dir / "wrench_distribution_cases.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        headers = ["case", "success", "status"]
        headers += [f"WL_{x}" for x in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]
        headers += [f"WR_{x}" for x in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]
        headers += [f"Wobj_{x}" for x in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]
        headers += ["eq_residual", "max_tau_violation", "objective"]
        writer.writerow(headers)

        for name, result in cases.items():
            writer.writerow(
                [
                    name,
                    result.success,
                    result.status,
                    *result.wrench_left,
                    *result.wrench_right,
                    *result.object_wrench_reconstructed,
                    result.equality_residual,
                    result.max_torque_violation,
                    result.objective,
                ]
            )

    return path


def save_plot(
    output_dir: Path,
    baseline,
    limited,
    limited_left_limits: np.ndarray,
) -> Path:
    path = output_dir / "wrench_distribution_qp.png"

    labels = ["Left", "Right"]
    baseline_fz = [baseline.wrench_left[2], baseline.wrench_right[2]]
    limited_fz = [limited.wrench_left[2], limited.wrench_right[2]]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))

    x = np.arange(2)
    width = 0.34
    axes[0].bar(x - width / 2, baseline_fz, width, label="baseline")
    axes[0].bar(x + width / 2, limited_fz, width, label="left torque limited")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("allocated Fz [N]")
    axes[0].set_title("Task09: QP redistributes object-support wrench")
    axes[0].grid(True, axis="y")
    axes[0].legend()

    joints = np.arange(1, 8)
    assert limited.tau_left is not None
    assert limited.tau_right is not None
    axes[1].plot(joints, np.abs(limited.tau_left), marker="o", label="|tau_L|")
    axes[1].plot(joints, limited_left_limits, linestyle="--", label="left limits")
    axes[1].plot(joints, np.abs(limited.tau_right), marker="o", label="|tau_R|")
    axes[1].set_xlabel("joint index")
    axes[1].set_ylabel("joint torque [N m]")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    output_dir = ROOT / "outputs" / "task09"
    output_dir.mkdir(parents=True, exist_ok=True)

    fr3_xml = resolve_fr3_xml(model_path)
    scene_xml = output_dir / "task09_dual_fr3_geometry.xml"
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

    baseline = allocator.solve(
        desired_wrench,
        jacobian_left=jl,
        jacobian_right=jr,
        gravity_left=gl,
        gravity_right=gr,
        torque_limits_left=TORQUE_LIMITS,
        torque_limits_right=TORQUE_LIMITS,
    )
    if not baseline.success:
        raise RuntimeError(f"Task09 baseline QP failed: {baseline.status}")

    left_limits, right_limits, limited_joint, baseline_joint_tau = choose_demo_torque_limit(
        allocator, desired_wrench, jl, jr, gl, gr
    )

    limited = allocator.solve(
        desired_wrench,
        jacobian_left=jl,
        jacobian_right=jr,
        gravity_left=gl,
        gravity_right=gr,
        torque_limits_left=left_limits,
        torque_limits_right=right_limits,
    )
    if not limited.success:
        raise RuntimeError(f"Task09 limited QP unexpectedly failed: {limited.status}")

    tiny_limits = np.full(7, 0.05)
    infeasible = allocator.solve(
        desired_wrench,
        jacobian_left=jl,
        jacobian_right=jr,
        gravity_left=gl,
        gravity_right=gr,
        torque_limits_left=tiny_limits,
        torque_limits_right=tiny_limits,
    )

    cases = {
        "baseline": baseline,
        "left_torque_limited": limited,
        "intentionally_infeasible": infeasible,
    }

    print("\n=== Task09 Stage 1: wrench distribution + QP ===")
    print("Decision variable       : f = [W_L; W_R] in R^12")
    print("Wrench order            : [Fx, Fy, Fz, Mx, My, Mz]")
    print("Task/reference frame    : WORLD")
    print("Object desired wrench   :", np.array2string(desired_wrench, precision=4))
    print("Left contact offset r_L :", np.array2string(r_left, precision=4), "m")
    print("Right contact offset r_R:", np.array2string(r_right, precision=4), "m")
    print("Grasp matrix shape      :", grasp_matrix.shape)
    print("Grasp matrix rank       :", np.linalg.matrix_rank(grasp_matrix))

    print("\n--- Case A: normal torque limits ---")
    print("success                 :", baseline.success)
    print("W_L                     :", np.array2string(baseline.wrench_left, precision=4))
    print("W_R                     :", np.array2string(baseline.wrench_right, precision=4))
    print(
        "reconstructed W_object  :",
        np.array2string(baseline.object_wrench_reconstructed, precision=4),
    )
    print(f"equality residual       : {baseline.equality_residual:.3e}")
    print(
        f"vertical load split     : L={baseline.wrench_left[2]:.4f} N, "
        f"R={baseline.wrench_right[2]:.4f} N"
    )

    print("\n--- Case B: one left joint torque limit tightened ---")
    print(f"limited left joint      : J{limited_joint + 1}")
    print(f"baseline |tau|          : {baseline_joint_tau:.4f} N m")
    print(f"new left torque limit   : {left_limits[limited_joint]:.4f} N m")
    print("success                 :", limited.success)
    print("W_L                     :", np.array2string(limited.wrench_left, precision=4))
    print("W_R                     :", np.array2string(limited.wrench_right, precision=4))
    print(
        f"vertical load split     : L={limited.wrench_left[2]:.4f} N, "
        f"R={limited.wrench_right[2]:.4f} N"
    )
    print(f"equality residual       : {limited.equality_residual:.3e}")
    print(f"max torque violation    : {limited.max_torque_violation:.3e} N m")

    print("\n--- Case C: intentionally impossible torque limits ---")
    print("success                 :", infeasible.success)
    print("solver status           :", infeasible.status)
    print(f"equality residual       : {infeasible.equality_residual:.3e}")
    print(f"max torque violation    : {infeasible.max_torque_violation:.3e} N m")
    if infeasible.success:
        print("WARNING: this MuJoCo geometry still found a feasible solution at 0.05 N m.")
    else:
        print("Expected result         : infeasible / rejected clearly")

    plot_path = save_plot(output_dir, baseline, limited, left_limits)
    csv_path = save_csv(output_dir, cases, desired_wrench)

    print(f"\nPlot : {plot_path}")
    print(f"CSV  : {csv_path}")
    print(f"Scene: {scene_xml}")

    print(
        "\nInterpretation target: Gf=W_des fixes the object-level task, while the "
        "QP chooses one wrench split among many feasible splits and can react to "
        "joint-torque constraints."
    )


if __name__ == "__main__":
    main()
