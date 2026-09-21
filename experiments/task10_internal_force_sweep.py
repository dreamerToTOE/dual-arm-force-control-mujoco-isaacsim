"""Task10 Stage 1: internal-force reference sweep.

The object-level task is kept fixed:

    W_obj_des = [0, 0, mg, 0, 0, 0].

Only the desired symmetric internal compression is changed.  The QP objective
tracks the full reference

    f_ref = f_task + f_internal_des,

while preserving:
    G f = W_obj_des,
    joint torque limits,
    unilateral side-contact constraints,
    friction-pyramid constraints,
    contact moment limits.

This shows the key tradeoff:
    more internal compression -> more friction margin,
    but usually more joint-torque effort.
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

from controllers.internal_wrench import (
    decompose_wrench,
    minimum_norm_task_component,
    nullspace_projector,
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


MU = 0.80
NORMAL_FORCE_MAX = 30.0
MOMENT_LIMITS = np.array([2.0, 2.0, 2.0])

INTERNAL_COMPRESSION_REFERENCES = np.array(
    [6.5, 10.0, 15.0, 20.0, 25.0, 40.0],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to MuJoCo Menagerie franka_fr3/scene.xml or fr3.xml",
    )
    return parser.parse_args()


def stacked(result) -> np.ndarray:
    return np.concatenate((result.wrench_left, result.wrench_right))


def compression_from_wrench(f: np.ndarray) -> float:
    """Average inward side compression for a zero-net-X allocation."""
    return 0.5 * float(f[0] - f[6])


def friction_utilization(f: np.ndarray, mu: float) -> tuple[float, float]:
    """Return left/right utilization of |Fy|+|Fz| <= mu Fn."""
    wl = f[:6]
    wr = f[6:]
    fn_l = float(wl[0])
    fn_r = float(-wr[0])

    tangential_l = float(abs(wl[1]) + abs(wl[2]))
    tangential_r = float(abs(wr[1]) + abs(wr[2]))

    util_l = tangential_l / (mu * fn_l) if fn_l > 1e-12 else float("inf")
    util_r = tangential_r / (mu * fn_r) if fn_r > 1e-12 else float("inf")
    return util_l, util_r


def torque_utilization(result) -> tuple[float, float]:
    if result.tau_left is None or result.tau_right is None:
        raise RuntimeError("Task10 result is missing joint torque values")

    util_l = float(np.max(np.abs(result.tau_left) / TORQUE_LIMITS))
    util_r = float(np.max(np.abs(result.tau_right) / TORQUE_LIMITS))
    return util_l, util_r


def save_csv(output_dir: Path, rows: list[dict]) -> Path:
    path = output_dir / "internal_force_sweep.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_plot(output_dir: Path, rows: list[dict]) -> Path:
    path = output_dir / "internal_force_sweep.png"

    requested = np.array([row["compression_des_N"] for row in rows])
    realized = np.array([row["compression_actual_N"] for row in rows])
    friction_l = 100.0 * np.array([row["friction_util_left"] for row in rows])
    friction_r = 100.0 * np.array([row["friction_util_right"] for row in rows])
    torque_l = 100.0 * np.array([row["torque_util_left"] for row in rows])
    torque_r = 100.0 * np.array([row["torque_util_right"] for row in rows])
    object_fz = np.array([row["object_Fz_N"] for row in rows])
    object_fx = np.array([row["object_Fx_N"] for row in rows])

    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)

    axes[0].plot(requested, requested, linestyle="--", label="desired compression")
    axes[0].plot(requested, realized, marker="o", label="QP realized compression")
    axes[0].axhline(NORMAL_FORCE_MAX, linestyle=":", label="contact normal max")
    axes[0].set_ylabel("internal compression [N]")
    axes[0].set_title("Task10: internal force changes grasp loading without changing object wrench")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(requested, friction_l, marker="o", label="left friction utilization")
    axes[1].plot(requested, friction_r, marker="o", label="right friction utilization")
    axes[1].axhline(100.0, linestyle="--", label="friction boundary")
    axes[1].set_ylabel("friction utilization [%]")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(requested, torque_l, marker="o", label="left max torque utilization")
    axes[2].plot(requested, torque_r, marker="o", label="right max torque utilization")
    axes[2].axhline(100.0, linestyle="--", label="torque limit")
    axes[2].set_ylabel("max joint torque utilization [%]")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(requested, object_fz, marker="o", label="object Fz")
    axes[3].plot(requested, object_fx, marker="o", label="object Fx")
    axes[3].axhline(BOX_MASS * GRAVITY, linestyle="--", label="desired Fz")
    axes[3].set_xlabel("desired internal compression [N]")
    axes[3].set_ylabel("reconstructed object force [N]")
    axes[3].grid(True)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    output_dir = ROOT / "outputs" / "task10"
    output_dir.mkdir(parents=True, exist_ok=True)

    fr3_xml = resolve_fr3_xml(model_path)
    scene_xml = output_dir / "task10_dual_fr3_geometry.xml"
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

    desired_object_wrench = np.array(
        [0.0, 0.0, BOX_MASS * GRAVITY, 0.0, 0.0, 0.0],
        dtype=float,
    )

    allocator = DualArmWrenchQP(grasp_matrix)
    contact_a, contact_b = side_grasp_contact_constraints(
        friction_coefficient=MU,
        normal_force_max=NORMAL_FORCE_MAX,
        moment_limits=MOMENT_LIMITS,
    )

    f_task = minimum_norm_task_component(
        grasp_matrix,
        desired_object_wrench,
    )
    null_projector = nullspace_projector(grasp_matrix)

    print("\n=== Task10 Stage 1: internal-force reference sweep ===")
    print("Object desired wrench   :", np.array2string(desired_object_wrench, precision=4))
    print("Task component f_task   :", np.array2string(f_task, precision=4))
    print("rank(G)                 :", np.linalg.matrix_rank(grasp_matrix))
    print("dim null(G)             :", 12 - np.linalg.matrix_rank(grasp_matrix))
    print(f"Friction coefficient mu : {MU:.2f}")
    print(f"Normal force max        : {NORMAL_FORCE_MAX:.2f} N per contact")
    print("Left contact offset     :", np.array2string(r_left, precision=4), "m")
    print("Right contact offset    :", np.array2string(r_right, precision=4), "m")
    print(
        "Projector check ||G N||_inf:",
        f"{np.linalg.norm(grasp_matrix @ null_projector, ord=np.inf):.3e}",
    )

    rows: list[dict] = []

    print(
        "\nref[N]  actual[N]  friction L/R[%]  torque L/R[%]  "
        "||G f_int||inf   object Fx/Fz[N]  status"
    )
    print("-" * 118)

    for compression_des in INTERNAL_COMPRESSION_REFERENCES:
        f_int_des = symmetric_side_compression(compression_des)
        null_residual_des = float(
            np.linalg.norm(grasp_matrix @ f_int_des, ord=np.inf)
        )

        reference = f_task + f_int_des

        result = allocator.solve(
            desired_object_wrench,
            reference_wrench=reference,
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
                f"Task10 QP failed for internal reference {compression_des:.2f} N: "
                f"{result.status}"
            )

        f = stacked(result)
        decomposition = decompose_wrench(
            grasp_matrix,
            f,
            desired_object_wrench,
        )

        compression_actual = compression_from_wrench(f)
        friction_l, friction_r = friction_utilization(f, MU)
        torque_l, torque_r = torque_utilization(result)
        w_obj = result.object_wrench_reconstructed

        row = {
            "compression_des_N": float(compression_des),
            "compression_actual_N": compression_actual,
            "compression_error_N": compression_actual - float(compression_des),
            "friction_util_left": friction_l,
            "friction_util_right": friction_r,
            "torque_util_left": torque_l,
            "torque_util_right": torque_r,
            "internal_null_residual": decomposition.internal_object_residual,
            "desired_internal_null_residual": null_residual_des,
            "object_Fx_N": float(w_obj[0]),
            "object_Fz_N": float(w_obj[2]),
            "object_wrench_error_inf": result.equality_residual,
            "contact_violation": result.max_linear_inequality_violation,
            "torque_violation_Nm": result.max_torque_violation,
        }
        rows.append(row)

        print(
            f"{compression_des:6.1f}  "
            f"{compression_actual:9.3f}  "
            f"{100*friction_l:7.2f}/{100*friction_r:7.2f}  "
            f"{100*torque_l:7.2f}/{100*torque_r:7.2f}  "
            f"{decomposition.internal_object_residual:14.3e}  "
            f"{w_obj[0]:7.3f}/{w_obj[2]:7.3f}  OK"
        )

    plot_path = save_plot(output_dir, rows)
    csv_path = save_csv(output_dir, rows)

    print("\nKey interpretation:")
    print(
        "Increasing internal compression should reduce friction utilization "
        "(increase friction margin), while joint-torque utilization generally grows."
    )
    print(
        "The reconstructed object wrench remains fixed because the added compression "
        "lies in null(G)."
    )
    print(
        "The 40 N reference is intentionally above the 30 N/contact normal-force "
        "constraint, so the realized value should stop at the feasible boundary "
        "rather than violating the constraint."
    )

    print(f"\nPlot : {plot_path}")
    print(f"CSV  : {csv_path}")
    print(f"Scene: {scene_xml}")


if __name__ == "__main__":
    main()
