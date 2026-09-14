"""Task01: FR3 TCP Jacobian, wrench mapping, and finite-difference check.

Examples:
    python3 experiments/task01_wrench_jacobian.py \
        --model ~/mujoco_menagerie/franka_fr3/scene.xml

    python3 experiments/task01_wrench_jacobian.py \
        --model ~/mujoco_menagerie/franka_fr3/scene.xml --viewer
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.mujoco_adapter import FR3_JOINT_NAMES, MuJoCoAdapter


HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to Menagerie FR3 scene.xml or fr3.xml")
    parser.add_argument("--force-z", type=float, default=10.0, help="TCP force along +base Z [N]")
    parser.add_argument("--eps", type=float, default=1e-6, help="Finite-difference step [rad]")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open a MuJoCo viewer showing the FR3, TCP frame, and commanded force arrow",
    )
    return parser.parse_args()


def disable_builtin_position_actuators(model: mujoco.MjModel) -> None:
    if model.nu == 0:
        return
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_biasprm[:, :] = 0.0


def set_q(adapter: MuJoCoAdapter, q: np.ndarray) -> None:
    if q.shape != (7,):
        raise ValueError(f"Expected q shape (7,), got {q.shape}")
    for joint, value in zip(adapter.joint_map, q):
        adapter.data.qpos[joint.qpos_adr] = float(value)
        adapter.data.qvel[joint.dof_adr] = 0.0
    adapter.forward()


def finite_difference_tcp_position_jacobian(
    adapter: MuJoCoAdapter,
    q0: np.ndarray,
    eps: float,
) -> np.ndarray:
    j_fd = np.zeros((3, 7), dtype=float)

    for i in range(7):
        q_plus = q0.copy()
        q_minus = q0.copy()
        q_plus[i] += eps
        q_minus[i] -= eps

        set_q(adapter, q_plus)
        p_plus, _ = adapter.get_tcp_pose(frame="base")

        set_q(adapter, q_minus)
        p_minus, _ = adapter.get_tcp_pose(frame="base")

        j_fd[:, i] = (p_plus - p_minus) / (2.0 * eps)

    set_q(adapter, q0)
    return j_fd


def add_force_arrow(viewer, p_tcp: np.ndarray, force_base: np.ndarray) -> None:
    """Add a purely visual arrow for a base-frame force vector.

    The arrow direction is physical; its displayed length is only a visualization
    scale and is not measured in Newtons or metres.
    """
    magnitude = float(np.linalg.norm(force_base))
    if magnitude <= 0.0:
        return

    direction = force_base / magnitude
    arrow_length = float(np.clip(0.02 * magnitude, 0.12, 0.35))
    arrow_start = np.asarray(p_tcp, dtype=float)
    arrow_end = arrow_start + arrow_length * direction

    scene = viewer.user_scn
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo viewer user scene has no free geom slots")

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.array([0.95, 0.25, 0.15, 1.0], dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.012,
        arrow_start,
        arrow_end,
    )
    scene.ngeom += 1


def show_viewer(adapter: MuJoCoAdapter, p_tcp: np.ndarray, wrench: np.ndarray) -> None:
    """Show a static Task01 visualization until the user closes the window."""
    import mujoco.viewer

    print("\n=== Task01 viewer ===")
    print("The colored frame at the attachment site is the TCP frame.")
    print("The arrow starts at the TCP and shows the commanded force direction in the base frame.")
    print("Arrow length is visual only; the numeric force remains the printed wrench value.")
    print("Close the MuJoCo window to return to the terminal.")

    with mujoco.viewer.launch_passive(adapter.model, adapter.data) as viewer:
        # Show site coordinate frames. The Menagerie FR3 uses attachment_site as
        # the TCP for this teaching project.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # Put the camera on the arm/TCP instead of relying on the default view.
        viewer.cam.lookat[:] = p_tcp
        viewer.cam.distance = 1.35
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0

        add_force_arrow(viewer, p_tcp, wrench[:3])

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

    adapter = MuJoCoAdapter.from_xml_path(str(model_path))
    disable_builtin_position_actuators(adapter.model)
    set_q(adapter, HOME_Q.copy())

    q = adapter.get_q()
    p_tcp, r_tcp = adapter.get_tcp_pose(frame="base")
    j = adapter.get_jacobian(frame="base")

    wrench = np.array([0.0, 0.0, args.force_z, 0.0, 0.0, 0.0], dtype=float)
    tau_from_wrench = j.T @ wrench

    j_fd = finite_difference_tcp_position_jacobian(adapter, q, args.eps)
    j_linear = j[:3, :]
    fd_error = j_fd - j_linear
    max_abs_error = float(np.max(np.abs(fd_error)))

    np.set_printoptions(precision=6, suppress=True)

    print("\n=== Task01 state ===")
    print(f"q [rad] = {q}")
    print(f"TCP position in base [m] = {p_tcp}")
    print("TCP rotation in base =")
    print(r_tcp)

    print("\n=== TCP geometric Jacobian in base frame ===")
    print(f"J shape = {j.shape}")
    print("rows = [vx, vy, vz, wx, wy, wz]")
    print(j)

    print("\n=== Wrench -> joint torque ===")
    print(f"W = [Fx,Fy,Fz,Mx,My,Mz] = {wrench}")
    print("tau = J^T W [N*m] =")
    for name, value in zip(FR3_JOINT_NAMES, tau_from_wrench):
        print(f"  {name}: {value: .6f}")

    print("\n=== Finite-difference check of translational Jacobian ===")
    print("Analytical Jv =")
    print(j_linear)
    print("Finite-difference Jv =")
    print(j_fd)
    print("Jv_fd - Jv =")
    print(fd_error)
    print(f"Max |error| = {max_abs_error:.3e}")

    if j.shape != (6, 7):
        raise RuntimeError(f"Unexpected Jacobian shape: {j.shape}")
    if not np.isfinite(j).all() or not np.isfinite(tau_from_wrench).all():
        raise RuntimeError("NaN/Inf detected in Jacobian or torque mapping")

    tolerance = 1e-5
    if max_abs_error > tolerance:
        raise RuntimeError(
            f"Finite-difference Jacobian check failed: {max_abs_error:.3e} > {tolerance:.1e}"
        )

    print("\nPASS: J is 6x7, tau = J^T W is finite, and translational Jacobian matches finite differences.")

    if args.viewer:
        show_viewer(adapter, p_tcp, wrench)


if __name__ == "__main__":
    main()
