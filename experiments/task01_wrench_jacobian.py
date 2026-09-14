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


def _next_user_geom(viewer):
    scene = viewer.user_scn
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo viewer user scene has no free geom slots")
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    return geom


def _add_sphere(viewer, position: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    geom = _next_user_geom(viewer)
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=float),
        np.asarray(position, dtype=float),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )


def _add_arrow(
    viewer,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    rgba: np.ndarray,
) -> None:
    geom = _next_user_geom(viewer)
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        float(width),
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )


def draw_task01_debug_geometry(
    viewer,
    p_tcp: np.ndarray,
    r_tcp: np.ndarray,
    force_base: np.ndarray,
) -> None:
    """Redraw the TCP marker, TCP axes, and base-frame force arrow every frame."""
    # Passive viewer versions can rebuild their rendered scene during sync(), so
    # reconstruct our user geoms on every frame for reliable visibility.
    viewer.user_scn.ngeom = 0

    p_tcp = np.asarray(p_tcp, dtype=float)
    r_tcp = np.asarray(r_tcp, dtype=float)
    force_base = np.asarray(force_base, dtype=float)

    # Bright TCP marker so the user can immediately locate the origin.
    _add_sphere(
        viewer,
        p_tcp,
        radius=0.025,
        rgba=np.array([1.0, 1.0, 0.15, 1.0], dtype=np.float32),
    )

    # Explicit TCP axes. Columns of R_base_tcp are the TCP x/y/z axes expressed
    # in the base frame.
    axis_length = 0.16
    axis_width = 0.008
    axis_colors = (
        np.array([0.95, 0.10, 0.10, 1.0], dtype=np.float32),  # TCP +X: red
        np.array([0.10, 0.90, 0.20, 1.0], dtype=np.float32),  # TCP +Y: green
        np.array([0.10, 0.35, 1.00, 1.0], dtype=np.float32),  # TCP +Z: blue
    )
    for axis_index, color in enumerate(axis_colors):
        axis_end = p_tcp + axis_length * r_tcp[:, axis_index]
        _add_arrow(viewer, p_tcp, axis_end, axis_width, color)

    # Force arrow is expressed in the BASE frame, not in the TCP frame.
    magnitude = float(np.linalg.norm(force_base))
    if magnitude > 0.0:
        direction = force_base / magnitude
        force_length = float(np.clip(0.025 * magnitude, 0.25, 0.45))
        force_start = p_tcp + 0.035 * direction
        force_end = force_start + force_length * direction
        _add_arrow(
            viewer,
            force_start,
            force_end,
            width=0.022,
            rgba=np.array([1.0, 0.55, 0.05, 1.0], dtype=np.float32),
        )


def show_viewer(
    adapter: MuJoCoAdapter,
    p_tcp: np.ndarray,
    r_tcp: np.ndarray,
    wrench: np.ndarray,
) -> None:
    """Show a static Task01 visualization until the user closes the window."""
    import mujoco.viewer

    print("\n=== Task01 viewer ===")
    print("Yellow sphere : TCP origin")
    print("Red arrow     : TCP +X")
    print("Green arrow   : TCP +Y")
    print("Blue arrow    : TCP +Z")
    print("Orange arrow  : commanded force, expressed in BASE frame")
    print("Arrow lengths are visual only; the numeric force remains the printed wrench value.")
    print("Close the MuJoCo window to return to the terminal.")

    with mujoco.viewer.launch_passive(adapter.model, adapter.data) as viewer:
        viewer.cam.lookat[:] = p_tcp
        viewer.cam.distance = 1.15
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0

        while viewer.is_running():
            draw_task01_debug_geometry(viewer, p_tcp, r_tcp, wrench[:3])
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
        show_viewer(adapter, p_tcp, r_tcp, wrench)


if __name__ == "__main__":
    main()
