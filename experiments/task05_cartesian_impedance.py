"""Task05: 3D translational Cartesian impedance control.

The TCP is connected to its HOME position by a virtual Cartesian spring-damper:

    F_cmd   = Kx (p_des - p) + Dx (v_des - v)
    tau_task = Jv.T @ F_cmd
    tau      = tau_task + g(q) + tau_external

A BASE-frame external force pulse is applied at the TCP through the equivalent
generalized torque Jv.T @ F_ext. This is a disturbance-response experiment, not
closed-loop contact-force regulation (that comes in Task06).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.cartesian_impedance import TranslationalCartesianImpedance
from sim.mujoco_adapter import MuJoCoAdapter


HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
FORCE_LIMIT = 60.0

# Same disturbance for all cases: +BASE-X force at the TCP.
EXT_FORCE = np.array([8.0, 0.0, 0.0])
PULSE_START = 1.0
PULSE_END = 3.0
DURATION = 5.0

CASES = {
    # stiffness sweep: similar damping, different spring stiffness
    "soft": (np.array([120.0, 120.0, 120.0]), np.array([28.0, 28.0, 28.0])),
    "hard": (np.array([400.0, 400.0, 400.0]), np.array([48.0, 48.0, 48.0])),
    # damping sweep: same stiffness, very different damping
    "low_damping": (np.array([250.0, 250.0, 250.0]), np.array([5.0, 5.0, 5.0])),
    "damped": (np.array([250.0, 250.0, 250.0]), np.array([35.0, 35.0, 35.0])),
}


@dataclass
class RunResult:
    name: str
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    force_cmd: np.ndarray
    force_ext: np.ndarray
    tau_task: np.ndarray
    tau_total: np.ndarray
    orientation_error_deg: np.ndarray
    target_position: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to Menagerie FR3 scene.xml or fr3.xml")
    parser.add_argument("--viewer", action="store_true", help="Replay the damped case in MuJoCo viewer")
    parser.add_argument(
        "--viewer-case",
        choices=tuple(CASES.keys()),
        default="damped",
        help="Case shown in the optional viewer",
    )
    return parser.parse_args()


def disable_builtin_position_actuators(model: mujoco.MjModel) -> None:
    if model.nu:
        model.actuator_gainprm[:, :] = 0.0
        model.actuator_biasprm[:, :] = 0.0


def set_home(adapter: MuJoCoAdapter) -> None:
    for joint, value in zip(adapter.joint_map, HOME_Q):
        adapter.data.qpos[joint.qpos_adr] = float(value)
        adapter.data.qvel[joint.dof_adr] = 0.0
    adapter.clear_joint_torque()
    adapter.forward()


def reset_home(adapter: MuJoCoAdapter) -> None:
    mujoco.mj_resetData(adapter.model, adapter.data)
    set_home(adapter)


def rotation_error_deg(r0: np.ndarray, r: np.ndarray) -> float:
    rel = r0.T @ r
    cosine = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def external_force(t: float) -> np.ndarray:
    if PULSE_START <= t < PULSE_END:
        return EXT_FORCE.copy()
    return np.zeros(3)


def simulate_case(model_path: Path, name: str) -> RunResult:
    stiffness, damping = CASES[name]
    adapter = MuJoCoAdapter.from_xml_path(str(model_path))
    disable_builtin_position_actuators(adapter.model)
    set_home(adapter)

    controller = TranslationalCartesianImpedance.from_gains(
        stiffness=stiffness,
        damping=damping,
        force_limits=FORCE_LIMIT,
    )

    p_des, r_des = adapter.get_tcp_pose(frame="base")
    dt = float(adapter.model.opt.timestep)
    steps = int(np.ceil(DURATION / dt)) + 1

    time_log = np.zeros(steps)
    p_log = np.zeros((steps, 3))
    v_log = np.zeros((steps, 3))
    fcmd_log = np.zeros((steps, 3))
    fext_log = np.zeros((steps, 3))
    tau_task_log = np.zeros((steps, 7))
    tau_total_log = np.zeros((steps, 7))
    ori_log = np.zeros(steps)

    for k in range(steps):
        t = float(adapter.data.time)
        qdot = adapter.get_qdot()
        p, r = adapter.get_tcp_pose(frame="base")
        j = adapter.get_jacobian(frame="base")
        jv = j[:3, :]
        v = jv @ qdot

        f_cmd = controller.compute(
            position=p,
            velocity=v,
            desired_position=p_des,
            desired_velocity=np.zeros(3),
        )
        f_ext = external_force(t)

        tau_task = jv.T @ f_cmd
        tau_ext = jv.T @ f_ext
        tau_total = tau_task + tau_ext + adapter.get_gravity()
        tau_total = np.clip(tau_total, -TORQUE_LIMITS, TORQUE_LIMITS)

        time_log[k] = t
        p_log[k] = p
        v_log[k] = v
        fcmd_log[k] = f_cmd
        fext_log[k] = f_ext
        tau_task_log[k] = tau_task
        tau_total_log[k] = tau_total
        ori_log[k] = rotation_error_deg(r_des, r)

        if k < steps - 1:
            adapter.set_joint_torque(tau_total)
            adapter.step()

        if not np.isfinite(adapter.data.qpos).all() or not np.isfinite(adapter.data.qvel).all():
            raise RuntimeError(f"NaN/Inf in case {name} at t={t:.4f}s")

    return RunResult(
        name=name,
        time=time_log,
        position=p_log,
        velocity=v_log,
        force_cmd=fcmd_log,
        force_ext=fext_log,
        tau_task=tau_task_log,
        tau_total=tau_total_log,
        orientation_error_deg=ori_log,
        target_position=p_des,
        stiffness=stiffness,
        damping=damping,
    )


def recovery_time(result: RunResult, tolerance: float = 0.005, hold: float = 0.25) -> float:
    error_norm = np.linalg.norm(result.target_position - result.position, axis=1)
    start = int(np.searchsorted(result.time, PULSE_END))
    dt = float(np.median(np.diff(result.time)))
    window = max(1, int(np.ceil(hold / dt)))
    for i in range(start, max(start, len(result.time) - window)):
        if np.all(error_norm[i : i + window] <= tolerance):
            return float(result.time[i] - PULSE_END)
    return float("nan")


def print_metrics(results: list[RunResult]) -> None:
    print("\n=== Task05 translational Cartesian impedance ===")
    print("Frame                 : BASE")
    print(f"External force pulse  : {EXT_FORCE} N, t=[{PULSE_START:.1f}, {PULSE_END:.1f}) s")
    print("Controller             : F_cmd = Kx(p_des-p) + Dx(v_des-v)")
    print("Torque mapping         : tau_task = Jv^T F_cmd")
    print("Gravity compensation   : enabled")
    print("NOTE                   : orientation is intentionally NOT controlled in this first 3D task")
    print("\ncase          Kx[N/m]   Dx[Ns/m]   peak |e_p|[m]   pulse-end |dx|[m]   F/Kx[m]   recovery<5mm[s]   max ori drift[deg]")
    print("----------------------------------------------------------------------------------------------------------------")

    for result in results:
        error = result.target_position[None, :] - result.position
        error_norm = np.linalg.norm(error, axis=1)
        dx = result.position[:, 0] - result.target_position[0]
        # Average shortly before the force is released, after the main transient.
        mask = (result.time >= PULSE_END - 0.35) & (result.time < PULSE_END - 0.05)
        dx_tail = float(np.mean(np.abs(dx[mask]))) if np.any(mask) else float("nan")
        expected = abs(float(EXT_FORCE[0])) / float(result.stiffness[0])
        rec = recovery_time(result)
        print(
            f"{result.name:<13} "
            f"{result.stiffness[0]:>8.1f}   "
            f"{result.damping[0]:>9.1f}   "
            f"{np.max(error_norm):>13.5f}   "
            f"{dx_tail:>16.5f}   "
            f"{expected:>8.5f}   "
            f"{rec:>16.4f}   "
            f"{np.max(result.orientation_error_deg):>18.3f}"
        )


def save_csv(result: RunResult, output_dir: Path) -> Path:
    path = output_dir / f"{result.name}.csv"
    headers = ["time", "px", "py", "pz", "vx", "vy", "vz", "Fcmd_x", "Fcmd_y", "Fcmd_z", "Fext_x", "Fext_y", "Fext_z"]
    headers += [f"tau_task_{i}" for i in range(1, 8)]
    headers += [f"tau_total_{i}" for i in range(1, 8)]
    headers += ["orientation_error_deg"]
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for k in range(len(result.time)):
            writer.writerow([
                result.time[k],
                *result.position[k],
                *result.velocity[k],
                *result.force_cmd[k],
                *result.force_ext[k],
                *result.tau_task[k],
                *result.tau_total[k],
                result.orientation_error_deg[k],
            ])
    return path


def save_stiffness_plot(results: dict[str, RunResult], output_dir: Path) -> Path:
    path = output_dir / "stiffness_comparison.png"
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
    for name in ("soft", "hard"):
        result = results[name]
        dx = result.position[:, 0] - result.target_position[0]
        err = np.linalg.norm(result.target_position[None, :] - result.position, axis=1)
        axes[0].plot(result.time, dx, label=f"{name}: K={result.stiffness[0]:.0f}")
        axes[1].plot(result.time, err, label=name)
        axes[2].plot(result.time, result.force_cmd[:, 0], label=f"Fcmd_x {name}")
    axes[2].plot(results["soft"].time, results["soft"].force_ext[:, 0], linestyle="--", label="Fext_x")
    axes[0].axvspan(PULSE_START, PULSE_END, alpha=0.12)
    axes[1].axvspan(PULSE_START, PULSE_END, alpha=0.12)
    axes[2].axvspan(PULSE_START, PULSE_END, alpha=0.12)
    axes[0].set_ylabel("TCP dx [m]")
    axes[0].set_title("Task05 stiffness sweep: same external force, soft vs hard Cartesian spring")
    axes[1].set_ylabel("|position error| [m]")
    axes[2].set_ylabel("force X [N]")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_damping_plot(results: dict[str, RunResult], output_dir: Path) -> Path:
    path = output_dir / "damping_comparison.png"
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
    for name in ("low_damping", "damped"):
        result = results[name]
        dx = result.position[:, 0] - result.target_position[0]
        axes[0].plot(result.time, dx, label=f"{name}: D={result.damping[0]:.0f}")
        axes[1].plot(result.time, result.velocity[:, 0], label=name)
        axes[2].plot(result.time, result.force_cmd[:, 0], label=name)
    axes[0].axvspan(PULSE_START, PULSE_END, alpha=0.12)
    axes[1].axvspan(PULSE_START, PULSE_END, alpha=0.12)
    axes[2].axvspan(PULSE_START, PULSE_END, alpha=0.12)
    axes[0].set_ylabel("TCP dx [m]")
    axes[0].set_title("Task05 damping sweep: same stiffness, low vs higher Cartesian damping")
    axes[1].set_ylabel("TCP vx [m/s]")
    axes[2].set_ylabel("Fcmd_x [N]")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _next_geom(viewer):
    scene = viewer.user_scn
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo viewer user scene has no free geom slots")
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    return geom


def _add_sphere(viewer, position: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    geom = _next_geom(viewer)
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius]),
        np.asarray(position, dtype=float),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )


def _add_arrow(viewer, start: np.ndarray, vector: np.ndarray, rgba: np.ndarray) -> None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return
    geom = _next_geom(viewer)
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    direction = vector / norm
    length = float(np.clip(0.025 * norm, 0.08, 0.35))
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.018,
        np.asarray(start, dtype=float),
        np.asarray(start + length * direction, dtype=float),
    )


def draw_viewer(viewer, p: np.ndarray, p_des: np.ndarray, f_ext: np.ndarray, f_cmd: np.ndarray) -> None:
    viewer.user_scn.ngeom = 0
    _add_sphere(viewer, p_des, 0.025, np.array([0.1, 0.9, 0.3, 1.0]))
    _add_sphere(viewer, p, 0.020, np.array([1.0, 0.9, 0.1, 1.0]))
    _add_arrow(viewer, p + np.array([0.0, 0.05, 0.0]), f_ext, np.array([0.95, 0.15, 0.15, 1.0]))
    _add_arrow(viewer, p + np.array([0.0, -0.05, 0.0]), f_cmd, np.array([1.0, 0.55, 0.05, 1.0]))


def replay_viewer(model_path: Path, case_name: str) -> None:
    import mujoco.viewer

    stiffness, damping = CASES[case_name]
    adapter = MuJoCoAdapter.from_xml_path(str(model_path))
    disable_builtin_position_actuators(adapter.model)
    set_home(adapter)
    p_des, _ = adapter.get_tcp_pose(frame="base")
    controller = TranslationalCartesianImpedance.from_gains(stiffness, damping, FORCE_LIMIT)

    print("\n=== Task05 viewer replay ===")
    print(f"Case         : {case_name}, K={stiffness[0]:.1f} N/m, D={damping[0]:.1f} N*s/m")
    print("Green sphere : desired TCP position")
    print("Yellow sphere: actual TCP position")
    print("Red arrow    : external disturbance force (+BASE-X during the pulse)")
    print("Orange arrow : impedance restoring force F_cmd")
    print("The cycle repeats automatically. Close the viewer to stop.")

    with mujoco.viewer.launch_passive(adapter.model, adapter.data) as viewer:
        viewer.cam.lookat[:] = p_des
        viewer.cam.distance = 1.15
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0
        last_wall = time.time()

        while viewer.is_running():
            if adapter.data.time >= DURATION:
                reset_home(adapter)

            t = float(adapter.data.time)
            qdot = adapter.get_qdot()
            p, _ = adapter.get_tcp_pose(frame="base")
            jv = adapter.get_jacobian(frame="base")[:3, :]
            v = jv @ qdot
            f_cmd = controller.compute(p, v, p_des)
            f_ext = external_force(t)
            tau = jv.T @ (f_cmd + f_ext) + adapter.get_gravity()
            tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
            adapter.set_joint_torque(tau)
            adapter.step()

            draw_viewer(viewer, p, p_des, f_ext, f_cmd)
            viewer.sync()

            dt = float(adapter.model.opt.timestep)
            now = time.time()
            sleep = dt - (now - last_wall)
            if sleep > 0.0:
                time.sleep(sleep)
            last_wall = time.time()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

    output_dir = ROOT / "outputs" / "task05"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_list = [simulate_case(model_path, name) for name in CASES]
    results = {result.name: result for result in result_list}
    print_metrics(result_list)

    csv_paths = [save_csv(result, output_dir) for result in result_list]
    stiffness_plot = save_stiffness_plot(results, output_dir)
    damping_plot = save_damping_plot(results, output_dir)

    for result in result_list:
        if not np.isfinite(result.position).all() or not np.isfinite(result.tau_total).all():
            raise RuntimeError(f"Non-finite result in {result.name}")

    print(f"\nStiffness plot: {stiffness_plot}")
    print(f"Damping plot  : {damping_plot}")
    for path in csv_paths:
        print(f"CSV           : {path}")
    print("\nPASS: all translational Cartesian-impedance cases completed without NaN/Inf.")
    print("Acceptance target: explain stiffness/compliance, damping, recovery, and why 3D translation does not constrain orientation.")

    if args.viewer:
        replay_viewer(model_path, args.viewer_case)


if __name__ == "__main__":
    main()
