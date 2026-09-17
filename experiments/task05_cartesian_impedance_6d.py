"""Task05 stage 2: compare translation-only and full 6D Cartesian impedance.

Both cases receive the same BASE-frame external wrench pulse:

    W_ext = [Fx, Fy, Fz, Mx, My, Mz]

The translation-only case regulates TCP position but does not actively regulate
orientation. The full 6D case adds rotational spring-damper behavior:

    F_cmd = Kt (p_des - p) + Dt (v_des - v)
    M_cmd = Kr e_R + Dr (omega_des - omega)
    W_cmd = [F_cmd, M_cmd]
    tau    = J.T @ W_cmd + J.T @ W_ext + g(q)

All Cartesian quantities are expressed in BASE frame.
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

from controllers.cartesian_impedance import (
    CartesianImpedance6D,
    TranslationalCartesianImpedance,
    rotation_error_base,
)
from sim.mujoco_adapter import MuJoCoAdapter


HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

PULSE_START = 1.0
PULSE_END = 3.0
DURATION = 6.0

# Same disturbance for both controllers: translate +X and twist about +BASE-Y.
W_EXT = np.array([6.0, 0.0, 0.0, 0.0, 1.2, 0.0])

KT = np.array([250.0, 250.0, 250.0])
DT = np.array([35.0, 35.0, 35.0])
KR = np.array([20.0, 20.0, 20.0])
DR = np.array([4.0, 4.0, 4.0])

FORCE_LIMIT = 60.0
MOMENT_LIMIT = 8.0
MODES = ("translation_only", "full_6d")


@dataclass
class RunResult:
    mode: str
    time: np.ndarray
    position: np.ndarray
    orientation_error_rad: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    wrench_cmd: np.ndarray
    wrench_ext: np.ndarray
    tau_total: np.ndarray
    target_position: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to Menagerie FR3 scene.xml or fr3.xml")
    parser.add_argument("--viewer", action="store_true", help="Replay the full-6D case in MuJoCo viewer")
    parser.add_argument("--viewer-mode", choices=MODES, default="full_6d")
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


def external_wrench(t: float) -> np.ndarray:
    if PULSE_START <= t < PULSE_END:
        return W_EXT.copy()
    return np.zeros(6)


def build_controllers() -> tuple[TranslationalCartesianImpedance, CartesianImpedance6D]:
    translation = TranslationalCartesianImpedance.from_gains(
        stiffness=KT,
        damping=DT,
        force_limits=FORCE_LIMIT,
    )
    full = CartesianImpedance6D.from_gains(
        translational_stiffness=KT,
        translational_damping=DT,
        rotational_stiffness=KR,
        rotational_damping=DR,
        force_limits=FORCE_LIMIT,
        moment_limits=MOMENT_LIMIT,
    )
    return translation, full


def command_wrench(
    mode: str,
    translation: TranslationalCartesianImpedance,
    full: CartesianImpedance6D,
    p: np.ndarray,
    r: np.ndarray,
    v: np.ndarray,
    omega: np.ndarray,
    p_des: np.ndarray,
    r_des: np.ndarray,
) -> np.ndarray:
    if mode == "translation_only":
        force = translation.compute(
            position=p,
            velocity=v,
            desired_position=p_des,
            desired_velocity=np.zeros(3),
        )
        return np.concatenate((force, np.zeros(3)))

    if mode == "full_6d":
        return full.compute(
            position=p,
            rotation=r,
            linear_velocity=v,
            angular_velocity=omega,
            desired_position=p_des,
            desired_rotation=r_des,
        )

    raise ValueError(f"Unknown mode: {mode}")


def simulate(model_path: Path, mode: str) -> RunResult:
    adapter = MuJoCoAdapter.from_xml_path(str(model_path))
    disable_builtin_position_actuators(adapter.model)
    set_home(adapter)
    translation, full = build_controllers()

    p_des, r_des = adapter.get_tcp_pose(frame="base")
    dt = float(adapter.model.opt.timestep)
    steps = int(np.ceil(DURATION / dt)) + 1

    time_log = np.zeros(steps)
    p_log = np.zeros((steps, 3))
    er_log = np.zeros((steps, 3))
    v_log = np.zeros((steps, 3))
    omega_log = np.zeros((steps, 3))
    wcmd_log = np.zeros((steps, 6))
    wext_log = np.zeros((steps, 6))
    tau_log = np.zeros((steps, 7))

    for k in range(steps):
        t = float(adapter.data.time)
        qdot = adapter.get_qdot()
        p, r = adapter.get_tcp_pose(frame="base")
        j = adapter.get_jacobian(frame="base")
        twist = j @ qdot
        v = twist[:3]
        omega = twist[3:]

        w_cmd = command_wrench(mode, translation, full, p, r, v, omega, p_des, r_des)
        w_ext = external_wrench(t)
        tau = j.T @ (w_cmd + w_ext) + adapter.get_gravity()
        tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)

        time_log[k] = t
        p_log[k] = p
        er_log[k] = rotation_error_base(r, r_des)
        v_log[k] = v
        omega_log[k] = omega
        wcmd_log[k] = w_cmd
        wext_log[k] = w_ext
        tau_log[k] = tau

        if k < steps - 1:
            adapter.set_joint_torque(tau)
            adapter.step()

        if not np.isfinite(adapter.data.qpos).all() or not np.isfinite(adapter.data.qvel).all():
            raise RuntimeError(f"NaN/Inf in {mode} at t={t:.4f}s")

    return RunResult(
        mode=mode,
        time=time_log,
        position=p_log,
        orientation_error_rad=er_log,
        linear_velocity=v_log,
        angular_velocity=omega_log,
        wrench_cmd=wcmd_log,
        wrench_ext=wext_log,
        tau_total=tau_log,
        target_position=p_des,
    )


def save_csv(result: RunResult, output_dir: Path) -> Path:
    path = output_dir / f"{result.mode}_6d_stage.csv"
    headers = ["time", "px", "py", "pz", "erx", "ery", "erz", "vx", "vy", "vz", "wx", "wy", "wz"]
    headers += ["Fcmd_x", "Fcmd_y", "Fcmd_z", "Mcmd_x", "Mcmd_y", "Mcmd_z"]
    headers += ["Fext_x", "Fext_y", "Fext_z", "Mext_x", "Mext_y", "Mext_z"]
    headers += [f"tau{i}" for i in range(1, 8)]
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for k in range(len(result.time)):
            writer.writerow([
                result.time[k],
                *result.position[k],
                *result.orientation_error_rad[k],
                *result.linear_velocity[k],
                *result.angular_velocity[k],
                *result.wrench_cmd[k],
                *result.wrench_ext[k],
                *result.tau_total[k],
            ])
    return path


def save_comparison_plot(results: dict[str, RunResult], output_dir: Path) -> Path:
    path = output_dir / "pose_impedance_3d_vs_6d.png"
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 11))

    for mode in MODES:
        result = results[mode]
        pos_error = np.linalg.norm(result.target_position[None, :] - result.position, axis=1)
        ori_error_deg = np.degrees(np.linalg.norm(result.orientation_error_rad, axis=1))
        axes[0].plot(result.time, pos_error, label=mode)
        axes[1].plot(result.time, ori_error_deg, label=mode)
        axes[2].plot(result.time, result.wrench_cmd[:, 0], label=f"{mode}: Fcmd_x")
        axes[3].plot(result.time, result.wrench_cmd[:, 4], label=f"{mode}: Mcmd_y")

    for ax in axes:
        ax.axvspan(PULSE_START, PULSE_END, alpha=0.12)
        ax.grid(True)
        ax.legend()

    axes[0].set_ylabel("|position error| [m]")
    axes[0].set_title("Task05 stage 2: translation-only vs full 6D Cartesian impedance")
    axes[1].set_ylabel("orientation error [deg]")
    axes[2].set_ylabel("Fcmd_x [N]")
    axes[3].set_ylabel("Mcmd_y [N m]")
    axes[3].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def print_metrics(results: dict[str, RunResult]) -> None:
    print("\n=== Task05 stage 2: 3D vs 6D Cartesian impedance ===")
    print("Frame              : BASE")
    print(f"External wrench    : {W_EXT}")
    print(f"Pulse              : t=[{PULSE_START:.1f}, {PULSE_END:.1f}) s")
    print(f"Translation gains  : Kt={KT[0]:.1f} N/m, Dt={DT[0]:.1f} N*s/m")
    print(f"Rotation gains     : Kr={KR[0]:.1f} N*m/rad, Dr={DR[0]:.1f} N*m*s/rad (full_6d only)")
    print("\nmode              peak pos err[m]   peak ori err[deg]   final pos err[m]   final ori err[deg]   peak |tau|[Nm]")
    print("-----------------------------------------------------------------------------------------------------------")

    for mode in MODES:
        result = results[mode]
        pos_error = np.linalg.norm(result.target_position[None, :] - result.position, axis=1)
        ori_error_deg = np.degrees(np.linalg.norm(result.orientation_error_rad, axis=1))
        print(
            f"{mode:<18} "
            f"{np.max(pos_error):>15.5f}   "
            f"{np.max(ori_error_deg):>17.3f}   "
            f"{pos_error[-1]:>16.5f}   "
            f"{ori_error_deg[-1]:>18.3f}   "
            f"{np.max(np.abs(result.tau_total)):>14.4f}"
        )


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


def _add_arrow(viewer, start: np.ndarray, vector: np.ndarray, rgba: np.ndarray, width: float = 0.014) -> None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
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
    length = float(np.clip(norm, 0.07, 0.24))
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        width,
        np.asarray(start, dtype=float),
        np.asarray(start + length * direction, dtype=float),
    )


def draw_viewer_geometry(
    viewer,
    p: np.ndarray,
    r: np.ndarray,
    p_des: np.ndarray,
    r_des: np.ndarray,
    w_ext: np.ndarray,
    w_cmd: np.ndarray,
) -> None:
    viewer.user_scn.ngeom = 0
    _add_sphere(viewer, p_des, 0.018, np.array([0.1, 0.9, 0.2, 1.0]))
    _add_sphere(viewer, p, 0.014, np.array([1.0, 0.9, 0.1, 1.0]))

    axis_colors = (
        np.array([0.95, 0.1, 0.1, 1.0]),
        np.array([0.1, 0.9, 0.2, 1.0]),
        np.array([0.1, 0.35, 1.0, 1.0]),
    )
    for i, color in enumerate(axis_colors):
        _add_arrow(viewer, p, 0.10 * r[:, i], color, width=0.008)

    # External force and restoring force use their physical directions. Moment
    # arrows show only the torque-axis direction (not a point force).
    _add_arrow(viewer, p + np.array([0.0, 0.0, 0.05]), 0.025 * w_ext[:3], np.array([1.0, 0.1, 0.1, 1.0]), width=0.018)
    _add_arrow(viewer, p + np.array([0.0, 0.0, -0.05]), 0.025 * w_cmd[:3], np.array([1.0, 0.55, 0.05, 1.0]), width=0.018)
    _add_arrow(viewer, p + np.array([0.0, 0.07, 0.0]), 0.10 * w_ext[3:], np.array([0.8, 0.2, 0.9, 1.0]), width=0.014)
    _add_arrow(viewer, p + np.array([0.0, -0.07, 0.0]), 0.10 * w_cmd[3:], np.array([0.1, 0.8, 0.9, 1.0]), width=0.014)


def replay_viewer(model_path: Path, mode: str) -> None:
    import mujoco.viewer

    adapter = MuJoCoAdapter.from_xml_path(str(model_path))
    disable_builtin_position_actuators(adapter.model)
    set_home(adapter)
    translation, full = build_controllers()
    p_des, r_des = adapter.get_tcp_pose(frame="base")

    print("\n=== Task05 stage-2 viewer ===")
    print(f"Mode          : {mode}")
    print("Green sphere  : desired TCP position")
    print("Yellow sphere : actual TCP position")
    print("RGB arrows    : current TCP axes")
    print("Red arrow     : external force")
    print("Orange arrow  : impedance restoring force")
    print("Purple arrow  : external moment axis")
    print("Cyan arrow    : impedance restoring moment axis")
    print("The disturbance cycle repeats. Close the viewer to stop.")

    with mujoco.viewer.launch_passive(adapter.model, adapter.data) as viewer:
        viewer.cam.lookat[:] = p_des
        viewer.cam.distance = 1.25
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0

        last_wall = time.time()
        while viewer.is_running():
            if adapter.data.time >= DURATION:
                mujoco.mj_resetData(adapter.model, adapter.data)
                set_home(adapter)

            t = float(adapter.data.time)
            qdot = adapter.get_qdot()
            p, r = adapter.get_tcp_pose(frame="base")
            j = adapter.get_jacobian(frame="base")
            twist = j @ qdot
            w_cmd = command_wrench(mode, translation, full, p, r, twist[:3], twist[3:], p_des, r_des)
            w_ext = external_wrench(t)
            tau = j.T @ (w_cmd + w_ext) + adapter.get_gravity()
            tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)

            adapter.set_joint_torque(tau)
            adapter.step()
            draw_viewer_geometry(viewer, p, r, p_des, r_des, w_ext, w_cmd)
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

    results = {mode: simulate(model_path, mode) for mode in MODES}
    print_metrics(results)
    plot_path = save_comparison_plot(results, output_dir)
    csv_paths = [save_csv(results[mode], output_dir) for mode in MODES]

    print(f"\nComparison plot: {plot_path}")
    for path in csv_paths:
        print(f"CSV            : {path}")

    if not all(np.isfinite(result.tau_total).all() for result in results.values()):
        raise RuntimeError("Non-finite result detected")

    print("\nPASS: both translation-only and full-6D impedance simulations completed without NaN/Inf.")
    print("Acceptance target: full_6d should reduce orientation drift under the same external moment disturbance.")

    if args.viewer:
        replay_viewer(model_path, args.viewer_mode)


if __name__ == "__main__":
    main()
