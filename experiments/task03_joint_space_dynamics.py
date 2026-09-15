"""Task03: 7-DoF joint-space dynamics control.

Compare:
  1) joint PD + gravity compensation
  2) computed torque / inverse dynamics

Examples:
    python3 experiments/task03_joint_space_dynamics.py \
        --model ~/mujoco_menagerie/franka_fr3/scene.xml

    python3 experiments/task03_joint_space_dynamics.py \
        --model ~/mujoco_menagerie/franka_fr3/scene.xml --viewer
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

from controllers.computed_torque import ComputedTorqueController
from controllers.joint_pd import JointPDController
from sim.mujoco_adapter import FR3_JOINT_NAMES, MuJoCoAdapter


HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
# Small, safe 7-joint motion. Every joint moves, but the trajectory stays well
# inside the Menagerie FR3 joint limits.
DELTA_Q = np.array([0.12, -0.08, 0.10, 0.12, -0.08, 0.10, 0.08])
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

# PD+gravity gains have torque/rad and torque/(rad/s) meaning.
PD_KP = np.array([70.0, 70.0, 65.0, 65.0, 35.0, 30.0, 24.0])
PD_KD = np.array([14.0, 14.0, 12.0, 12.0, 7.0, 6.0, 5.0])

# Computed-torque gains act on the desired acceleration v, so their numerical
# values are not directly comparable with the raw joint-PD gains above.
CT_KP = np.full(7, 36.0)
CT_KD = np.full(7, 12.0)

MODES = ("pd_gravity", "computed_torque")


@dataclass
class RunResult:
    mode: str
    time: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    tau: np.ndarray
    q_des: np.ndarray
    qdot_des: np.ndarray
    qddot_des: np.ndarray
    rms_per_joint: np.ndarray
    max_per_joint: np.ndarray
    peak_tau_per_joint: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to Menagerie FR3 scene.xml or fr3.xml")
    parser.add_argument("--move-time", type=float, default=4.0, help="Quintic motion duration [s]")
    parser.add_argument("--hold-time", type=float, default=2.0, help="Final-pose hold duration [s]")
    parser.add_argument("--viewer", action="store_true", help="Replay the selected controller in MuJoCo")
    parser.add_argument(
        "--viewer-mode",
        choices=MODES,
        default="computed_torque",
        help="Controller used for viewer replay",
    )
    return parser.parse_args()


def disable_builtin_position_actuators(model: mujoco.MjModel) -> None:
    if model.nu == 0:
        return
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_biasprm[:, :] = 0.0


def set_home(adapter: MuJoCoAdapter) -> None:
    for joint, value in zip(adapter.joint_map, HOME_Q):
        adapter.data.qpos[joint.qpos_adr] = float(value)
        adapter.data.qvel[joint.dof_adr] = 0.0
    adapter.clear_joint_torque()
    adapter.forward()


def quintic_reference(t: float, move_time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum-jerk/quintic interpolation from HOME_Q to HOME_Q + DELTA_Q."""
    if t <= 0.0:
        return HOME_Q.copy(), np.zeros(7), np.zeros(7)
    if t >= move_time:
        return HOME_Q + DELTA_Q, np.zeros(7), np.zeros(7)

    u = t / move_time
    s = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    sdot = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / move_time
    sddot = (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / (move_time**2)

    return (
        HOME_Q + DELTA_Q * s,
        DELTA_Q * sdot,
        DELTA_Q * sddot,
    )


def validate_target(adapter: MuJoCoAdapter) -> None:
    target = HOME_Q + DELTA_Q
    for index, (joint, value) in enumerate(zip(adapter.joint_map, target)):
        lower, upper = adapter.model.jnt_range[joint.joint_id]
        if not (lower <= value <= upper):
            raise ValueError(
                f"Target for {FR3_JOINT_NAMES[index]} = {value:.4f} rad is outside "
                f"[{lower:.4f}, {upper:.4f}]"
            )


def make_pd_controller() -> JointPDController:
    return JointPDController.from_gains(
        kp=PD_KP,
        kd=PD_KD,
        dof=7,
        torque_limits=TORQUE_LIMITS,
    )


def make_ct_controller() -> ComputedTorqueController:
    return ComputedTorqueController.from_gains(
        kp=CT_KP,
        kd=CT_KD,
        dof=7,
        torque_limits=TORQUE_LIMITS,
    )


def compute_tau(
    adapter: MuJoCoAdapter,
    mode: str,
    q_des: np.ndarray,
    qdot_des: np.ndarray,
    qddot_des: np.ndarray,
    pd: JointPDController,
    ct: ComputedTorqueController,
) -> np.ndarray:
    q = adapter.get_q()
    qdot = adapter.get_qdot()

    if mode == "pd_gravity":
        return pd.compute(
            q=q,
            qdot=qdot,
            q_des=q_des,
            qdot_des=qdot_des,
            tau_ff=adapter.get_gravity(),
        )
    if mode == "computed_torque":
        return ct.compute(
            q=q,
            qdot=qdot,
            q_des=q_des,
            qdot_des=qdot_des,
            qddot_des=qddot_des,
            mass_matrix=adapter.get_mass_matrix(),
            bias=adapter.get_bias(),
        )
    raise ValueError(f"Unknown mode: {mode}")


def inspect_dynamics(model_path: Path) -> None:
    adapter = MuJoCoAdapter.from_xml_path(str(model_path))
    disable_builtin_position_actuators(adapter.model)
    set_home(adapter)

    m = adapter.get_mass_matrix()
    g = adapter.get_gravity()
    cqd = adapter.get_coriolis_centrifugal()
    symmetry_error = float(np.max(np.abs(m - m.T)))
    eigvals = np.linalg.eigvalsh(0.5 * (m + m.T))

    np.set_printoptions(precision=5, suppress=True)
    print("\n=== Task03 dynamics at HOME ===")
    print(f"M(q) shape              : {m.shape}")
    print("M(q) [kg m^2-like generalized inertia] =")
    print(m)
    print(f"max |M-M^T|             : {symmetry_error:.3e}")
    print(f"eigenvalue range of M   : [{eigvals.min():.6f}, {eigvals.max():.6f}]")
    print(f"g(q) [N m]              : {g}")
    print(f"C(q,qdot)qdot at rest   : {cqd}")

    if m.shape != (7, 7):
        raise RuntimeError(f"Unexpected mass-matrix shape: {m.shape}")
    if symmetry_error > 1e-9:
        raise RuntimeError(f"Mass matrix is unexpectedly asymmetric: {symmetry_error:.3e}")
    if eigvals.min() <= 0.0:
        raise RuntimeError("Mass matrix is not positive definite at HOME")


def simulate(model_path: Path, mode: str, move_time: float, hold_time: float) -> RunResult:
    adapter = MuJoCoAdapter.from_xml_path(str(model_path))
    disable_builtin_position_actuators(adapter.model)
    set_home(adapter)
    validate_target(adapter)

    pd = make_pd_controller()
    ct = make_ct_controller()

    dt = float(adapter.model.opt.timestep)
    duration = move_time + hold_time
    steps = int(np.ceil(duration / dt)) + 1

    time_log = np.zeros(steps)
    q_log = np.zeros((steps, 7))
    qdot_log = np.zeros((steps, 7))
    tau_log = np.zeros((steps, 7))
    qdes_log = np.zeros((steps, 7))
    qdotdes_log = np.zeros((steps, 7))
    qddotdes_log = np.zeros((steps, 7))

    for k in range(steps):
        t = float(adapter.data.time)
        q_des, qdot_des, qddot_des = quintic_reference(t, move_time)
        tau = compute_tau(adapter, mode, q_des, qdot_des, qddot_des, pd, ct)

        time_log[k] = t
        q_log[k] = adapter.get_q()
        qdot_log[k] = adapter.get_qdot()
        tau_log[k] = tau
        qdes_log[k] = q_des
        qdotdes_log[k] = qdot_des
        qddotdes_log[k] = qddot_des

        if k < steps - 1:
            adapter.set_joint_torque(tau)
            adapter.step()

        if not np.isfinite(adapter.data.qpos).all() or not np.isfinite(adapter.data.qvel).all():
            raise RuntimeError(f"NaN/Inf in {mode} at t={t:.4f}s")

    error = qdes_log - q_log
    rms = np.sqrt(np.mean(error**2, axis=0))
    max_error = np.max(np.abs(error), axis=0)
    peak_tau = np.max(np.abs(tau_log), axis=0)

    return RunResult(
        mode=mode,
        time=time_log,
        q=q_log,
        qdot=qdot_log,
        tau=tau_log,
        q_des=qdes_log,
        qdot_des=qdotdes_log,
        qddot_des=qddotdes_log,
        rms_per_joint=rms,
        max_per_joint=max_error,
        peak_tau_per_joint=peak_tau,
    )


def save_csv(result: RunResult, output_dir: Path) -> Path:
    path = output_dir / f"{result.mode}.csv"
    headers = ["time"]
    headers += [f"q{i}" for i in range(1, 8)]
    headers += [f"q_des{i}" for i in range(1, 8)]
    headers += [f"qdot{i}" for i in range(1, 8)]
    headers += [f"qdot_des{i}" for i in range(1, 8)]
    headers += [f"qddot_des{i}" for i in range(1, 8)]
    headers += [f"tau{i}" for i in range(1, 8)]

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for k in range(len(result.time)):
            writer.writerow(
                [
                    result.time[k],
                    *result.q[k],
                    *result.q_des[k],
                    *result.qdot[k],
                    *result.qdot_des[k],
                    *result.qddot_des[k],
                    *result.tau[k],
                ]
            )
    return path


def save_comparison_plot(results: list[RunResult], output_dir: Path) -> Path:
    path = output_dir / "tracking_comparison.png"
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))

    for result in results:
        error = result.q_des - result.q
        max_abs_error_t = np.max(np.abs(error), axis=1)
        rms_error_t = np.sqrt(np.mean(error**2, axis=1))
        max_abs_tau_t = np.max(np.abs(result.tau), axis=1)
        label = result.mode.replace("_", "+")
        axes[0].plot(result.time, max_abs_error_t, label=label)
        axes[1].plot(result.time, rms_error_t, label=label)
        axes[2].plot(result.time, max_abs_tau_t, label=label)

    axes[0].set_ylabel("max |e_i| [rad]")
    axes[0].set_title("Task03: 7-DoF tracking comparison")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_ylabel("RMS across joints [rad]")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].set_ylabel("max |tau_i| [N m]")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_joint_error_plot(results: list[RunResult], output_dir: Path) -> Path:
    path = output_dir / "computed_torque_joint_errors.png"
    ct = next(result for result in results if result.mode == "computed_torque")
    error = ct.q_des - ct.q

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(FR3_JOINT_NAMES):
        ax.plot(ct.time, error[:, i], label=name)
    ax.set_title("Task03: computed-torque joint tracking errors")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("q_des - q [rad]")
    ax.grid(True)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def print_metrics(results: list[RunResult]) -> None:
    print("\n=== Task03 tracking metrics ===")
    print("Per-joint RMS error [rad]")
    header = "mode             " + "  ".join(f"J{i}" for i in range(1, 8))
    print(header)
    print("-" * len(header))
    for result in results:
        values = "  ".join(f"{v:.5f}" for v in result.rms_per_joint)
        print(f"{result.mode:<16} {values}")

    print("\nSummary")
    print("mode             RMS all joints [rad]   max error [rad]   max |tau| [Nm]")
    print("------------------------------------------------------------------------")
    for result in results:
        overall_rms = float(np.sqrt(np.mean((result.q_des - result.q) ** 2)))
        max_error = float(np.max(result.max_per_joint))
        max_tau = float(np.max(result.peak_tau_per_joint))
        print(f"{result.mode:<16} {overall_rms:>19.6e}   {max_error:>14.6e}   {max_tau:>13.4f}")


def replay_viewer(model_path: Path, mode: str, move_time: float) -> None:
    import mujoco.viewer

    adapter = MuJoCoAdapter.from_xml_path(str(model_path))
    disable_builtin_position_actuators(adapter.model)
    set_home(adapter)
    pd = make_pd_controller()
    ct = make_ct_controller()

    print("\n=== Task03 viewer replay ===")
    print(f"Mode: {mode}")
    print("All seven FR3 joints follow the same smooth quintic time law with different offsets.")
    print("The motion repeats so it is easy to inspect visually. Close the viewer to stop.")

    with mujoco.viewer.launch_passive(adapter.model, adapter.data) as viewer:
        viewer.cam.lookat[:] = np.array([0.25, 0.0, 0.45])
        viewer.cam.distance = 1.55
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0

        wall_start = time.time()
        last_wall = wall_start
        while viewer.is_running():
            elapsed = time.time() - wall_start
            cycle = 2.0 * move_time + 1.0
            phase = elapsed % cycle

            if phase <= move_time:
                q_des, qdot_des, qddot_des = quintic_reference(phase, move_time)
            elif phase <= move_time + 0.5:
                q_des, qdot_des, qddot_des = quintic_reference(move_time, move_time)
            else:
                # Return trajectory: swap endpoints by reusing the same quintic scalar.
                return_t = min(phase - (move_time + 0.5), move_time)
                q_fwd, qdot_fwd, qddot_fwd = quintic_reference(return_t, move_time)
                q_des = HOME_Q + DELTA_Q - (q_fwd - HOME_Q)
                qdot_des = -qdot_fwd
                qddot_des = -qddot_fwd

            tau = compute_tau(adapter, mode, q_des, qdot_des, qddot_des, pd, ct)
            adapter.set_joint_torque(tau)
            adapter.step()
            viewer.sync()

            dt = float(adapter.model.opt.timestep)
            now = time.time()
            sleep_time = dt - (now - last_wall)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            last_wall = time.time()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")
    if args.move_time <= 0.0 or args.hold_time < 0.0:
        raise ValueError("move-time must be > 0 and hold-time must be >= 0")

    inspect_dynamics(model_path)
    results = [
        simulate(model_path, mode, args.move_time, args.hold_time)
        for mode in MODES
    ]
    print_metrics(results)

    output_dir = ROOT / "outputs" / "task03"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = [save_csv(result, output_dir) for result in results]
    comparison_plot = save_comparison_plot(results, output_dir)
    joint_error_plot = save_joint_error_plot(results, output_dir)

    print(f"\nComparison plot: {comparison_plot}")
    print(f"Joint-error plot: {joint_error_plot}")
    for path in csv_paths:
        print(f"CSV             : {path}")

    for result in results:
        if not np.isfinite(result.q).all() or not np.isfinite(result.tau).all():
            raise RuntimeError(f"Non-finite result in {result.mode}")

    print("\nPASS: M(q) is symmetric positive definite at HOME and both 7-DoF controllers completed without NaN/Inf.")
    print("Next acceptance step: compare tracking error and explain M(q), C(q,qdot)qdot, g(q), and model dependence.")

    if args.viewer:
        replay_viewer(model_path, args.viewer_mode, args.move_time)


if __name__ == "__main__":
    main()
