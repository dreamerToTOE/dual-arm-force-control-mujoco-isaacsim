"""Task02: single-joint torque control with P, PD, and PD+gravity compensation.

Examples:
    python3 experiments/task02_single_joint_pd.py \
        --model ~/mujoco_menagerie/franka_fr3/scene.xml

    python3 experiments/task02_single_joint_pd.py \
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

from controllers.joint_pd import JointPDController
from sim.mujoco_adapter import FR3_JOINT_NAMES, MuJoCoAdapter


HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
DEFAULT_KP = np.array([60.0, 60.0, 55.0, 55.0, 30.0, 25.0, 18.0])
DEFAULT_KD = np.array([12.0, 12.0, 10.0, 10.0, 6.0, 5.0, 4.0])
MODES = ("p", "pd", "pd_gravity")


@dataclass
class RunResult:
    mode: str
    time: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    tau: np.ndarray
    q_des: np.ndarray
    metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to Menagerie FR3 scene.xml or fr3.xml")
    parser.add_argument("--joint", type=int, default=4, choices=range(1, 8), help="Joint to step")
    parser.add_argument("--step", type=float, default=0.20, help="Target angle step [rad]")
    parser.add_argument("--duration", type=float, default=4.0, help="Simulation duration per mode [s]")
    parser.add_argument("--viewer", action="store_true", help="Replay PD+gravity in MuJoCo viewer")
    parser.add_argument(
        "--viewer-mode",
        choices=MODES,
        default="pd_gravity",
        help="Controller mode used for the optional viewer replay",
    )
    return parser.parse_args()


def disable_builtin_position_actuators(model: mujoco.MjModel) -> None:
    if model.nu == 0:
        return
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_biasprm[:, :] = 0.0


def set_home(adapter: MuJoCoAdapter) -> None:
    for joint, value in zip(adapter.joint_map, HOME_Q):
        adapter.data.qpos[joint.qpos_adr] = value
        adapter.data.qvel[joint.dof_adr] = 0.0
    adapter.clear_joint_torque()
    adapter.forward()


def make_controller(mode: str) -> JointPDController:
    kd = np.zeros(7) if mode == "p" else DEFAULT_KD
    return JointPDController.from_gains(
        kp=DEFAULT_KP,
        kd=kd,
        dof=7,
        torque_limits=TORQUE_LIMITS,
    )


def compute_metrics(time_log: np.ndarray, q_log: np.ndarray, q0: float, q_des: float) -> dict[str, float]:
    step = q_des - q0
    direction = 1.0 if step >= 0.0 else -1.0
    magnitude = abs(step)
    progress = direction * (q_log - q0)

    def first_crossing(level: float) -> float:
        indices = np.where(progress >= level * magnitude)[0]
        return float(time_log[indices[0]]) if len(indices) else float("nan")

    t10 = first_crossing(0.10)
    t90 = first_crossing(0.90)
    rise_time = t90 - t10 if np.isfinite(t10) and np.isfinite(t90) else float("nan")

    overshoot_amount = max(0.0, float(np.max(direction * (q_log - q_des))))
    overshoot_pct = 100.0 * overshoot_amount / magnitude if magnitude > 0.0 else 0.0

    error = q_des - q_log
    tail_count = max(1, int(0.10 * len(error)))
    steady_state_error = float(np.mean(error[-tail_count:]))

    band = max(0.02 * magnitude, 1e-3)
    outside = np.where(np.abs(error) > band)[0]
    if len(outside) == 0:
        settling_time = 0.0
    elif outside[-1] >= len(time_log) - 1:
        settling_time = float("nan")
    else:
        settling_time = float(time_log[outside[-1] + 1])

    return {
        "rise_time_s": rise_time,
        "overshoot_pct": overshoot_pct,
        "settling_time_s": settling_time,
        "steady_state_error_rad": steady_state_error,
    }


def simulate_mode(model_path: Path, mode: str, joint_index: int, step: float, duration: float) -> RunResult:
    adapter = MuJoCoAdapter.from_xml_path(str(model_path))
    disable_builtin_position_actuators(adapter.model)
    set_home(adapter)

    controller = make_controller(mode)
    q_des = HOME_Q.copy()
    q_des[joint_index] += step

    lower, upper = adapter.model.jnt_range[adapter.joint_map[joint_index].joint_id]
    if not (lower <= q_des[joint_index] <= upper):
        raise ValueError(
            f"Target {q_des[joint_index]:.4f} rad is outside {FR3_JOINT_NAMES[joint_index]} "
            f"range [{lower:.4f}, {upper:.4f}]"
        )

    dt = float(adapter.model.opt.timestep)
    steps = int(np.ceil(duration / dt))
    time_log = np.zeros(steps)
    q_log = np.zeros((steps, 7))
    qdot_log = np.zeros((steps, 7))
    tau_log = np.zeros((steps, 7))

    for k in range(steps):
        q = adapter.get_q()
        qdot = adapter.get_qdot()
        tau_ff = adapter.get_gravity() if mode == "pd_gravity" else None
        tau = controller.compute(q, qdot, q_des, tau_ff=tau_ff)

        time_log[k] = adapter.data.time
        q_log[k] = q
        qdot_log[k] = qdot
        tau_log[k] = tau

        adapter.set_joint_torque(tau)
        adapter.step()

        if not np.isfinite(adapter.data.qpos).all() or not np.isfinite(adapter.data.qvel).all():
            raise RuntimeError(f"NaN/Inf detected in mode {mode} at t={adapter.data.time:.4f}s")

    metrics = compute_metrics(
        time_log,
        q_log[:, joint_index],
        HOME_Q[joint_index],
        q_des[joint_index],
    )
    return RunResult(mode, time_log, q_log, qdot_log, tau_log, q_des, metrics)


def save_csv(result: RunResult, output_dir: Path) -> Path:
    path = output_dir / f"{result.mode}.csv"
    headers = ["time"]
    headers += [f"q{i}" for i in range(1, 8)]
    headers += [f"qdot{i}" for i in range(1, 8)]
    headers += [f"tau{i}" for i in range(1, 8)]

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for k in range(len(result.time)):
            writer.writerow(
                [result.time[k], *result.q[k], *result.qdot[k], *result.tau[k]]
            )
    return path


def save_plot(results: list[RunResult], joint_index: int, output_dir: Path) -> Path:
    path = output_dir / "single_joint_comparison.png"
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))

    for result in results:
        label = result.mode.replace("_", "+")
        axes[0].plot(result.time, result.q[:, joint_index], label=label)
        axes[1].plot(
            result.time,
            result.q_des[joint_index] - result.q[:, joint_index],
            label=label,
        )
        axes[2].plot(result.time, result.tau[:, joint_index], label=label)

    axes[0].axhline(results[0].q_des[joint_index], linestyle="--", label="q_des")
    axes[0].set_ylabel("q [rad]")
    axes[0].set_title(f"Task02: {FR3_JOINT_NAMES[joint_index]} step response")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_ylabel("error [rad]")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].set_ylabel("tau [N m]")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def replay_viewer(model_path: Path, mode: str, joint_index: int, step: float) -> None:
    import mujoco.viewer

    adapter = MuJoCoAdapter.from_xml_path(str(model_path))
    disable_builtin_position_actuators(adapter.model)
    set_home(adapter)
    controller = make_controller(mode)

    q_des_home = HOME_Q.copy()
    q_des_step = HOME_Q.copy()
    q_des_step[joint_index] += step

    print("\n=== Task02 viewer replay ===")
    print(f"Mode        : {mode}")
    print(f"Target joint: {FR3_JOINT_NAMES[joint_index]}")
    print(f"Step        : {step:+.3f} rad")
    print("The robot holds HOME for 1 s, then the target joint receives the angle step.")
    print("Close the MuJoCo window to return to the terminal.")

    with mujoco.viewer.launch_passive(adapter.model, adapter.data) as viewer:
        viewer.cam.lookat[:] = np.array([0.30, 0.0, 0.45])
        viewer.cam.distance = 1.45
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0

        wall_start = time.time()
        last_wall = wall_start
        while viewer.is_running():
            elapsed = time.time() - wall_start
            q_des = q_des_home if elapsed < 1.0 else q_des_step

            q = adapter.get_q()
            qdot = adapter.get_qdot()
            tau_ff = adapter.get_gravity() if mode == "pd_gravity" else None
            tau = controller.compute(q, qdot, q_des, tau_ff=tau_ff)
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

    joint_index = args.joint - 1
    output_dir = ROOT / "outputs" / "task02"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        simulate_mode(model_path, mode, joint_index, args.step, args.duration)
        for mode in MODES
    ]

    print("\n=== Task02 single-joint torque-control comparison ===")
    print(f"Joint       : {FR3_JOINT_NAMES[joint_index]}")
    print(f"q0          : {HOME_Q[joint_index]: .6f} rad")
    print(f"q_des       : {results[0].q_des[joint_index]: .6f} rad")
    print(f"step        : {args.step:+.6f} rad")
    print("\nmode        rise[s]   overshoot[%]   settle[s]   steady error[rad]   peak |tau|[Nm]")
    print("--------------------------------------------------------------------------------")

    csv_paths: list[Path] = []
    for result in results:
        m = result.metrics
        peak_tau = float(np.max(np.abs(result.tau[:, joint_index])))
        print(
            f"{result.mode:<11} "
            f"{m['rise_time_s']:>8.4f}   "
            f"{m['overshoot_pct']:>12.3f}   "
            f"{m['settling_time_s']:>9.4f}   "
            f"{m['steady_state_error_rad']:>17.6e}   "
            f"{peak_tau:>13.4f}"
        )
        csv_paths.append(save_csv(result, output_dir))

    plot_path = save_plot(results, joint_index, output_dir)

    print(f"\nPlot: {plot_path}")
    for path in csv_paths:
        print(f"CSV : {path}")

    for result in results:
        if not np.isfinite(result.q).all() or not np.isfinite(result.tau).all():
            raise RuntimeError(f"Non-finite result in mode {result.mode}")

    print("\nPASS: P, PD, and PD+gravity simulations completed without NaN/Inf.")
    print("Next acceptance step: inspect the metrics/plot and explain Kp, Kd, and gravity compensation.")

    if args.viewer:
        replay_viewer(model_path, args.viewer_mode, joint_index, args.step)


if __name__ == "__main__":
    main()
