"""Task00: FR3 state readout and single-joint torque-pulse experiment.

Example:
    python experiments/task00_torque_pulse.py \
        --model ~/mujoco_menagerie/franka_fr3/scene.xml \
        --joint 4 --torque 1.0 --pulse-duration 0.05
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

from sim.mujoco_adapter import FR3_JOINT_NAMES, MuJoCoAdapter


HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to Menagerie FR3 scene.xml or fr3.xml")
    parser.add_argument("--joint", type=int, default=4, choices=range(1, 8), help="FR3 joint number")
    parser.add_argument("--torque", type=float, default=1.0, help="Pulse amplitude in N*m")
    parser.add_argument("--pulse-start", type=float, default=0.25, help="Pulse start time in s")
    parser.add_argument("--pulse-duration", type=float, default=0.05, help="Pulse duration in s")
    parser.add_argument("--duration", type=float, default=1.5, help="Total simulation duration in s")
    parser.add_argument("--gravity", action="store_true", help="Keep gravity enabled (off by default for isolation)")
    parser.add_argument("--show", action="store_true", help="Show plot window after simulation")
    return parser.parse_args()


def disable_builtin_position_actuators(model: mujoco.MjModel) -> None:
    """Disable Menagerie's position servos so they do not fight our torque pulse."""
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


def print_model_summary(adapter: MuJoCoAdapter) -> None:
    model = adapter.model
    print("\n=== FR3 joint map ===")
    print("idx | joint name | joint_id | qpos_adr | dof_adr | range [rad]")
    for index, joint in enumerate(adapter.joint_map):
        joint_range = model.jnt_range[joint.joint_id]
        print(
            f"{index:>3} | {joint.name:<10} | {joint.joint_id:>8} | "
            f"{joint.qpos_adr:>8} | {joint.dof_adr:>7} | "
            f"[{joint_range[0]: .4f}, {joint_range[1]: .4f}]"
        )

    print("\n=== MuJoCo actuators found in model ===")
    if model.nu == 0:
        print("No actuators found.")
    else:
        for actuator_id in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            print(f"actuator[{actuator_id}] = {name}")
        print(
            "NOTE: Menagerie FR3 actuators are position servos. Task00 disables them "
            "and applies torque via qfrc_applied."
        )


def save_csv(path: Path, time: np.ndarray, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["time"]
    headers += [f"q{i}" for i in range(1, 8)]
    headers += [f"qdot{i}" for i in range(1, 8)]
    headers += [f"tau{i}" for i in range(1, 8)]

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for k in range(len(time)):
            writer.writerow([time[k], *q[k], *qdot[k], *tau[k]])


def save_plot(path: Path, time: np.ndarray, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray, joint_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
    axes[0].plot(time, q[:, joint_index])
    axes[0].set_ylabel("q [rad]")
    axes[0].grid(True)

    axes[1].plot(time, qdot[:, joint_index])
    axes[1].set_ylabel("qdot [rad/s]")
    axes[1].grid(True)

    axes[2].plot(time, tau[:, joint_index])
    axes[2].set_ylabel("tau [N m]")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True)

    fig.suptitle(f"Task00 torque pulse: {FR3_JOINT_NAMES[joint_index]}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

    adapter = MuJoCoAdapter.from_xml_path(str(model_path))
    disable_builtin_position_actuators(adapter.model)
    if not args.gravity:
        adapter.model.opt.gravity[:] = 0.0

    set_home(adapter)
    print_model_summary(adapter)
    print(f"\nInitial q     = {adapter.get_q()}")
    print(f"Initial qdot  = {adapter.get_qdot()}")
    print(f"Timestep      = {adapter.model.opt.timestep:.6f} s")

    target = args.joint - 1
    dt = float(adapter.model.opt.timestep)
    steps = int(np.ceil(args.duration / dt))
    pulse_end = args.pulse_start + args.pulse_duration

    time_log = np.zeros(steps)
    q_log = np.zeros((steps, 7))
    qdot_log = np.zeros((steps, 7))
    tau_log = np.zeros((steps, 7))

    for k in range(steps):
        t = float(adapter.data.time)
        tau = np.zeros(7)
        if args.pulse_start <= t < pulse_end:
            tau[target] = args.torque

        adapter.set_joint_torque(tau)
        time_log[k] = t
        q_log[k] = adapter.get_q()
        qdot_log[k] = adapter.get_qdot()
        tau_log[k] = tau
        adapter.step()

        if not np.isfinite(adapter.data.qpos).all() or not np.isfinite(adapter.data.qvel).all():
            raise RuntimeError(f"NaN/Inf detected at t={t:.6f} s")

    output_dir = ROOT / "outputs" / "task00"
    csv_path = output_dir / "torque_pulse.csv"
    plot_path = output_dir / "torque_pulse.png"
    save_csv(csv_path, time_log, q_log, qdot_log, tau_log)
    save_plot(plot_path, time_log, q_log, qdot_log, tau_log, target)

    commanded_peak = np.max(np.abs(tau_log), axis=0)
    moved = np.max(np.abs(q_log - q_log[0]), axis=0)

    print("\n=== Task00 result ===")
    print(f"Torque pulse target : {FR3_JOINT_NAMES[target]}")
    print(f"Peak commanded tau  : {commanded_peak}")
    print(f"Max |q-q0| [rad]    : {moved}")
    print(f"CSV                  : {csv_path}")
    print(f"Plot                 : {plot_path}")
    print("PASS: simulation completed without NaN/Inf.")

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
