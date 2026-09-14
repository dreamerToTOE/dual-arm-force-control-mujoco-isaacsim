"""Task02 gain-sweep experiment for single-joint PD+gravity control.

Compares the effect of Kp and Kd independently on FR3 joint4 step response.

Example:
    python3 experiments/task02_gain_sweep.py \
        --model ~/mujoco_menagerie/franka_fr3/scene.xml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

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
BASE_KP = np.array([60.0, 60.0, 55.0, 55.0, 30.0, 25.0, 18.0])
BASE_KD = np.array([12.0, 12.0, 10.0, 10.0, 6.0, 5.0, 4.0])


@dataclass
class SweepResult:
    label: str
    gain_value: float
    time: np.ndarray
    q: np.ndarray
    tau: np.ndarray
    q_des: float
    metrics: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to Menagerie FR3 scene.xml or fr3.xml")
    parser.add_argument("--joint", type=int, default=4, choices=range(1, 8), help="Joint to step")
    parser.add_argument("--step", type=float, default=0.20, help="Target angle step [rad]")
    parser.add_argument("--duration", type=float, default=6.0, help="Simulation duration per run [s]")
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


def compute_metrics(time_log: np.ndarray, q_log: np.ndarray, q0: float, q_des: float) -> dict[str, float]:
    step = q_des - q0
    direction = 1.0 if step >= 0.0 else -1.0
    magnitude = abs(step)
    progress = direction * (q_log - q0)

    def first_crossing(level: float) -> float:
        idx = np.where(progress >= level * magnitude)[0]
        return float(time_log[idx[0]]) if len(idx) else float("nan")

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


def simulate(
    model_path: Path,
    joint_index: int,
    step: float,
    duration: float,
    kp_joint: float,
    kd_joint: float,
    label: str,
    gain_value: float,
) -> SweepResult:
    adapter = MuJoCoAdapter.from_xml_path(str(model_path))
    disable_builtin_position_actuators(adapter.model)
    set_home(adapter)

    kp = BASE_KP.copy()
    kd = BASE_KD.copy()
    kp[joint_index] = kp_joint
    kd[joint_index] = kd_joint
    controller = JointPDController.from_gains(kp, kd, dof=7, torque_limits=TORQUE_LIMITS)

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
    q_log = np.zeros(steps)
    tau_log = np.zeros(steps)

    for k in range(steps):
        q = adapter.get_q()
        qdot = adapter.get_qdot()
        tau = controller.compute(q, qdot, q_des, tau_ff=adapter.get_gravity())

        time_log[k] = adapter.data.time
        q_log[k] = q[joint_index]
        tau_log[k] = tau[joint_index]

        adapter.set_joint_torque(tau)
        adapter.step()

        if not np.isfinite(adapter.data.qpos).all() or not np.isfinite(adapter.data.qvel).all():
            raise RuntimeError(f"NaN/Inf detected in {label} run")

    metrics = compute_metrics(time_log, q_log, HOME_Q[joint_index], q_des[joint_index])
    return SweepResult(label, gain_value, time_log, q_log, tau_log, q_des[joint_index], metrics)


def plot_sweep(results: list[SweepResult], title: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    for result in results:
        axes[0].plot(result.time, result.q, label=result.label)
        axes[1].plot(result.time, result.tau, label=result.label)

    axes[0].axhline(results[0].q_des, linestyle="--", label="q_des")
    axes[0].set_ylabel("q [rad]")
    axes[0].set_title(title)
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_ylabel("tau [N m]")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def print_table(name: str, results: list[SweepResult]) -> None:
    print(f"\n=== {name} sweep ===")
    print("case          rise[s]   overshoot[%]   settle[s]   steady error[rad]   peak |tau|[Nm]")
    print("------------------------------------------------------------------------------------")
    for result in results:
        m = result.metrics
        peak_tau = float(np.max(np.abs(result.tau)))
        print(
            f"{result.label:<13} "
            f"{m['rise_time_s']:>8.4f}   "
            f"{m['overshoot_pct']:>12.3f}   "
            f"{m['settling_time_s']:>9.4f}   "
            f"{m['steady_state_error_rad']:>17.6e}   "
            f"{peak_tau:>13.4f}"
        )


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

    joint_index = args.joint - 1
    base_kp = float(BASE_KP[joint_index])
    base_kd = float(BASE_KD[joint_index])

    kp_values = [0.45 * base_kp, base_kp, 2.0 * base_kp]
    kd_values = [0.20 * base_kd, base_kd, 2.5 * base_kd]

    kp_results = [
        simulate(
            model_path,
            joint_index,
            args.step,
            args.duration,
            kp_joint=value,
            kd_joint=base_kd,
            label=f"Kp={value:.1f}",
            gain_value=value,
        )
        for value in kp_values
    ]

    kd_results = [
        simulate(
            model_path,
            joint_index,
            args.step,
            args.duration,
            kp_joint=base_kp,
            kd_joint=value,
            label=f"Kd={value:.1f}",
            gain_value=value,
        )
        for value in kd_values
    ]

    output_dir = ROOT / "outputs" / "task02"
    output_dir.mkdir(parents=True, exist_ok=True)
    kp_plot = output_dir / "kp_sweep.png"
    kd_plot = output_dir / "kd_sweep.png"

    plot_sweep(kp_results, f"Task02 Kp sweep: {FR3_JOINT_NAMES[joint_index]} (Kd={base_kd:.1f})", kp_plot)
    plot_sweep(kd_results, f"Task02 Kd sweep: {FR3_JOINT_NAMES[joint_index]} (Kp={base_kp:.1f})", kd_plot)

    print("\n=== Task02 gain sweep ===")
    print(f"Joint : {FR3_JOINT_NAMES[joint_index]}")
    print(f"q0    : {HOME_Q[joint_index]: .6f} rad")
    print(f"step  : {args.step:+.6f} rad")
    print("All runs use PD + gravity compensation; only the named target-joint gain changes.")

    print_table("Kp", kp_results)
    print_table("Kd", kd_results)

    print(f"\nKp plot: {kp_plot}")
    print(f"Kd plot: {kd_plot}")
    print("\nPASS: gain-sweep simulations completed without NaN/Inf.")
    print("Interpretation target: Kp mainly changes stiffness/speed and steady error; Kd mainly changes damping/overshoot/speed.")


if __name__ == "__main__":
    main()
