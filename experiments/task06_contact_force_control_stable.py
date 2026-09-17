"""Task06 stage 1B: stabilized real-contact normal-force control.

This experiment keeps the same P-vs-PI teaching goal as stage 1, but applies
stabilization only after contact has been established:

1) the original, proven Cartesian-impedance approach is kept unchanged;
2) measured normal force is low-pass filtered in force-control mode;
3) the push-force command is slew-rate limited;
4) P/PI gains are reduced relative to stage 1.

It also reports force ripple/std, contact-loss ratio, and a sustained settling
time so a good average force cannot hide unstable contact chatter.
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

from controllers.force_pi import ForcePIController
from sim.mujoco_adapter import MuJoCoAdapter
from experiments.task06_contact_force_control import (
    HOME_Q,
    TORQUE_LIMITS,
    PROBE_RADIUS,
    INITIAL_GAP,
    CONTACT_THRESHOLD,
    F_DES,
    disable_builtin_position_actuators,
    set_home,
    resolve_fr3_xml,
    build_contact_scene,
    contact_normal_force,
    build_impedance_controllers,
)


# Important: keep the same proven approach depth as Stage 1.  Stabilization is
# a post-contact concern; weakening the approach can prevent contact entirely.
APPROACH_EXTRA = 0.012

FORCE_COMMAND_LIMIT = 16.0
INITIAL_FORCE_COMMAND = 4.0
FORCE_FILTER_TAU = 0.040
COMMAND_SLEW_RATE = 20.0  # N/s
DURATION = 7.0

SETTLE_BAND = 0.5
SETTLE_HOLD = 0.25
STABILITY_TAIL = 1.0
CONTACT_LOSS_FORCE = 0.05

MODES = ("p", "pi")


@dataclass
class RunResult:
    mode: str
    time: np.ndarray
    normal_force_raw: np.ndarray
    normal_force_filtered: np.ndarray
    force_command: np.ndarray
    force_error: np.ndarray
    tcp_z: np.ndarray
    clearance: np.ndarray
    tau_total: np.ndarray
    contact_time: float
    surface_z: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to Menagerie FR3 scene.xml or fr3.xml")
    parser.add_argument("--viewer", action="store_true", help="Replay stabilized PI case")
    return parser.parse_args()


def build_force_controller(mode: str) -> ForcePIController:
    if mode == "p":
        return ForcePIController(
            kp=0.8,
            ki=0.0,
            command_min=0.5,
            command_max=FORCE_COMMAND_LIMIT,
        )
    if mode == "pi":
        return ForcePIController(
            kp=0.8,
            ki=0.35,
            command_min=0.5,
            command_max=FORCE_COMMAND_LIMIT,
            integral_limit=16.0,
        )
    raise ValueError(mode)


def low_pass(previous: float, sample: float, dt: float) -> float:
    alpha = dt / (FORCE_FILTER_TAU + dt)
    return float(previous + alpha * (sample - previous))


def slew_limit(previous: float, desired: float, dt: float) -> float:
    max_delta = COMMAND_SLEW_RATE * dt
    return float(previous + np.clip(desired - previous, -max_delta, max_delta))


def simulate_case(contact_xml: Path, mode: str, surface_z: float) -> RunResult:
    adapter = MuJoCoAdapter.from_xml_path(str(contact_xml))
    disable_builtin_position_actuators(adapter.model)
    set_home(adapter)

    probe_id = mujoco.mj_name2id(adapter.model, mujoco.mjtObj.mjOBJ_GEOM, "task06_probe")
    surface_id = mujoco.mj_name2id(adapter.model, mujoco.mjtObj.mjOBJ_GEOM, "task06_surface")
    if probe_id < 0 or surface_id < 0:
        raise RuntimeError("Task06 contact geoms missing from generated model")

    p_home, r_home = adapter.get_tcp_pose(frame="base")
    approach_target = p_home + np.array([0.0, 0.0, -(INITIAL_GAP + APPROACH_EXTRA)])
    approach_controller, hold_controller = build_impedance_controllers()
    force_controller = build_force_controller(mode)

    dt = float(adapter.model.opt.timestep)
    steps = int(np.ceil(DURATION / dt)) + 1

    time_log = np.zeros(steps)
    force_raw_log = np.zeros(steps)
    force_filt_log = np.zeros(steps)
    command_log = np.zeros(steps)
    error_log = np.zeros(steps)
    z_log = np.zeros(steps)
    clearance_log = np.zeros(steps)
    tau_log = np.zeros((steps, 7))

    in_force_control = False
    contact_time = float("nan")
    filtered_force = 0.0
    force_command_state = 0.0
    min_clearance = float("inf")

    for k in range(steps):
        t = float(adapter.data.time)
        p, r = adapter.get_tcp_pose(frame="base")
        j = adapter.get_jacobian(frame="base")
        twist = j @ adapter.get_qdot()
        measured_force = contact_normal_force(adapter.model, adapter.data, probe_id, surface_id)
        clearance = float(p[2] - (surface_z + PROBE_RADIUS))
        min_clearance = min(min_clearance, clearance)

        # Contact detection deliberately uses the RAW contact force.  Filtering
        # is only introduced after the physical contact has been established.
        if not in_force_control and measured_force >= CONTACT_THRESHOLD:
            in_force_control = True
            contact_time = t
            force_controller.reset()
            filtered_force = measured_force
            force_command_state = float(
                np.clip(max(INITIAL_FORCE_COMMAND, measured_force), 0.5, F_DES)
            )

        if not in_force_control:
            # Keep Stage-1 approach behavior: no force-loop filtering/slew logic
            # is allowed to weaken the motion toward the surface.
            filtered_force = measured_force
            wrench_cmd = approach_controller.compute(
                position=p,
                rotation=r,
                linear_velocity=twist[:3],
                angular_velocity=twist[3:],
                desired_position=approach_target,
                desired_rotation=r_home,
            )
            force_command = 0.0
            force_error = F_DES - measured_force
        else:
            filtered_force = low_pass(filtered_force, measured_force, dt)

            # X/Y and orientation are held by impedance.  Z position stiffness
            # is zero in hold_controller; the normal direction is force-controlled.
            hold_target = np.array([p_home[0], p_home[1], p[2]])
            wrench_cmd = hold_controller.compute(
                position=p,
                rotation=r,
                linear_velocity=twist[:3],
                angular_velocity=twist[3:],
                desired_position=hold_target,
                desired_rotation=r_home,
            )

            force_error = F_DES - filtered_force
            desired_command = force_controller.compute(force_error, dt)
            force_command_state = slew_limit(force_command_state, desired_command, dt)
            force_command = force_command_state
            wrench_cmd[2] += -force_command

        tau = j.T @ wrench_cmd + adapter.get_gravity()
        tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)

        time_log[k] = t
        force_raw_log[k] = measured_force
        force_filt_log[k] = filtered_force
        command_log[k] = force_command
        error_log[k] = force_error
        z_log[k] = p[2]
        clearance_log[k] = clearance
        tau_log[k] = tau

        if k < steps - 1:
            adapter.set_joint_torque(tau)
            adapter.step()

        if not np.isfinite(adapter.data.qpos).all() or not np.isfinite(adapter.data.qvel).all():
            raise RuntimeError(f"NaN/Inf in Task06 stable mode={mode} at t={t:.4f}s")

    if not np.isfinite(contact_time):
        raise RuntimeError(
            f"Mode {mode} never established contact. "
            f"Closest probe clearance was {min_clearance*1000.0:.3f} mm."
        )

    return RunResult(
        mode=mode,
        time=time_log,
        normal_force_raw=force_raw_log,
        normal_force_filtered=force_filt_log,
        force_command=command_log,
        force_error=error_log,
        tcp_z=z_log,
        clearance=clearance_log,
        tau_total=tau_log,
        contact_time=contact_time,
        surface_z=surface_z,
    )


def post_contact_mask(result: RunResult, start_after: float = 0.0) -> np.ndarray:
    return result.time >= result.contact_time + start_after


def settle_time(result: RunResult) -> float:
    mask = post_contact_mask(result)
    t = result.time[mask] - result.contact_time
    f = result.normal_force_filtered[mask]
    if len(t) < 2:
        return float("nan")

    dt = float(np.median(np.diff(t)))
    window = max(1, int(np.ceil(SETTLE_HOLD / dt)))
    within = np.abs(f - F_DES) <= SETTLE_BAND

    for i in range(0, max(0, len(within) - window + 1)):
        if np.all(within[i : i + window]):
            return float(t[i])
    return float("nan")


def force_metrics(result: RunResult) -> dict[str, float]:
    mask = post_contact_mask(result)
    t = result.time[mask] - result.contact_time
    f_raw = result.normal_force_raw[mask]
    f_filt = result.normal_force_filtered[mask]
    if len(t) == 0:
        raise RuntimeError("No post-contact samples")

    reached = np.flatnonzero(f_filt >= 0.9 * F_DES)
    rise90 = float(t[reached[0]]) if len(reached) else float("nan")
    overshoot = max(0.0, float(np.max(f_filt) - F_DES)) / F_DES * 100.0

    tail_start = max(result.contact_time + 0.5, DURATION - STABILITY_TAIL)
    tail_mask = result.time >= tail_start
    tail_raw = result.normal_force_raw[tail_mask]

    steady_error = float(F_DES - np.mean(tail_raw))
    force_std = float(np.std(tail_raw))
    force_ripple = float(np.ptp(tail_raw))

    stable_mask = result.time >= result.contact_time + 0.2
    lost = result.normal_force_raw[stable_mask] <= CONTACT_LOSS_FORCE
    contact_loss_ratio = 100.0 * float(np.mean(lost)) if np.any(stable_mask) else float("nan")

    return {
        "rise90": rise90,
        "settle": settle_time(result),
        "overshoot": overshoot,
        "steady_error": steady_error,
        "force_std": force_std,
        "force_ripple": force_ripple,
        "contact_loss_ratio": contact_loss_ratio,
        "peak_force_raw": float(np.max(f_raw)),
        "peak_tau": float(np.max(np.abs(result.tau_total))),
    }


def save_csv(result: RunResult, output_dir: Path) -> Path:
    path = output_dir / f"{result.mode}_stable.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        headers = [
            "time",
            "normal_force_raw",
            "normal_force_filtered",
            "force_command",
            "force_error",
            "tcp_z",
            "clearance",
        ]
        headers += [f"tau{i}" for i in range(1, 8)]
        writer.writerow(headers)
        for k in range(len(result.time)):
            writer.writerow([
                result.time[k],
                result.normal_force_raw[k],
                result.normal_force_filtered[k],
                result.force_command[k],
                result.force_error[k],
                result.tcp_z[k],
                result.clearance[k],
                *result.tau_total[k],
            ])
    return path


def save_main_plot(results: dict[str, RunResult], output_dir: Path) -> Path:
    path = output_dir / "p_vs_pi_force_control_stable.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for col, mode in enumerate(MODES):
        result = results[mode]
        mask = post_contact_mask(result)
        t = result.time[mask] - result.contact_time
        axes[0, col].plot(t, result.normal_force_raw[mask], alpha=0.35, label="raw F_meas")
        axes[0, col].plot(t, result.normal_force_filtered[mask], label="filtered F_meas")
        axes[0, col].axhline(F_DES, linestyle="--", label="F_des")
        axes[0, col].set_title(f"{mode.upper()} force response")
        axes[0, col].set_ylabel("normal force [N]")
        axes[0, col].set_xlabel("time since contact [s]")
        axes[0, col].grid(True)
        axes[0, col].legend()

    for mode in MODES:
        result = results[mode]
        mask = post_contact_mask(result)
        t = result.time[mask] - result.contact_time
        axes[1, 0].plot(t, result.force_command[mask], label=mode.upper())
        axes[1, 1].plot(t, 1000.0 * result.clearance[mask], label=mode.upper())

    axes[1, 0].set_title("Push-force command")
    axes[1, 0].set_ylabel("command [N]")
    axes[1, 0].set_xlabel("time since contact [s]")
    axes[1, 1].set_title("Probe-plane clearance")
    axes[1, 1].set_ylabel("clearance [mm]")
    axes[1, 1].set_xlabel("time since contact [s]")
    for ax in axes[1, :]:
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_steady_zoom(results: dict[str, RunResult], output_dir: Path) -> Path:
    path = output_dir / "steady_force_zoom.png"
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

    for ax, mode in zip(axes, MODES):
        result = results[mode]
        end = float(result.time[-1])
        start = end - STABILITY_TAIL
        mask = result.time >= start
        t = result.time[mask] - result.contact_time
        ax.plot(t, result.normal_force_raw[mask], alpha=0.55, label="raw F_meas")
        ax.plot(t, result.normal_force_filtered[mask], label="filtered F_meas")
        ax.axhline(F_DES, linestyle="--", label="F_des")
        ax.set_ylabel("force [N]")
        ax.set_title(f"{mode.upper()} last {STABILITY_TAIL:.1f} s")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("time since contact [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _next_geom(viewer):
    scene = viewer.user_scn
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("viewer user scene full")
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    return geom


def _add_arrow(viewer, start: np.ndarray, vector: np.ndarray, rgba: np.ndarray) -> None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
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
    length = float(np.clip(0.018 * norm, 0.06, 0.28))
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.016,
        np.asarray(start, dtype=float),
        np.asarray(start + length * direction, dtype=float),
    )


def show_viewer(contact_xml: Path) -> None:
    import mujoco.viewer

    adapter = MuJoCoAdapter.from_xml_path(str(contact_xml))
    disable_builtin_position_actuators(adapter.model)
    set_home(adapter)

    probe_id = mujoco.mj_name2id(adapter.model, mujoco.mjtObj.mjOBJ_GEOM, "task06_probe")
    surface_id = mujoco.mj_name2id(adapter.model, mujoco.mjtObj.mjOBJ_GEOM, "task06_surface")

    p_home, r_home = adapter.get_tcp_pose(frame="base")
    approach_target = p_home + np.array([0.0, 0.0, -(INITIAL_GAP + APPROACH_EXTRA)])
    approach_controller, hold_controller = build_impedance_controllers()
    force_controller = build_force_controller("pi")

    in_force_control = False
    filtered_force = 0.0
    force_command_state = 0.0
    dt = float(adapter.model.opt.timestep)

    print("\n=== Task06 stabilized viewer ===")
    print("Yellow sphere : physical TCP contact probe")
    print("Red arrow     : rate-limited commanded push force (-BASE-Z)")
    print("Cyan arrow    : raw measured normal force (+BASE-Z)")
    print("PI feedback uses low-pass filtered force.")

    with mujoco.viewer.launch_passive(adapter.model, adapter.data) as viewer:
        viewer.cam.lookat[:] = p_home
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0

        while viewer.is_running() and adapter.data.time < DURATION:
            p, r = adapter.get_tcp_pose(frame="base")
            j = adapter.get_jacobian(frame="base")
            twist = j @ adapter.get_qdot()
            measured_force = contact_normal_force(adapter.model, adapter.data, probe_id, surface_id)

            if not in_force_control and measured_force >= CONTACT_THRESHOLD:
                in_force_control = True
                force_controller.reset()
                filtered_force = measured_force
                force_command_state = float(
                    np.clip(max(INITIAL_FORCE_COMMAND, measured_force), 0.5, F_DES)
                )

            force_command = 0.0
            if not in_force_control:
                filtered_force = measured_force
                wrench_cmd = approach_controller.compute(
                    p, r, twist[:3], twist[3:], approach_target, r_home
                )
            else:
                filtered_force = low_pass(filtered_force, measured_force, dt)
                hold_target = np.array([p_home[0], p_home[1], p[2]])
                wrench_cmd = hold_controller.compute(
                    p, r, twist[:3], twist[3:], hold_target, r_home
                )
                desired_command = force_controller.compute(F_DES - filtered_force, dt)
                force_command_state = slew_limit(force_command_state, desired_command, dt)
                force_command = force_command_state
                wrench_cmd[2] += -force_command

            tau = j.T @ wrench_cmd + adapter.get_gravity()
            tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
            adapter.set_joint_torque(tau)
            adapter.step()

            viewer.user_scn.ngeom = 0
            _add_arrow(
                viewer,
                p + np.array([0.06, 0.0, 0.0]),
                np.array([0.0, 0.0, -force_command]),
                np.array([1.0, 0.15, 0.1, 1.0]),
            )
            _add_arrow(
                viewer,
                p + np.array([-0.06, 0.0, 0.0]),
                np.array([0.0, 0.0, measured_force]),
                np.array([0.1, 0.9, 0.95, 1.0]),
            )
            viewer.sync()
            time.sleep(max(0.0, dt))


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    fr3_xml = resolve_fr3_xml(model_path)

    output_dir = ROOT / "outputs" / "task06"
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_xml = output_dir / "task06_contact_scene_stable.xml"
    surface_z, home_tcp_z = build_contact_scene(fr3_xml, contact_xml)

    results = {mode: simulate_case(contact_xml, mode, surface_z) for mode in MODES}
    metrics = {mode: force_metrics(result) for mode, result in results.items()}

    print("\n=== Task06 stage 1B: stabilized contact-force control ===")
    print("Contact geometry       : spherical TCP probe against horizontal plane")
    print(f"Desired normal force   : {F_DES:.2f} N")
    print("Approach               : Stage-1 impedance restored; stabilization starts after contact")
    print("Feedback signal        : low-pass filtered MuJoCo contact normal force")
    print(f"Force filter tau       : {FORCE_FILTER_TAU:.3f} s")
    print(f"Command slew rate      : {COMMAND_SLEW_RATE:.1f} N/s")
    print("P gains                : Kp=0.8, Ki=0")
    print("PI gains               : Kp=0.8, Ki=0.35")
    print(f"HOME TCP z             : {home_tcp_z:.5f} m")
    print(f"Surface z              : {surface_z:.5f} m")

    print("\nTracking metrics")
    print("mode   contact[s]   rise90_filt[s]   settle+/-0.5N[s]   overshoot_filt[%]   steady error[N]")
    print("------------------------------------------------------------------------------------------------")
    for mode in MODES:
        m = metrics[mode]
        r = results[mode]
        print(
            f"{mode.upper():<5}  {r.contact_time:>9.4f}   {m['rise90']:>14.4f}   "
            f"{m['settle']:>17.4f}   {m['overshoot']:>17.3f}   {m['steady_error']:>15.4f}"
        )

    print("\nStability metrics (last 1.0 s unless noted)")
    print("mode   force STD[N]   force ripple[N]   contact loss[%]   raw peak force[N]   peak |tau|[Nm]")
    print("------------------------------------------------------------------------------------------------")
    for mode in MODES:
        m = metrics[mode]
        print(
            f"{mode.upper():<5}  {m['force_std']:>12.4f}   {m['force_ripple']:>15.4f}   "
            f"{m['contact_loss_ratio']:>15.3f}   {m['peak_force_raw']:>17.4f}   {m['peak_tau']:>14.4f}"
        )

    main_plot = save_main_plot(results, output_dir)
    zoom_plot = save_steady_zoom(results, output_dir)
    csv_paths = [save_csv(results[mode], output_dir) for mode in MODES]

    print(f"\nMain plot : {main_plot}")
    print(f"Zoom plot : {zoom_plot}")
    for path in csv_paths:
        print(f"CSV       : {path}")
    print(f"Scene     : {contact_xml}")

    print("\nPASS: both stabilized P/PI simulations completed without NaN/Inf.")
    print(
        "Control-quality acceptance: PI should reduce steady error versus P, "
        "with low force STD/ripple, near-zero contact loss, and a sustained settle time."
    )

    if args.viewer:
        show_viewer(contact_xml)


if __name__ == "__main__":
    main()
