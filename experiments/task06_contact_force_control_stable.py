"""Task06 stage 1B: stabilized real-contact normal-force control.

This experiment keeps the same P-vs-PI teaching goal as stage 1 but adds three
practical ingredients needed for contact control:

1) low-pass filtering of measured normal force,
2) a slew-rate limit on the commanded push force,
3) lower P/PI gains.

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
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.cartesian_impedance import CartesianImpedance6D
from controllers.force_pi import ForcePIController
from sim.mujoco_adapter import MuJoCoAdapter


HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

PROBE_RADIUS = 0.025
INITIAL_GAP = 0.015
APPROACH_EXTRA = 0.006
CONTACT_THRESHOLD = 0.5

F_DES = 10.0
FORCE_COMMAND_LIMIT = 16.0
INITIAL_FORCE_COMMAND = 4.0
FORCE_FILTER_TAU = 0.030
COMMAND_SLEW_RATE = 40.0
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
    contact_present: np.ndarray
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


def resolve_fr3_xml(model_path: Path) -> Path:
    if model_path.name == "fr3.xml":
        return model_path
    sibling = model_path.parent / "fr3.xml"
    if sibling.exists():
        return sibling
    raise FileNotFoundError(
        f"Could not find fr3.xml next to {model_path}. Pass Menagerie fr3.xml or scene.xml."
    )


def _find_parent(root: ET.Element, child: ET.Element) -> ET.Element:
    for parent in root.iter():
        for candidate in parent:
            if candidate is child:
                return parent
    raise RuntimeError("Could not find XML parent element")


def build_contact_scene(fr3_xml: Path, output_xml: Path) -> tuple[float, float]:
    base_adapter = MuJoCoAdapter.from_xml_path(str(fr3_xml))
    disable_builtin_position_actuators(base_adapter.model)
    set_home(base_adapter)
    p_world, _ = base_adapter.get_tcp_pose(frame="world")
    surface_z = float(p_world[2] - PROBE_RADIUS - INITIAL_GAP)

    tree = ET.parse(fr3_xml)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str((fr3_xml.parent / "assets").resolve()))

    for default in root.iter("default"):
        if default.get("class") == "collision":
            for geom in default.findall("geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")

    site = None
    for candidate in root.iter("site"):
        if candidate.get("name") == "attachment_site":
            site = candidate
            break
    if site is None:
        raise RuntimeError("attachment_site not found in fr3.xml")

    parent = _find_parent(root, site)
    probe = ET.Element(
        "geom",
        {
            "name": "task06_probe",
            "type": "sphere",
            "pos": site.get("pos", "0 0 0.107"),
            "size": f"{PROBE_RADIUS}",
            "mass": "0.001",
            "rgba": "1 0.8 0.05 1",
            "contype": "1",
            "conaffinity": "1",
            "condim": "3",
            "friction": "0.8 0.005 0.0001",
        },
    )
    parent.insert(list(parent).index(site), probe)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("worldbody not found in fr3.xml")

    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "task06_surface",
            "type": "plane",
            "pos": f"0 0 {surface_z:.9f}",
            "size": "1 1 0.05",
            "rgba": "0.25 0.35 0.45 1",
            "contype": "1",
            "conaffinity": "1",
            "condim": "3",
            "friction": "0.8 0.005 0.0001",
            "solref": "0.030 1.0",
            "solimp": "0.9 0.95 0.001 0.5 2",
        },
    )

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="unicode")
    return surface_z, float(p_world[2])


def contact_measurement(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    probe_geom_id: int,
    surface_geom_id: int,
) -> tuple[float, bool]:
    total = 0.0
    present = False
    wrench = np.zeros(6, dtype=float)

    for i in range(data.ncon):
        contact = data.contact[i]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair != {probe_geom_id, surface_geom_id}:
            continue
        present = True
        mujoco.mj_contactForce(model, data, i, wrench)
        total += max(0.0, float(wrench[0]))

    return total, present


def build_impedance_controllers() -> tuple[CartesianImpedance6D, CartesianImpedance6D]:
    approach = CartesianImpedance6D.from_gains(
        translational_stiffness=[160.0, 160.0, 140.0],
        translational_damping=[30.0, 30.0, 40.0],
        rotational_stiffness=[15.0, 15.0, 15.0],
        rotational_damping=[3.0, 3.0, 3.0],
        force_limits=[30.0, 30.0, 25.0],
        moment_limits=[6.0, 6.0, 6.0],
    )
    lateral_hold = CartesianImpedance6D.from_gains(
        translational_stiffness=[180.0, 180.0, 0.0],
        translational_damping=[30.0, 30.0, 0.0],
        rotational_stiffness=[15.0, 15.0, 15.0],
        rotational_damping=[3.0, 3.0, 3.0],
        force_limits=[35.0, 35.0, 35.0],
        moment_limits=[6.0, 6.0, 6.0],
    )
    return approach, lateral_hold


def build_force_controller(mode: str) -> ForcePIController:
    if mode == "p":
        return ForcePIController(kp=0.8, ki=0.0, command_max=FORCE_COMMAND_LIMIT)
    if mode == "pi":
        return ForcePIController(
            kp=0.8,
            ki=0.8,
            command_max=FORCE_COMMAND_LIMIT,
            integral_limit=14.0,
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
    contact_log = np.zeros(steps, dtype=bool)
    command_log = np.zeros(steps)
    error_log = np.zeros(steps)
    z_log = np.zeros(steps)
    clearance_log = np.zeros(steps)
    tau_log = np.zeros((steps, 7))

    in_force_control = False
    contact_time = float("nan")
    filtered_force = 0.0
    force_command_state = 0.0

    for k in range(steps):
        t = float(adapter.data.time)
        p, r = adapter.get_tcp_pose(frame="base")
        j = adapter.get_jacobian(frame="base")
        twist = j @ adapter.get_qdot()

        measured_force, contact_present = contact_measurement(
            adapter.model, adapter.data, probe_id, surface_id
        )

        if not in_force_control and measured_force >= CONTACT_THRESHOLD:
            in_force_control = True
            contact_time = t
            force_controller.reset()
            filtered_force = measured_force
            force_command_state = float(
                np.clip(max(INITIAL_FORCE_COMMAND, measured_force), 0.0, F_DES)
            )

        if not in_force_control:
            filtered_force = low_pass(filtered_force, measured_force, dt)
            wrench_cmd = approach_controller.compute(
                position=p,
                rotation=r,
                linear_velocity=twist[:3],
                angular_velocity=twist[3:],
                desired_position=approach_target,
                desired_rotation=r_home,
            )
            force_command = 0.0
            force_error = F_DES - filtered_force
        else:
            filtered_force = low_pass(filtered_force, measured_force, dt)

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
        contact_log[k] = contact_present
        command_log[k] = force_command
        error_log[k] = force_error
        z_log[k] = p[2]
        clearance_log[k] = p[2] - (surface_z + PROBE_RADIUS)
        tau_log[k] = tau

        if k < steps - 1:
            adapter.set_joint_torque(tau)
            adapter.step()

        if not np.isfinite(adapter.data.qpos).all() or not np.isfinite(adapter.data.qvel).all():
            raise RuntimeError(f"NaN/Inf in Task06 stable mode={mode} at t={t:.4f}s")

    if not np.isfinite(contact_time):
        raise RuntimeError(f"Mode {mode} never established contact")

    return RunResult(
        mode=mode,
        time=time_log,
        normal_force_raw=force_raw_log,
        normal_force_filtered=force_filt_log,
        contact_present=contact_log,
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

    stable_contact_mask = result.time >= result.contact_time + 0.2
    lost = np.logical_or(
        ~result.contact_present[stable_contact_mask],
        result.normal_force_raw[stable_contact_mask] <= CONTACT_LOSS_FORCE,
    )
    contact_loss_ratio = 100.0 * float(np.mean(lost)) if np.any(stable_contact_mask) else float("nan")

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
            "contact_present",
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
                int(result.contact_present[k]),
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
            measured_force, _ = contact_measurement(
                adapter.model, adapter.data, probe_id, surface_id
            )

            if not in_force_control and measured_force >= CONTACT_THRESHOLD:
                in_force_control = True
                force_controller.reset()
                filtered_force = measured_force
                force_command_state = float(
                    np.clip(max(INITIAL_FORCE_COMMAND, measured_force), 0.0, F_DES)
                )

            force_command = 0.0
            if not in_force_control:
                filtered_force = low_pass(filtered_force, measured_force, dt)
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
    print("Desired normal force   : 10.00 N")
    print("Feedback signal        : low-pass filtered MuJoCo contact normal force")
    print(f"Force filter tau       : {FORCE_FILTER_TAU:.3f} s")
    print(f"Command slew rate      : {COMMAND_SLEW_RATE:.1f} N/s")
    print("P gains                : Kp=0.8, Ki=0")
    print("PI gains               : Kp=0.8, Ki=0.8")
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
