"""Task06 stage 1: real MuJoCo contact and normal-force control.

The script builds a temporary FR3 contact scene with a small spherical probe at
TCP and a horizontal plane just below HOME. It compares:

  1) P force control
  2) PI force control

After a gentle Cartesian-impedance approach, the controller regulates the
measured contact normal force. The normal-force command is mapped through the
Jacobian transpose and gravity compensation is added.

This is the first true contact-force closed loop in the project.
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
APPROACH_EXTRA = 0.012
CONTACT_THRESHOLD = 0.5
F_DES = 10.0
FORCE_COMMAND_LIMIT = 18.0
DURATION = 5.0

MODES = ("p", "pi")


@dataclass
class RunResult:
    mode: str
    time: np.ndarray
    normal_force: np.ndarray
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
    parser.add_argument("--viewer", action="store_true", help="Replay PI case with MuJoCo viewer")
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
    """Create a standalone Task06 model with one TCP probe and raised plane."""
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

    # Disable original link collision meshes for this teaching experiment, so
    # the raised plane contacts only the deliberately added spherical TCP probe.
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
    site_index = list(parent).index(site)
    parent.insert(site_index, probe)

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
            # Positive solref = time constant, damping ratio. Smaller time
            # constant gives a firmer contact; stage 2 will vary this value.
            "solref": "0.015 1.0",
            "solimp": "0.9 0.95 0.001 0.5 2",
        },
    )

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="unicode")
    return surface_z, float(p_world[2])


def contact_normal_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    probe_geom_id: int,
    surface_geom_id: int,
) -> float:
    total = 0.0
    wrench = np.zeros(6, dtype=float)
    for i in range(data.ncon):
        contact = data.contact[i]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair != {probe_geom_id, surface_geom_id}:
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        total += max(0.0, float(wrench[0]))
    return total


def build_impedance_controllers() -> tuple[CartesianImpedance6D, CartesianImpedance6D]:
    approach = CartesianImpedance6D.from_gains(
        translational_stiffness=[180.0, 180.0, 220.0],
        translational_damping=[28.0, 28.0, 32.0],
        rotational_stiffness=[15.0, 15.0, 15.0],
        rotational_damping=[3.0, 3.0, 3.0],
        force_limits=[35.0, 35.0, 35.0],
        moment_limits=[6.0, 6.0, 6.0],
    )
    lateral_hold = CartesianImpedance6D.from_gains(
        translational_stiffness=[180.0, 180.0, 0.0],
        translational_damping=[28.0, 28.0, 0.0],
        rotational_stiffness=[15.0, 15.0, 15.0],
        rotational_damping=[3.0, 3.0, 3.0],
        force_limits=[35.0, 35.0, 35.0],
        moment_limits=[6.0, 6.0, 6.0],
    )
    return approach, lateral_hold


def build_force_controller(mode: str) -> ForcePIController:
    if mode == "p":
        return ForcePIController(kp=1.5, ki=0.0, command_max=FORCE_COMMAND_LIMIT)
    if mode == "pi":
        return ForcePIController(kp=1.5, ki=1.8, command_max=FORCE_COMMAND_LIMIT, integral_limit=8.0)
    raise ValueError(mode)


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
    force_log = np.zeros(steps)
    command_log = np.zeros(steps)
    error_log = np.zeros(steps)
    z_log = np.zeros(steps)
    clearance_log = np.zeros(steps)
    tau_log = np.zeros((steps, 7))

    in_force_control = False
    contact_time = float("nan")

    for k in range(steps):
        t = float(adapter.data.time)
        p, r = adapter.get_tcp_pose(frame="base")
        j = adapter.get_jacobian(frame="base")
        qdot = adapter.get_qdot()
        twist = j @ qdot
        measured_force = contact_normal_force(adapter.model, adapter.data, probe_id, surface_id)

        if not in_force_control and measured_force >= CONTACT_THRESHOLD:
            in_force_control = True
            contact_time = t
            force_controller.reset()

        if not in_force_control:
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
            hold_target = np.array([p_home[0], p_home[1], p[2]])
            wrench_cmd = hold_controller.compute(
                position=p,
                rotation=r,
                linear_velocity=twist[:3],
                angular_velocity=twist[3:],
                desired_position=hold_target,
                desired_rotation=r_home,
            )
            force_error = F_DES - measured_force
            force_command = force_controller.compute(force_error, dt)
            # Surface is below the probe, so pushing into it is -BASE-Z.
            wrench_cmd[2] += -force_command

        tau = j.T @ wrench_cmd + adapter.get_gravity()
        tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)

        time_log[k] = t
        force_log[k] = measured_force
        command_log[k] = force_command
        error_log[k] = force_error
        z_log[k] = p[2]
        clearance_log[k] = p[2] - (surface_z + PROBE_RADIUS)
        tau_log[k] = tau

        if k < steps - 1:
            adapter.set_joint_torque(tau)
            adapter.step()

        if not np.isfinite(adapter.data.qpos).all() or not np.isfinite(adapter.data.qvel).all():
            raise RuntimeError(f"NaN/Inf in Task06 mode={mode} at t={t:.4f}s")

    if not np.isfinite(contact_time):
        raise RuntimeError(f"Mode {mode} never established contact")

    return RunResult(
        mode=mode,
        time=time_log,
        normal_force=force_log,
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


def force_metrics(result: RunResult) -> dict[str, float]:
    mask = post_contact_mask(result)
    t = result.time[mask] - result.contact_time
    f = result.normal_force[mask]
    if len(t) == 0:
        raise RuntimeError("No post-contact samples")

    reached = np.flatnonzero(f >= 0.9 * F_DES)
    rise = float(t[reached[0]]) if len(reached) else float("nan")
    overshoot = max(0.0, float(np.max(f) - F_DES)) / F_DES * 100.0

    tail_start = max(result.contact_time, DURATION - 0.6)
    tail = result.normal_force[result.time >= tail_start]
    steady_error = float(F_DES - np.mean(tail))
    return {
        "rise": rise,
        "overshoot": overshoot,
        "steady_error": steady_error,
        "peak_force": float(np.max(f)),
        "peak_tau": float(np.max(np.abs(result.tau_total))),
    }


def save_csv(result: RunResult, output_dir: Path) -> Path:
    path = output_dir / f"{result.mode}.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        headers = ["time", "normal_force", "force_command", "force_error", "tcp_z", "clearance"]
        headers += [f"tau{i}" for i in range(1, 8)]
        writer.writerow(headers)
        for k in range(len(result.time)):
            writer.writerow([
                result.time[k],
                result.normal_force[k],
                result.force_command[k],
                result.force_error[k],
                result.tcp_z[k],
                result.clearance[k],
                *result.tau_total[k],
            ])
    return path


def save_plot(results: dict[str, RunResult], output_dir: Path) -> Path:
    path = output_dir / "p_vs_pi_force_control.png"
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))

    max_rel = 0.0
    for mode in MODES:
        result = results[mode]
        mask = post_contact_mask(result)
        t = result.time[mask] - result.contact_time
        max_rel = max(max_rel, float(t[-1]))
        axes[0].plot(t, result.normal_force[mask], label=mode.upper())
        axes[1].plot(t, result.force_command[mask], label=mode.upper())
        axes[2].plot(t, 1000.0 * result.clearance[mask], label=mode.upper())

    axes[0].axhline(F_DES, linestyle="--", label="F_des")
    axes[0].set_ylabel("normal force [N]")
    axes[0].set_title("Task06: P vs PI normal-force control after first contact")
    axes[1].set_ylabel("push command [N]")
    axes[2].set_ylabel("probe clearance [mm]")
    axes[2].set_xlabel("time since contact [s]")
    axes[0].set_xlim(0.0, max_rel)
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


def show_viewer(contact_xml: Path, surface_z: float) -> None:
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

    print("\n=== Task06 viewer ===")
    print("Yellow sphere : physical TCP contact probe")
    print("Red arrow     : commanded push force (-BASE-Z)")
    print("Cyan arrow    : measured contact normal force (+BASE-Z)")
    print("Viewer replays the PI controller live. Close the window to stop.")

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

            force_command = 0.0
            if not in_force_control:
                wrench_cmd = approach_controller.compute(
                    p, r, twist[:3], twist[3:], approach_target, r_home
                )
            else:
                hold_target = np.array([p_home[0], p_home[1], p[2]])
                wrench_cmd = hold_controller.compute(
                    p, r, twist[:3], twist[3:], hold_target, r_home
                )
                force_command = force_controller.compute(F_DES - measured_force, adapter.model.opt.timestep)
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
            time.sleep(max(0.0, float(adapter.model.opt.timestep)))


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    fr3_xml = resolve_fr3_xml(model_path)

    output_dir = ROOT / "outputs" / "task06"
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_xml = output_dir / "task06_contact_scene.xml"
    surface_z, home_tcp_z = build_contact_scene(fr3_xml, contact_xml)

    results = {mode: simulate_case(contact_xml, mode, surface_z) for mode in MODES}
    metrics = {mode: force_metrics(result) for mode, result in results.items()}

    print("\n=== Task06 contact-force control ===")
    print("Contact geometry       : spherical TCP probe against horizontal plane")
    print("Force measurement      : MuJoCo mj_contactForce normal component")
    print(f"Desired normal force   : {F_DES:.2f} N")
    print("Push direction         : -BASE-Z")
    print(f"HOME TCP z             : {home_tcp_z:.5f} m")
    print(f"Surface z              : {surface_z:.5f} m")
    print(f"Initial geometric gap  : {INITIAL_GAP*1000:.1f} mm")
    print("\nmode   contact[s]   rise90[s]   overshoot[%]   steady error[N]   peak force[N]   peak |tau|[Nm]")
    print("------------------------------------------------------------------------------------------------")
    for mode in MODES:
        m = metrics[mode]
        r = results[mode]
        print(
            f"{mode.upper():<5}  {r.contact_time:>9.4f}   {m['rise']:>9.4f}   "
            f"{m['overshoot']:>12.3f}   {m['steady_error']:>15.4f}   "
            f"{m['peak_force']:>13.4f}   {m['peak_tau']:>14.4f}"
        )

    plot_path = save_plot(results, output_dir)
    csv_paths = [save_csv(results[mode], output_dir) for mode in MODES]

    print(f"\nPlot: {plot_path}")
    for path in csv_paths:
        print(f"CSV : {path}")
    print(f"Scene: {contact_xml}")
    print("\nPASS: contact was established and both force-control loops completed without NaN/Inf.")
    print("Acceptance target: PI should reduce steady normal-force error relative to P without unstable contact oscillation.")

    if args.viewer:
        show_viewer(contact_xml, surface_z)


if __name__ == "__main__":
    main()
