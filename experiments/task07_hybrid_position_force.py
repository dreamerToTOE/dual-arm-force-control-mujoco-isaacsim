"""Task07: hybrid Cartesian position/force control on a horizontal plane.

After the same gentle approach used in Task06, the controller explicitly splits
6D task space with selection matrices:

    W_cmd = S_p W_pos + S_f W_force

where
    S_p = diag(1, 1, 0, 1, 1, 1)
    S_f = diag(0, 0, 1, 0, 0, 0)

Thus X/Y translation and orientation are impedance-controlled while normal Z
is force-controlled.  After normal force settles, the TCP slides smoothly in
+BASE-X while maintaining the desired contact force.
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

from controllers.cartesian_impedance import CartesianImpedance6D, rotation_error_base
from controllers.hybrid_position_force import HybridSelection6D
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
from experiments.task06_contact_force_control_stable import (
    FORCE_FILTER_TAU,
    COMMAND_SLEW_RATE,
    build_force_controller,
    low_pass,
    slew_limit,
)
from sim.mujoco_adapter import MuJoCoAdapter


POSITION_MASK = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])

APPROACH_EXTRA = 0.012
DURATION = 21.0
FORCE_SETTLE_WAIT = 7.0
SLIDE_DURATION = 10.0
SLIDE_DISTANCE = 0.040
POST_SLIDE_HOLD = 1.0

SURFACE_TIMECONST = 0.050
SURFACE_DAMPING_RATIO = 1.0
SURFACE_FRICTION = "0 0 0"

INITIAL_FORCE_COMMAND = 4.0
FORCE_COMMAND_LIMIT = 16.0
CONTACT_LOSS_FORCE = 0.05


@dataclass
class RunResult:
    time: np.ndarray
    normal_force_raw: np.ndarray
    normal_force_filtered: np.ndarray
    force_command: np.ndarray
    desired_position: np.ndarray
    tcp_position: np.ndarray
    position_error: np.ndarray
    orientation_error: np.ndarray
    clearance: np.ndarray
    tau_total: np.ndarray
    contact_time: float
    slide_start_time: float
    slide_end_time: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to Menagerie FR3 scene.xml or fr3.xml")
    parser.add_argument("--viewer", action="store_true", help="Replay the hybrid slide in MuJoCo viewer")
    return parser.parse_args()


def build_task07_scene(fr3_xml: Path, output_xml: Path) -> tuple[float, float]:
    surface_z, home_z = build_contact_scene(fr3_xml, output_xml)

    tree = ET.parse(output_xml)
    root = tree.getroot()
    found = set()
    for geom in root.iter("geom"):
        if geom.get("name") in {"task06_probe", "task06_surface"}:
            geom.set("solref", f"{SURFACE_TIMECONST:.6f} {SURFACE_DAMPING_RATIO:.6f}")
            geom.set("solimp", "0.9 0.95 0.001 0.5 2")
            geom.set("friction", SURFACE_FRICTION)
            # Stage-1 hybrid-control teaching scene: normal constraint only.
            # Removing tangential friction isolates the selection-matrix concept;
            # friction/stick-slip robustness is intentionally deferred.
            geom.set("condim", "1")
            found.add(geom.get("name"))

    missing = {"task06_probe", "task06_surface"} - found
    if missing:
        raise RuntimeError(f"Task07 contact geoms missing while editing scene: {sorted(missing)}")

    tree.write(output_xml, encoding="unicode")
    return surface_z, home_z


def build_position_controller() -> CartesianImpedance6D:
    # A full 6D impedance wrench is computed first.  The selection matrix will
    # deliberately remove its Z force so that Z is not position-controlled.
    return CartesianImpedance6D.from_gains(
        translational_stiffness=[1800.0, 1800.0, 250.0],
        translational_damping=[100.0, 100.0, 35.0],
        rotational_stiffness=[20.0, 20.0, 20.0],
        rotational_damping=[4.0, 4.0, 4.0],
        force_limits=[35.0, 35.0, 30.0],
        moment_limits=[6.0, 6.0, 6.0],
    )


def smooth_step(r: float) -> tuple[float, float]:
    """Quintic progress s(r) and ds/dr for r in [0, 1]."""
    r = float(np.clip(r, 0.0, 1.0))
    s = 10.0 * r**3 - 15.0 * r**4 + 6.0 * r**5
    ds_dr = 30.0 * r**2 - 60.0 * r**3 + 30.0 * r**4
    return s, ds_dr


def desired_tangential_state(
    t: float,
    contact_time: float,
    origin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    slide_start = contact_time + FORCE_SETTLE_WAIT
    slide_end = slide_start + SLIDE_DURATION

    p_des = origin.copy()
    v_des = np.zeros(3)

    if t <= slide_start:
        return p_des, v_des

    if t >= slide_end:
        p_des[0] += SLIDE_DISTANCE
        return p_des, v_des

    r = (t - slide_start) / SLIDE_DURATION
    s, ds_dr = smooth_step(r)
    p_des[0] += SLIDE_DISTANCE * s
    v_des[0] = SLIDE_DISTANCE * ds_dr / SLIDE_DURATION
    return p_des, v_des


def simulate(contact_xml: Path, surface_z: float) -> RunResult:
    adapter = MuJoCoAdapter.from_xml_path(str(contact_xml))
    disable_builtin_position_actuators(adapter.model)
    set_home(adapter)

    probe_id = mujoco.mj_name2id(adapter.model, mujoco.mjtObj.mjOBJ_GEOM, "task06_probe")
    surface_id = mujoco.mj_name2id(adapter.model, mujoco.mjtObj.mjOBJ_GEOM, "task06_surface")
    if probe_id < 0 or surface_id < 0:
        raise RuntimeError("Task07 contact geoms missing")

    selector = HybridSelection6D.from_position_mask(POSITION_MASK)
    approach_controller, _ = build_impedance_controllers()
    position_controller = build_position_controller()
    force_controller = build_force_controller("pi")

    p_home, r_home = adapter.get_tcp_pose(frame="base")
    approach_target = p_home + np.array([0.0, 0.0, -(INITIAL_GAP + APPROACH_EXTRA)])

    dt = float(adapter.model.opt.timestep)
    steps = int(np.ceil(DURATION / dt)) + 1

    time_log = np.zeros(steps)
    raw_force_log = np.zeros(steps)
    filtered_force_log = np.zeros(steps)
    command_log = np.zeros(steps)
    desired_p_log = np.zeros((steps, 3))
    p_log = np.zeros((steps, 3))
    pos_error_log = np.zeros((steps, 3))
    ori_error_log = np.zeros((steps, 3))
    clearance_log = np.zeros(steps)
    tau_log = np.zeros((steps, 7))

    in_force_control = False
    contact_time = float("nan")
    contact_origin = p_home.copy()
    filtered_force = 0.0
    force_command_state = 0.0

    for k in range(steps):
        t = float(adapter.data.time)
        p, r = adapter.get_tcp_pose(frame="base")
        j = adapter.get_jacobian(frame="base")
        twist = j @ adapter.get_qdot()
        measured_force = contact_normal_force(adapter.model, adapter.data, probe_id, surface_id)
        clearance = float(p[2] - (surface_z + PROBE_RADIUS))

        if not in_force_control and measured_force >= CONTACT_THRESHOLD:
            in_force_control = True
            contact_time = t
            contact_origin = p.copy()
            force_controller.reset()
            filtered_force = measured_force
            force_command_state = float(
                np.clip(max(INITIAL_FORCE_COMMAND, measured_force), 0.5, F_DES)
            )

        if not in_force_control:
            filtered_force = measured_force
            p_des = approach_target
            v_des = np.zeros(3)
            wrench_cmd = approach_controller.compute(
                position=p,
                rotation=r,
                linear_velocity=twist[:3],
                angular_velocity=twist[3:],
                desired_position=p_des,
                desired_rotation=r_home,
            )
            force_command = 0.0
        else:
            filtered_force = low_pass(filtered_force, measured_force, dt)
            p_des, v_des = desired_tangential_state(t, contact_time, contact_origin)

            w_pos = position_controller.compute(
                position=p,
                rotation=r,
                linear_velocity=twist[:3],
                angular_velocity=twist[3:],
                desired_position=p_des,
                desired_rotation=r_home,
                desired_linear_velocity=v_des,
            )

            force_error = F_DES - filtered_force
            desired_force_command = force_controller.compute(force_error, dt)
            force_command_state = slew_limit(
                force_command_state, desired_force_command, dt
            )
            force_command = float(np.clip(force_command_state, 0.0, FORCE_COMMAND_LIMIT))

            # Positive scalar command means "push into the horizontal surface".
            # In BASE coordinates that is -Z.
            w_force = np.array([0.0, 0.0, -force_command, 0.0, 0.0, 0.0])
            wrench_cmd = selector.combine(w_pos, w_force)

        tau = j.T @ wrench_cmd + adapter.get_gravity()
        tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)

        time_log[k] = t
        raw_force_log[k] = measured_force
        filtered_force_log[k] = filtered_force
        command_log[k] = force_command
        desired_p_log[k] = p_des
        p_log[k] = p
        pos_error_log[k] = p_des - p
        ori_error_log[k] = rotation_error_base(r, r_home)
        clearance_log[k] = clearance
        tau_log[k] = tau

        if k < steps - 1:
            adapter.set_joint_torque(tau)
            adapter.step()

        if not np.isfinite(adapter.data.qpos).all() or not np.isfinite(adapter.data.qvel).all():
            raise RuntimeError(f"NaN/Inf in Task07 at t={t:.4f}s")

    if not np.isfinite(contact_time):
        raise RuntimeError("Task07 never established contact")

    return RunResult(
        time=time_log,
        normal_force_raw=raw_force_log,
        normal_force_filtered=filtered_force_log,
        force_command=command_log,
        desired_position=desired_p_log,
        tcp_position=p_log,
        position_error=pos_error_log,
        orientation_error=ori_error_log,
        clearance=clearance_log,
        tau_total=tau_log,
        contact_time=contact_time,
        slide_start_time=contact_time + FORCE_SETTLE_WAIT,
        slide_end_time=contact_time + FORCE_SETTLE_WAIT + SLIDE_DURATION,
    )


def metrics(result: RunResult) -> dict[str, float]:
    slide = np.logical_and(
        result.time >= result.slide_start_time,
        result.time <= result.slide_end_time,
    )
    post_contact = result.time >= result.contact_time + 0.2

    if not np.any(slide):
        raise RuntimeError("Task07 slide window is empty")

    x_err = result.position_error[slide, 0]
    y_err = result.position_error[slide, 1]
    f = result.normal_force_raw[slide]

    ori_norm_deg = np.linalg.norm(result.orientation_error[slide], axis=1) * 180.0 / np.pi
    contact_loss = result.normal_force_raw[post_contact] <= CONTACT_LOSS_FORCE

    return {
        "x_rms_mm": 1000.0 * float(np.sqrt(np.mean(x_err**2))),
        "x_max_mm": 1000.0 * float(np.max(np.abs(x_err))),
        "y_rms_mm": 1000.0 * float(np.sqrt(np.mean(y_err**2))),
        "force_mean": float(np.mean(f)),
        "force_error_mean": float(F_DES - np.mean(f)),
        "force_std": float(np.std(f)),
        "force_ripple": float(np.ptp(f)),
        "contact_loss_ratio": 100.0 * float(np.mean(contact_loss)),
        "max_orientation_error_deg": float(np.max(ori_norm_deg)),
        "peak_tau": float(np.max(np.abs(result.tau_total))),
        "slide_distance_actual_mm": 1000.0
        * float(result.tcp_position[slide][-1, 0] - result.tcp_position[slide][0, 0]),
    }


def save_csv(result: RunResult, output_dir: Path) -> Path:
    path = output_dir / "hybrid_slide.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        headers = [
            "time",
            "normal_force_raw",
            "normal_force_filtered",
            "force_command",
            "x_des",
            "y_des",
            "z_des",
            "x",
            "y",
            "z",
            "ex",
            "ey",
            "ez",
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
                *result.desired_position[k],
                *result.tcp_position[k],
                *result.position_error[k],
                result.clearance[k],
                *result.tau_total[k],
            ])
    return path


def save_plot(result: RunResult, output_dir: Path) -> Path:
    path = output_dir / "hybrid_position_force.png"
    t_rel = result.time - result.contact_time
    slide_start = result.slide_start_time - result.contact_time
    slide_end = result.slide_end_time - result.contact_time

    mask = result.time >= result.contact_time
    t = t_rel[mask]

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 11))

    axes[0].plot(
        t,
        1000.0 * (result.desired_position[mask, 0] - result.desired_position[mask][0, 0]),
        linestyle="--",
        label="x_des",
    )
    axes[0].plot(
        t,
        1000.0 * (result.tcp_position[mask, 0] - result.tcp_position[mask][0, 0]),
        label="x",
    )
    axes[0].set_ylabel("tangential x [mm]")
    axes[0].set_title("Task07: hybrid position/force control while sliding on a plane")

    axes[1].plot(t, result.normal_force_raw[mask], alpha=0.45, label="raw F_meas")
    axes[1].plot(t, result.normal_force_filtered[mask], label="filtered F_meas")
    axes[1].axhline(F_DES, linestyle="--", label="F_des")
    axes[1].set_ylabel("normal force [N]")

    axes[2].plot(t, result.force_command[mask], label="push command")
    axes[2].set_ylabel("force command [N]")

    axes[3].plot(t, 1000.0 * result.clearance[mask], label="clearance")
    axes[3].set_ylabel("clearance [mm]")
    axes[3].set_xlabel("time since contact [s]")

    for ax in axes:
        ax.axvspan(slide_start, slide_end, alpha=0.08, label="_slide_window")
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


def _add_sphere(viewer, position: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    geom = _next_geom(viewer)
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, radius),
        np.asarray(position, dtype=float),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )


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
    length = float(np.clip(0.018 * norm, 0.06, 0.24))
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.014,
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

    selector = HybridSelection6D.from_position_mask(POSITION_MASK)
    approach_controller, _ = build_impedance_controllers()
    position_controller = build_position_controller()
    force_controller = build_force_controller("pi")

    p_home, r_home = adapter.get_tcp_pose(frame="base")
    approach_target = p_home + np.array([0.0, 0.0, -(INITIAL_GAP + APPROACH_EXTRA)])

    in_force_control = False
    contact_time = float("nan")
    contact_origin = p_home.copy()
    filtered_force = 0.0
    force_command_state = 0.0
    dt = float(adapter.model.opt.timestep)

    print("\n=== Task07 viewer ===")
    print("Green sphere : desired tangential TCP target")
    print("Yellow sphere: physical TCP contact probe")
    print("Red arrow    : commanded normal push force (-BASE-Z)")
    print("Cyan arrow   : measured contact normal force (+BASE-Z)")
    print("After force settles, the desired target slides +BASE-X.")

    with mujoco.viewer.launch_passive(adapter.model, adapter.data) as viewer:
        viewer.cam.lookat[:] = p_home
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0

        while viewer.is_running() and adapter.data.time < DURATION:
            t = float(adapter.data.time)
            p, r = adapter.get_tcp_pose(frame="base")
            j = adapter.get_jacobian(frame="base")
            twist = j @ adapter.get_qdot()
            measured_force = contact_normal_force(adapter.model, adapter.data, probe_id, surface_id)

            if not in_force_control and measured_force >= CONTACT_THRESHOLD:
                in_force_control = True
                contact_time = t
                contact_origin = p.copy()
                force_controller.reset()
                filtered_force = measured_force
                force_command_state = float(
                    np.clip(max(INITIAL_FORCE_COMMAND, measured_force), 0.5, F_DES)
                )

            force_command = 0.0
            if not in_force_control:
                filtered_force = measured_force
                p_des = approach_target
                wrench_cmd = approach_controller.compute(
                    p, r, twist[:3], twist[3:], p_des, r_home
                )
            else:
                filtered_force = low_pass(filtered_force, measured_force, dt)
                p_des, v_des = desired_tangential_state(t, contact_time, contact_origin)
                w_pos = position_controller.compute(
                    p,
                    r,
                    twist[:3],
                    twist[3:],
                    p_des,
                    r_home,
                    desired_linear_velocity=v_des,
                )
                desired_force_command = force_controller.compute(
                    F_DES - filtered_force, dt
                )
                force_command_state = slew_limit(
                    force_command_state, desired_force_command, dt
                )
                force_command = float(
                    np.clip(force_command_state, 0.0, FORCE_COMMAND_LIMIT)
                )
                w_force = np.array(
                    [0.0, 0.0, -force_command, 0.0, 0.0, 0.0]
                )
                wrench_cmd = selector.combine(w_pos, w_force)

            tau = j.T @ wrench_cmd + adapter.get_gravity()
            tau = np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)
            adapter.set_joint_torque(tau)
            adapter.step()

            viewer.user_scn.ngeom = 0
            _add_sphere(
                viewer,
                p_des + np.array([0.0, 0.0, 0.010]),
                0.010,
                np.array([0.1, 1.0, 0.2, 1.0]),
            )
            _add_arrow(
                viewer,
                p + np.array([0.055, 0.0, 0.0]),
                np.array([0.0, 0.0, -force_command]),
                np.array([1.0, 0.15, 0.1, 1.0]),
            )
            _add_arrow(
                viewer,
                p + np.array([-0.055, 0.0, 0.0]),
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

    output_dir = ROOT / "outputs" / "task07"
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_xml = output_dir / "task07_hybrid_scene.xml"
    surface_z, home_z = build_task07_scene(fr3_xml, contact_xml)

    selector = HybridSelection6D.from_position_mask(POSITION_MASK)
    validation = selector.validate()

    result = simulate(contact_xml, surface_z)
    m = metrics(result)

    print("\n=== Task07 hybrid position/force control ===")
    print("Task frame             : BASE")
    print("Wrench order           : [Fx, Fy, Fz, Mx, My, Mz]")
    print("Position mask          :", selector.position_mask)
    print("Force mask             :", selector.force_mask)
    print("S_p =")
    print(selector.S_position)
    print("S_f =")
    print(selector.S_force)
    print(
        "Selection validation  : "
        f"complement={validation['complement_error']:.1e}, "
        f"overlap={validation['overlap_error']:.1e}"
    )
    print(f"Desired normal force   : {F_DES:.2f} N")
    print(f"Tangential slide       : +BASE-X, {SLIDE_DISTANCE*1000:.1f} mm")
    print(f"Slide duration         : {SLIDE_DURATION:.2f} s")
    print(f"Slide starts after     : {FORCE_SETTLE_WAIT:.2f} s from first contact")
    print(f"Surface friction       : {SURFACE_FRICTION.split()[0]}")
    print("Tangential gains       : Kx=Ky=1800 N/m, Dx=Dy=100 N*s/m")
    print(f"HOME TCP z             : {home_z:.5f} m")

    print("\n=== Task07 metrics during tangential slide ===")
    print(f"X RMS tracking error   : {m['x_rms_mm']:.3f} mm")
    print(f"X max tracking error   : {m['x_max_mm']:.3f} mm")
    print(f"Y RMS tracking error   : {m['y_rms_mm']:.3f} mm")
    print(f"Actual X travel        : {m['slide_distance_actual_mm']:.3f} mm")
    print(f"Mean normal force      : {m['force_mean']:.4f} N")
    print(f"Mean force error       : {m['force_error_mean']:.4f} N")
    print(f"Force STD              : {m['force_std']:.4f} N")
    print(f"Force ripple           : {m['force_ripple']:.4f} N")
    print(f"Contact loss           : {m['contact_loss_ratio']:.3f} %")
    print(f"Max orientation error  : {m['max_orientation_error_deg']:.3f} deg")
    print(f"Peak |tau|             : {m['peak_tau']:.4f} N m")

    plot_path = save_plot(result, output_dir)
    csv_path = save_csv(result, output_dir)

    print(f"\nPlot : {plot_path}")
    print(f"CSV  : {csv_path}")
    print(f"Scene: {contact_xml}")
    control_pass = (
        m["x_rms_mm"] < 5.0
        and m["x_max_mm"] < 10.0
        and abs(m["force_error_mean"]) < 0.5
        and m["force_std"] < 0.5
        and m["force_ripple"] < 2.0
        and m["contact_loss_ratio"] < 0.5
    )
    print("\nNumerical execution: PASS (no NaN/Inf).")
    print("Control-quality acceptance:", "PASS" if control_pass else "NOT YET PASS")
    print(
        "Targets: X RMS < 5 mm, X max < 10 mm, |mean force error| < 0.5 N, "
        "force STD < 0.5 N, ripple < 2 N, contact loss < 0.5%."
    )

    if args.viewer:
        show_viewer(contact_xml, surface_z)


if __name__ == "__main__":
    main()
