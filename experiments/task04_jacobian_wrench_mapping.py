"""Task04: put Jacobian-transpose wrench mapping into a control loop.

This experiment demonstrates four things:
  1) the full 6x7 TCP Jacobian in the BASE frame;
  2) tau_task = J(q).T @ W_des for the same wrench at different poses;
  3) a slowly varying base-frame wrench mapped every simulation step;
  4) a translational near-singularity diagnostic using sigma_min(Jv).

The dynamic loop is NOT contact-force control yet. A joint PD+gravity posture
controller keeps the robot near HOME while the Cartesian task torque is added:

    tau_total = tau_posture + J(q).T @ W_des

Task06 will introduce actual contact-force regulation.
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
from controllers.wrench_mapping import JacobianTransposeWrenchMapper
from sim.mujoco_adapter import FR3_JOINT_NAMES, MuJoCoAdapter


HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
POSE_A = HOME_Q + np.array([0.35, -0.25, 0.30, 0.20, -0.20, 0.25, 0.15])
POSE_B = HOME_Q + np.array([-0.35, 0.25, -0.30, -0.20, 0.20, -0.25, -0.15])
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
POSTURE_KP = np.array([70.0, 70.0, 65.0, 65.0, 35.0, 30.0, 24.0])
POSTURE_KD = np.array([14.0, 14.0, 12.0, 12.0, 7.0, 6.0, 5.0])


@dataclass
class StaticResult:
    name: str
    q: np.ndarray
    jacobian: np.ndarray
    tau_task: np.ndarray
    sigma_min_jv: float


@dataclass
class LoopResult:
    time: np.ndarray
    q: np.ndarray
    wrench: np.ndarray
    tau_task: np.ndarray
    tau_posture: np.ndarray
    tau_total: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to Menagerie FR3 scene.xml or fr3.xml")
    parser.add_argument("--force-z", type=float, default=10.0, help="Static +BASE-Z force [N]")
    parser.add_argument("--dynamic-force", type=float, default=6.0, help="Dynamic force magnitude [N]")
    parser.add_argument("--frequency", type=float, default=0.15, help="Dynamic force rotation frequency [Hz]")
    parser.add_argument("--duration", type=float, default=8.0, help="Dynamic-loop duration [s]")
    parser.add_argument(
        "--singularity-samples",
        type=int,
        default=1500,
        help="Random joint configurations used for the low-sigma(Jv) diagnostic",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for singularity search")
    parser.add_argument("--viewer", action="store_true", help="Replay the dynamic wrench mapping in MuJoCo")
    return parser.parse_args()


def disable_builtin_position_actuators(model: mujoco.MjModel) -> None:
    if model.nu == 0:
        return
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_biasprm[:, :] = 0.0


def set_q(adapter: MuJoCoAdapter, q: np.ndarray) -> None:
    q = np.asarray(q, dtype=float)
    if q.shape != (7,):
        raise ValueError(f"Expected q shape (7,), got {q.shape}")
    for joint, value in zip(adapter.joint_map, q):
        adapter.data.qpos[joint.qpos_adr] = float(value)
        adapter.data.qvel[joint.dof_adr] = 0.0
    adapter.clear_joint_torque()
    adapter.forward()


def validate_pose(adapter: MuJoCoAdapter, q: np.ndarray, name: str) -> None:
    for i, (joint, value) in enumerate(zip(adapter.joint_map, q)):
        lower, upper = adapter.model.jnt_range[joint.joint_id]
        if not (lower <= value <= upper):
            raise ValueError(
                f"{name}: {FR3_JOINT_NAMES[i]}={value:.4f} is outside "
                f"[{lower:.4f}, {upper:.4f}]"
            )


def static_mapping(
    adapter: MuJoCoAdapter,
    name: str,
    q: np.ndarray,
    wrench_base: np.ndarray,
    mapper: JacobianTransposeWrenchMapper,
) -> StaticResult:
    validate_pose(adapter, q, name)
    set_q(adapter, q)
    jacobian = adapter.get_jacobian(frame="base")
    tau_task = mapper.compute(jacobian, wrench_base)
    sigma_min_jv = float(np.linalg.svd(jacobian[:3, :], compute_uv=False)[-1])
    return StaticResult(name, q.copy(), jacobian, tau_task, sigma_min_jv)


def search_low_sigma_pose(
    adapter: MuJoCoAdapter,
    samples: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Find a reproducible low-sigma translational Jacobian configuration.

    This is a kinematic diagnostic only. The returned random configuration is
    not claimed to be collision-free and is never used as a motion target.
    """
    if samples < 1:
        raise ValueError("singularity-samples must be >= 1")

    rng = np.random.default_rng(seed)
    lower = np.array(
        [adapter.model.jnt_range[j.joint_id, 0] for j in adapter.joint_map], dtype=float
    )
    upper = np.array(
        [adapter.model.jnt_range[j.joint_id, 1] for j in adapter.joint_map], dtype=float
    )

    # Stay away from hard limits while still exploring a broad workspace.
    center = 0.5 * (lower + upper)
    half_width = 0.45 * (upper - lower)

    best_q = HOME_Q.copy()
    set_q(adapter, best_q)
    best_sigma = float(np.linalg.svd(adapter.get_jacobian("base")[:3, :], compute_uv=False)[-1])

    for _ in range(samples):
        q = center + rng.uniform(-1.0, 1.0, size=7) * half_width
        set_q(adapter, q)
        sigma_min = float(
            np.linalg.svd(adapter.get_jacobian("base")[:3, :], compute_uv=False)[-1]
        )
        if sigma_min < best_sigma:
            best_sigma = sigma_min
            best_q = q.copy()

    set_q(adapter, HOME_Q)
    return best_q, best_sigma


def dynamic_wrench(t: float, magnitude: float, frequency: float) -> np.ndarray:
    """A slowly rotating pure force in the BASE X-Z plane."""
    phase = 2.0 * np.pi * frequency * t
    return np.array(
        [magnitude * np.sin(phase), 0.0, magnitude * np.cos(phase), 0.0, 0.0, 0.0],
        dtype=float,
    )


def make_posture_controller() -> JointPDController:
    # Do not clip the posture term separately. Only the final summed torque is
    # clipped to the FR3 joint limits.
    return JointPDController.from_gains(
        kp=POSTURE_KP,
        kd=POSTURE_KD,
        dof=7,
        torque_limits=None,
    )


def simulate_loop(
    model_path: Path,
    duration: float,
    magnitude: float,
    frequency: float,
) -> LoopResult:
    adapter = MuJoCoAdapter.from_xml_path(str(model_path))
    disable_builtin_position_actuators(adapter.model)
    set_q(adapter, HOME_Q)

    mapper = JacobianTransposeWrenchMapper(dof=7)
    posture = make_posture_controller()
    dt = float(adapter.model.opt.timestep)
    steps = int(np.ceil(duration / dt)) + 1

    time_log = np.zeros(steps)
    q_log = np.zeros((steps, 7))
    wrench_log = np.zeros((steps, 6))
    tau_task_log = np.zeros((steps, 7))
    tau_posture_log = np.zeros((steps, 7))
    tau_total_log = np.zeros((steps, 7))

    for k in range(steps):
        t = float(adapter.data.time)
        q = adapter.get_q()
        qdot = adapter.get_qdot()
        wrench = dynamic_wrench(t, magnitude, frequency)
        jacobian = adapter.get_jacobian(frame="base")
        tau_task = mapper.compute(jacobian, wrench)
        tau_posture = posture.compute(
            q=q,
            qdot=qdot,
            q_des=HOME_Q,
            qdot_des=np.zeros(7),
            tau_ff=adapter.get_gravity(),
        )
        tau_total = np.clip(tau_posture + tau_task, -TORQUE_LIMITS, TORQUE_LIMITS)

        time_log[k] = t
        q_log[k] = q
        wrench_log[k] = wrench
        tau_task_log[k] = tau_task
        tau_posture_log[k] = tau_posture
        tau_total_log[k] = tau_total

        if k < steps - 1:
            adapter.set_joint_torque(tau_total)
            adapter.step()

        if not np.isfinite(adapter.data.qpos).all() or not np.isfinite(adapter.data.qvel).all():
            raise RuntimeError(f"NaN/Inf detected at t={t:.4f}s")

    return LoopResult(
        time=time_log,
        q=q_log,
        wrench=wrench_log,
        tau_task=tau_task_log,
        tau_posture=tau_posture_log,
        tau_total=tau_total_log,
    )


def save_loop_csv(result: LoopResult, output_dir: Path) -> Path:
    path = output_dir / "task04_wrench_loop.csv"
    headers = ["time"]
    headers += ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    headers += [f"q{i}" for i in range(1, 8)]
    headers += [f"tau_task{i}" for i in range(1, 8)]
    headers += [f"tau_posture{i}" for i in range(1, 8)]
    headers += [f"tau_total{i}" for i in range(1, 8)]

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for k in range(len(result.time)):
            writer.writerow(
                [
                    result.time[k],
                    *result.wrench[k],
                    *result.q[k],
                    *result.tau_task[k],
                    *result.tau_posture[k],
                    *result.tau_total[k],
                ]
            )
    return path


def save_loop_plot(result: LoopResult, output_dir: Path) -> Path:
    path = output_dir / "wrench_loop.png"
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 9))

    axes[0].plot(result.time, result.wrench[:, 0], label="Fx")
    axes[0].plot(result.time, result.wrench[:, 2], label="Fz")
    axes[0].set_ylabel("force [N]")
    axes[0].set_title("Task04: BASE-frame wrench mapped by J(q)^T every control step")
    axes[0].grid(True)
    axes[0].legend()

    for i in range(7):
        axes[1].plot(result.time, result.tau_task[:, i], label=f"J{i+1}")
    axes[1].set_ylabel("tau_task [N m]")
    axes[1].grid(True)
    axes[1].legend(ncol=4)

    for i in range(7):
        axes[2].plot(result.time, result.q[:, i] - HOME_Q[i], label=f"J{i+1}")
    axes[2].set_ylabel("q - q_home [rad]")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True)
    axes[2].legend(ncol=4)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_pose_plot(results: list[StaticResult], output_dir: Path) -> Path:
    path = output_dir / "pose_torque_comparison.png"
    x = np.arange(7)
    width = 0.8 / len(results)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, result in enumerate(results):
        offset = (i - 0.5 * (len(results) - 1)) * width
        ax.bar(x + offset, result.tau_task, width=width, label=result.name)

    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"J{i}" for i in range(1, 8)])
    ax.set_ylabel("tau_task = J^T W [N m]")
    ax.set_title("Same BASE-frame wrench at different robot poses")
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _next_user_geom(viewer):
    scene = viewer.user_scn
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("MuJoCo viewer user scene has no free geom slots")
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    return geom


def _add_sphere(viewer, position: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    geom = _next_user_geom(viewer)
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=float),
        np.asarray(position, dtype=float),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )


def _add_arrow(
    viewer,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    rgba: np.ndarray,
) -> None:
    geom = _next_user_geom(viewer)
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        float(width),
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )


def draw_viewer_force(adapter: MuJoCoAdapter, viewer, wrench_base: np.ndarray) -> None:
    viewer.user_scn.ngeom = 0
    p_world, _ = adapter.get_tcp_pose(frame="world")
    base_id = mujoco.mj_name2id(adapter.model, mujoco.mjtObj.mjOBJ_BODY, "base")
    r_world_base = np.asarray(adapter.data.xmat[base_id], dtype=float).reshape(3, 3)
    force_world = r_world_base @ wrench_base[:3]
    magnitude = float(np.linalg.norm(force_world))

    _add_sphere(
        viewer,
        p_world,
        0.022,
        np.array([1.0, 1.0, 0.15, 1.0], dtype=np.float32),
    )
    if magnitude <= 1e-12:
        return

    direction = force_world / magnitude
    # Offset the visual arrow from the robot so it remains visible.
    reference = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(direction, reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    lateral = reference - np.dot(reference, direction) * direction
    lateral /= np.linalg.norm(lateral)
    start = p_world + 0.10 * lateral
    end = start + 0.055 * magnitude * direction

    _add_arrow(
        viewer,
        start,
        end,
        0.022,
        np.array([1.0, 0.55, 0.05, 1.0], dtype=np.float32),
    )


def replay_viewer(model_path: Path, magnitude: float, frequency: float) -> None:
    import mujoco.viewer

    adapter = MuJoCoAdapter.from_xml_path(str(model_path))
    disable_builtin_position_actuators(adapter.model)
    set_q(adapter, HOME_Q)
    mapper = JacobianTransposeWrenchMapper(dof=7)
    posture = make_posture_controller()

    print("\n=== Task04 viewer replay ===")
    print("Yellow sphere : TCP")
    print("Orange arrow  : slowly rotating desired force, expressed in BASE frame")
    print("Control law   : tau_total = PD+g posture torque + J(q)^T W_des")
    print("This is wrench-to-torque mapping, NOT contact-force regulation yet.")
    print("Close the viewer to stop.")

    with mujoco.viewer.launch_passive(adapter.model, adapter.data) as viewer:
        viewer.cam.lookat[:] = np.array([0.30, 0.0, 0.45])
        viewer.cam.distance = 1.35
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0

        last_wall = time.time()
        while viewer.is_running():
            t = float(adapter.data.time)
            q = adapter.get_q()
            qdot = adapter.get_qdot()
            wrench = dynamic_wrench(t, magnitude, frequency)
            tau_task = mapper.compute(adapter.get_jacobian("base"), wrench)
            tau_posture = posture.compute(
                q=q,
                qdot=qdot,
                q_des=HOME_Q,
                qdot_des=np.zeros(7),
                tau_ff=adapter.get_gravity(),
            )
            tau_total = np.clip(tau_posture + tau_task, -TORQUE_LIMITS, TORQUE_LIMITS)
            adapter.set_joint_torque(tau_total)
            adapter.step()

            draw_viewer_force(adapter, viewer, wrench)
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

    output_dir = ROOT / "outputs" / "task04"
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = MuJoCoAdapter.from_xml_path(str(model_path))
    disable_builtin_position_actuators(adapter.model)
    mapper = JacobianTransposeWrenchMapper(dof=7)
    wrench_static = np.array([0.0, 0.0, args.force_z, 0.0, 0.0, 0.0])

    print("\n=== Task04 convention ===")
    print("Twist/Jacobian frame : BASE")
    print("Wrench frame         : BASE")
    print("J row order          : [vx, vy, vz, wx, wy, wz]")
    print("Wrench order         : [Fx, Fy, Fz, Mx, My, Mz]")
    print(f"Static W_des         : {wrench_static}")

    static_results = []
    for name, q in (("HOME", HOME_Q), ("POSE_A", POSE_A), ("POSE_B", POSE_B)):
        result = static_mapping(adapter, name, q, wrench_static, mapper)
        static_results.append(result)
        print(f"\n=== {name} ===")
        print(f"q [rad] = {np.array2string(result.q, precision=5, suppress_small=True)}")
        print("J_base =")
        print(np.array2string(result.jacobian, precision=5, suppress_small=True))
        print("tau_task = J^T W_des [N m] =")
        print(np.array2string(result.tau_task, precision=5, suppress_small=True))
        print(f"sigma_min(Jv) = {result.sigma_min_jv:.6f}")

    # Virtual-power identity: qdot^T tau == xdot^T W.
    home = static_results[0]
    qdot_test = np.array([0.10, -0.08, 0.06, -0.04, 0.03, -0.02, 0.01])
    xdot_test = home.jacobian @ qdot_test
    joint_power = float(qdot_test @ home.tau_task)
    cart_power = float(xdot_test @ wrench_static)
    power_error = abs(joint_power - cart_power)
    print("\n=== Virtual-power check at HOME ===")
    print(f"qdot^T tau = {joint_power:.9f} W")
    print(f"xdot^T W   = {cart_power:.9f} W")
    print(f"|difference| = {power_error:.3e} W")

    low_sigma_q, low_sigma = search_low_sigma_pose(
        adapter, args.singularity_samples, args.seed
    )
    low_result = static_mapping(
        adapter, "LOW_SIGMA_DIAGNOSTIC", low_sigma_q, wrench_static, mapper
    )
    jv_low = low_result.jacobian[:3, :]
    u, singular_values, _ = np.linalg.svd(jv_low, full_matrices=False)
    weak_force_dir = u[:, -1]
    weak_force = args.force_z * weak_force_dir
    weak_tau = jv_low.T @ weak_force

    print("\n=== Translational near-singularity diagnostic ===")
    print("NOTE: this randomly searched q is a kinematic diagnostic only; it is not a motion target.")
    print(f"samples              : {args.singularity_samples}")
    print(f"q_low_sigma [rad]    : {np.array2string(low_sigma_q, precision=5, suppress_small=True)}")
    print(f"singular values Jv   : {np.array2string(singular_values, precision=6)}")
    print(f"sigma_min(Jv)        : {low_sigma:.6e}")
    print(f"same +BASE-Z tau     : {np.array2string(low_result.tau_task, precision=5, suppress_small=True)}")
    print(f"weak force dir BASE  : {np.array2string(weak_force_dir, precision=5, suppress_small=True)}")
    print(f"|weak force|         : {np.linalg.norm(weak_force):.3f} N")
    print(f"|Jv^T F_weak|        : {np.linalg.norm(weak_tau):.6f} N m")
    print("Important: near singularity does NOT mean J^T W must blow up; inverse-Jacobian mappings are the ones that can blow up.")

    loop_result = simulate_loop(
        model_path=model_path,
        duration=args.duration,
        magnitude=args.dynamic_force,
        frequency=args.frequency,
    )
    csv_path = save_loop_csv(loop_result, output_dir)
    loop_plot = save_loop_plot(loop_result, output_dir)
    pose_plot = save_pose_plot(static_results + [low_result], output_dir)

    peak_task = np.max(np.abs(loop_result.tau_task), axis=0)
    peak_total = np.max(np.abs(loop_result.tau_total), axis=0)
    max_deflection = np.max(np.abs(loop_result.q - HOME_Q), axis=0)

    print("\n=== Dynamic-loop summary ===")
    print("peak |tau_task| per joint [N m] =")
    print(np.array2string(peak_task, precision=5, suppress_small=True))
    print("peak |tau_total| per joint [N m] =")
    print(np.array2string(peak_total, precision=5, suppress_small=True))
    print("max |q-q_home| per joint [rad] =")
    print(np.array2string(max_deflection, precision=5, suppress_small=True))
    print(f"CSV       : {csv_path}")
    print(f"Loop plot : {loop_plot}")
    print(f"Pose plot : {pose_plot}")

    if power_error > 1e-10:
        raise RuntimeError(f"Virtual-power identity failed: {power_error:.3e}")
    for result in static_results + [low_result]:
        if result.jacobian.shape != (6, 7):
            raise RuntimeError(f"Unexpected J shape in {result.name}: {result.jacobian.shape}")
        if not np.isfinite(result.tau_task).all():
            raise RuntimeError(f"Non-finite tau_task in {result.name}")

    print("\nPASS: frame convention is explicit, J^T W mapping is finite, virtual power is conserved, and the dynamic loop completed.")

    if args.viewer:
        replay_viewer(model_path, args.dynamic_force, args.frequency)


if __name__ == "__main__":
    main()
