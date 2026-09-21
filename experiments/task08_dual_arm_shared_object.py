"""Task08: two FR3 arms rigidly carrying one shared object.

This baseline intentionally avoids QP wrench allocation.  A single desired
object pose generates consistent left/right TCP references.  Each arm then uses
its own Cartesian impedance controller and Jacobian-transpose torque mapping.

The shared box is connected to both arms with MuJoCo weld equality constraints,
representing an ideal rigid grasp.  This creates the first closed-chain,
dual-arm system in the learning roadmap.

Important: the logged left/right wrenches are CONTROLLER-COMMANDED task-space
wrenches, not measured grasp constraint wrenches.  Actual wrench allocation is
introduced later.
"""

from __future__ import annotations

import argparse
import copy
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
from sim.mujoco_adapter import MuJoCoAdapter


HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

LEFT_PREFIX = "left_"
RIGHT_PREFIX = "right_"

LEFT_BASE_POS = np.array([-0.625, 0.0, 0.0])
RIGHT_BASE_POS = np.array([0.625, 0.0, 0.0])
LEFT_BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
RIGHT_BASE_QUAT = np.array([0.0, 0.0, 0.0, 1.0])  # 180 deg about world Z.

BOX_MASS = 1.0
LIFT_DISTANCE = 0.040
MOTION_DELAY = 1.0
MOTION_DURATION = 4.0
HOLD_DURATION = 2.0
DURATION = MOTION_DELAY + MOTION_DURATION + HOLD_DURATION

GRAVITY = 9.81


@dataclass
class RunResult:
    time: np.ndarray
    object_position: np.ndarray
    object_position_des: np.ndarray
    object_orientation_error: np.ndarray
    left_position: np.ndarray
    left_position_des: np.ndarray
    right_position: np.ndarray
    right_position_des: np.ndarray
    relative_position_error: np.ndarray
    left_wrench_cmd: np.ndarray
    right_wrench_cmd: np.ndarray
    left_tau: np.ndarray
    right_tau: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path to MuJoCo Menagerie franka_fr3/scene.xml or fr3.xml",
    )
    parser.add_argument("--viewer", action="store_true", help="Replay Task08 in MuJoCo viewer")
    return parser.parse_args()


def resolve_fr3_xml(model_path: Path) -> Path:
    if model_path.name == "fr3.xml":
        return model_path
    candidate = model_path.parent / "fr3.xml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not locate fr3.xml next to {model_path}. "
        "Pass franka_fr3/scene.xml or franka_fr3/fr3.xml."
    )


def _fmt(values: np.ndarray) -> str:
    return " ".join(f"{float(x):.9g}" for x in values)


def _prefixed_robot_body(source_body: ET.Element, prefix: str) -> ET.Element:
    body = copy.deepcopy(source_body)

    # All object names inside the kinematic subtree must be unique.
    for elem in body.iter():
        name = elem.get("name")
        if name:
            elem.set("name", prefix + name)

        # Disable physical collisions for this baseline.  Shared-object
        # coordination is enforced only by the explicit weld constraints.
        if elem.tag == "geom" and elem.get("class") == "collision":
            elem.set("contype", "0")
            elem.set("conaffinity", "0")

    return body


def _copy_if_present(src_root: ET.Element, dst_root: ET.Element, tag: str) -> None:
    elem = src_root.find(tag)
    if elem is not None:
        dst_root.append(copy.deepcopy(elem))


def _rotation_matrix_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized MuJoCo wxyz quaternion."""
    r = np.asarray(r, dtype=float).reshape(3, 3)
    tr = float(np.trace(r))

    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=float)
    q /= np.linalg.norm(q)
    return q


def _weld_relpose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body1_name: str,
    body2_name: str,
) -> np.ndarray:
    """Return explicit weld relpose satisfied by the current configuration.

    body1 is the reference frame.  The translation is body2's origin expressed
    in body1, and the quaternion is body2's orientation relative to body1.
    """
    id1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
    id2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body2_name)
    if id1 < 0 or id2 < 0:
        raise RuntimeError(f"Missing body for weld relpose: {body1_name}, {body2_name}")

    p1 = np.asarray(data.xpos[id1], dtype=float)
    p2 = np.asarray(data.xpos[id2], dtype=float)
    r1 = np.asarray(data.xmat[id1], dtype=float).reshape(3, 3)
    r2 = np.asarray(data.xmat[id2], dtype=float).reshape(3, 3)

    p_rel = r1.T @ (p2 - p1)
    r_rel = r1.T @ r2
    q_rel = _rotation_matrix_to_quat_wxyz(r_rel)
    return np.concatenate((p_rel, q_rel))


def build_dual_scene(fr3_xml: Path, output_xml: Path) -> None:
    src_tree = ET.parse(fr3_xml)
    src_root = src_tree.getroot()

    source_base = src_root.find("./worldbody/body")
    if source_base is None:
        raise RuntimeError("Could not find FR3 base body in fr3.xml")

    root = ET.Element("mujoco", {"model": "task08_dual_fr3_shared_object"})

    compiler = copy.deepcopy(src_root.find("compiler"))
    if compiler is None:
        compiler = ET.Element("compiler", {"angle": "radian"})
    compiler.set("meshdir", str((fr3_xml.parent / "assets").resolve()))
    root.append(compiler)

    _copy_if_present(src_root, root, "option")
    _copy_if_present(src_root, root, "default")
    _copy_if_present(src_root, root, "asset")

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {
            "pos": "0 0 1.7",
            "dir": "0 0 -1",
            "directional": "true",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "2 2 0.05",
            "rgba": "0.18 0.22 0.28 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    left_base = _prefixed_robot_body(source_base, LEFT_PREFIX)
    left_base.set("pos", _fmt(LEFT_BASE_POS))
    left_base.set("quat", _fmt(LEFT_BASE_QUAT))
    worldbody.append(left_base)

    right_base = _prefixed_robot_body(source_base, RIGHT_PREFIX)
    right_base.set("pos", _fmt(RIGHT_BASE_POS))
    right_base.set("quat", _fmt(RIGHT_BASE_QUAT))
    worldbody.append(right_base)

    # The box is centered between the two HOME TCPs.  Explicit weld relposes
    # are computed below from the HOME configuration (not from qpos0).
    box_body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "shared_box",
            "pos": "0 0 0.6245",
        },
    )
    ET.SubElement(box_body, "freejoint", {"name": "shared_box_freejoint"})
    ET.SubElement(
        box_body,
        "geom",
        {
            "name": "shared_box_geom",
            "type": "box",
            "size": "0.070 0.090 0.040",
            "mass": f"{BOX_MASS}",
            "rgba": "0.85 0.45 0.12 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    # IMPORTANT: body-based welds with no explicit relpose inherit the relative
    # transform from model.qpos0.  Our experiment later moves both robots from
    # qpos0=0 to HOME_Q, so using the implicit weld pose would create a huge
    # constraint violation at t=0.  Build a temporary unconstrained model,
    # place both arms at HOME_Q, and explicitly encode the weld relposes that
    # are satisfied in that HOME configuration.
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    preweld_xml = output_xml.with_name(output_xml.stem + "_preweld.xml")
    ET.ElementTree(root).write(preweld_xml, encoding="unicode")

    pre_model = mujoco.MjModel.from_xml_path(str(preweld_xml))
    pre_data = mujoco.MjData(pre_model)
    pre_left, pre_right = make_adapters(pre_model, pre_data)
    set_home(pre_left, pre_right)

    left_relpose = _weld_relpose(
        pre_model, pre_data, "shared_box", "left_fr3_link7"
    )
    right_relpose = _weld_relpose(
        pre_model, pre_data, "shared_box", "right_fr3_link7"
    )

    equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "weld",
        {
            "name": "left_grasp_weld",
            "body1": "shared_box",
            "body2": "left_fr3_link7",
            "relpose": _fmt(left_relpose),
            "solref": "0.010 1",
            "solimp": "0.95 0.99 0.001 0.5 2",
        },
    )
    ET.SubElement(
        equality,
        "weld",
        {
            "name": "right_grasp_weld",
            "body1": "shared_box",
            "body2": "right_fr3_link7",
            "relpose": _fmt(right_relpose),
            "solref": "0.010 1",
            "solimp": "0.95 0.99 0.001 0.5 2",
        },
    )

    ET.ElementTree(root).write(output_xml, encoding="unicode")
    preweld_xml.unlink(missing_ok=True)


def make_adapters(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[MuJoCoAdapter, MuJoCoAdapter]:
    left = MuJoCoAdapter(
        model,
        data,
        joint_names=[f"left_fr3_joint{i}" for i in range(1, 8)],
        tcp_site_name="left_attachment_site",
        base_body_name="left_base",
    )
    right = MuJoCoAdapter(
        model,
        data,
        joint_names=[f"right_fr3_joint{i}" for i in range(1, 8)],
        tcp_site_name="right_attachment_site",
        base_body_name="right_base",
    )
    return left, right


def set_home(left: MuJoCoAdapter, right: MuJoCoAdapter) -> None:
    for adapter in (left, right):
        for joint, value in zip(adapter.joint_map, HOME_Q):
            adapter.data.qpos[joint.qpos_adr] = float(value)
            adapter.data.qvel[joint.dof_adr] = 0.0
    adapter = left
    adapter.data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(adapter.model, adapter.data)


def build_controller() -> CartesianImpedance6D:
    return CartesianImpedance6D.from_gains(
        translational_stiffness=[350.0, 350.0, 450.0],
        translational_damping=[38.0, 38.0, 45.0],
        rotational_stiffness=[18.0, 18.0, 18.0],
        rotational_damping=[3.5, 3.5, 3.5],
        force_limits=[45.0, 45.0, 45.0],
        moment_limits=[7.0, 7.0, 7.0],
    )


def smooth_step(t: float) -> tuple[float, float]:
    if t <= MOTION_DELAY:
        return 0.0, 0.0
    if t >= MOTION_DELAY + MOTION_DURATION:
        return 1.0, 0.0

    r = (t - MOTION_DELAY) / MOTION_DURATION
    s = 10 * r**3 - 15 * r**4 + 6 * r**5
    ds = (30 * r**2 - 60 * r**3 + 30 * r**4) / MOTION_DURATION
    return float(s), float(ds)


def object_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shared_box")
    if body_id < 0:
        raise RuntimeError("shared_box body missing")
    p = np.asarray(data.xpos[body_id], dtype=float).copy()
    r = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3).copy()
    return p, r


def simulate(scene_xml: Path) -> RunResult:
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    left, right = make_adapters(model, data)
    set_home(left, right)

    controller_l = build_controller()
    controller_r = build_controller()

    p_obj0, r_obj0 = object_pose(model, data)
    p_l0, r_l0 = left.get_tcp_pose(frame="world")
    p_r0, r_r0 = right.get_tcp_pose(frame="world")

    # Object->grasp offsets in the initial rigid-grasp configuration.
    rel_l = r_obj0.T @ (p_l0 - p_obj0)
    rel_r = r_obj0.T @ (p_r0 - p_obj0)
    r_obj_to_l = r_obj0.T @ r_l0
    r_obj_to_r = r_obj0.T @ r_r0
    relative_tcp0 = p_r0 - p_l0

    dt = float(model.opt.timestep)
    steps = int(np.ceil(DURATION / dt)) + 1

    t_log = np.zeros(steps)
    obj_log = np.zeros((steps, 3))
    obj_des_log = np.zeros((steps, 3))
    obj_ori_err_log = np.zeros((steps, 3))
    l_log = np.zeros((steps, 3))
    l_des_log = np.zeros((steps, 3))
    r_log = np.zeros((steps, 3))
    r_des_log = np.zeros((steps, 3))
    rel_err_log = np.zeros((steps, 3))
    wl_log = np.zeros((steps, 6))
    wr_log = np.zeros((steps, 6))
    tau_l_log = np.zeros((steps, 7))
    tau_r_log = np.zeros((steps, 7))

    nominal_half_weight = 0.5 * BOX_MASS * GRAVITY

    for k in range(steps):
        t = float(data.time)
        p_obj, r_obj = object_pose(model, data)
        p_l, r_l = left.get_tcp_pose(frame="world")
        p_r, r_r = right.get_tcp_pose(frame="world")

        j_l = left.get_jacobian(frame="world")
        j_r = right.get_jacobian(frame="world")
        twist_l = j_l @ left.get_qdot()
        twist_r = j_r @ right.get_qdot()

        s, ds = smooth_step(t)
        p_obj_des = p_obj0 + np.array([0.0, 0.0, LIFT_DISTANCE * s])
        v_obj_des = np.array([0.0, 0.0, LIFT_DISTANCE * ds])
        r_obj_des = r_obj0

        p_l_des = p_obj_des + r_obj_des @ rel_l
        p_r_des = p_obj_des + r_obj_des @ rel_r
        r_l_des = r_obj_des @ r_obj_to_l
        r_r_des = r_obj_des @ r_obj_to_r

        w_l = controller_l.compute(
            p_l,
            r_l,
            twist_l[:3],
            twist_l[3:],
            p_l_des,
            r_l_des,
            desired_linear_velocity=v_obj_des,
        )
        w_r = controller_r.compute(
            p_r,
            r_r,
            twist_r[:3],
            twist_r[3:],
            p_r_des,
            r_r_des,
            desired_linear_velocity=v_obj_des,
        )

        # Simple equal load-sharing baseline.  Task09 replaces this hand-chosen
        # 50/50 split with an explicit wrench allocation problem.
        w_l[2] += nominal_half_weight
        w_r[2] += nominal_half_weight

        tau_l = np.clip(j_l.T @ w_l + left.get_gravity(), -TORQUE_LIMITS, TORQUE_LIMITS)
        tau_r = np.clip(j_r.T @ w_r + right.get_gravity(), -TORQUE_LIMITS, TORQUE_LIMITS)

        left.set_joint_torque(tau_l)
        right.set_joint_torque(tau_r)

        t_log[k] = t
        obj_log[k] = p_obj
        obj_des_log[k] = p_obj_des
        obj_ori_err_log[k] = rotation_error_base(r_obj, r_obj_des)
        l_log[k] = p_l
        l_des_log[k] = p_l_des
        r_log[k] = p_r
        r_des_log[k] = p_r_des
        rel_err_log[k] = (p_r - p_l) - relative_tcp0
        wl_log[k] = w_l
        wr_log[k] = w_r
        tau_l_log[k] = tau_l
        tau_r_log[k] = tau_r

        if k < steps - 1:
            mujoco.mj_step(model, data)

        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f"NaN/Inf in Task08 at t={t:.4f}s")

    return RunResult(
        time=t_log,
        object_position=obj_log,
        object_position_des=obj_des_log,
        object_orientation_error=obj_ori_err_log,
        left_position=l_log,
        left_position_des=l_des_log,
        right_position=r_log,
        right_position_des=r_des_log,
        relative_position_error=rel_err_log,
        left_wrench_cmd=wl_log,
        right_wrench_cmd=wr_log,
        left_tau=tau_l_log,
        right_tau=tau_r_log,
    )


def metrics(result: RunResult) -> dict[str, float]:
    obj_err = result.object_position_des - result.object_position
    left_err = result.left_position_des - result.left_position
    right_err = result.right_position_des - result.right_position

    return {
        "object_rms_mm": 1000.0 * float(np.sqrt(np.mean(np.sum(obj_err**2, axis=1)))),
        "object_max_mm": 1000.0 * float(np.max(np.linalg.norm(obj_err, axis=1))),
        "object_final_z_mm": 1000.0
        * float(result.object_position[-1, 2] - result.object_position[0, 2]),
        "left_tcp_rms_mm": 1000.0
        * float(np.sqrt(np.mean(np.sum(left_err**2, axis=1)))),
        "right_tcp_rms_mm": 1000.0
        * float(np.sqrt(np.mean(np.sum(right_err**2, axis=1)))),
        "relative_rms_mm": 1000.0
        * float(np.sqrt(np.mean(np.sum(result.relative_position_error**2, axis=1)))),
        "relative_max_mm": 1000.0
        * float(np.max(np.linalg.norm(result.relative_position_error, axis=1))),
        "object_ori_max_deg": float(
            np.max(np.linalg.norm(result.object_orientation_error, axis=1)) * 180.0 / np.pi
        ),
        "left_peak_tau": float(np.max(np.abs(result.left_tau))),
        "right_peak_tau": float(np.max(np.abs(result.right_tau))),
        "left_peak_force_cmd": float(np.max(np.linalg.norm(result.left_wrench_cmd[:, :3], axis=1))),
        "right_peak_force_cmd": float(np.max(np.linalg.norm(result.right_wrench_cmd[:, :3], axis=1))),
    }


def save_csv(result: RunResult, output_dir: Path) -> Path:
    path = output_dir / "dual_arm_shared_object.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        headers = [
            "time",
            "obj_x", "obj_y", "obj_z",
            "obj_des_x", "obj_des_y", "obj_des_z",
            "left_x", "left_y", "left_z",
            "left_des_x", "left_des_y", "left_des_z",
            "right_x", "right_y", "right_z",
            "right_des_x", "right_des_y", "right_des_z",
            "rel_ex", "rel_ey", "rel_ez",
        ]
        headers += [f"WL_{x}" for x in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]
        headers += [f"WR_{x}" for x in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]
        headers += [f"tauL{i}" for i in range(1, 8)]
        headers += [f"tauR{i}" for i in range(1, 8)]
        writer.writerow(headers)

        for k in range(len(result.time)):
            writer.writerow([
                result.time[k],
                *result.object_position[k],
                *result.object_position_des[k],
                *result.left_position[k],
                *result.left_position_des[k],
                *result.right_position[k],
                *result.right_position_des[k],
                *result.relative_position_error[k],
                *result.left_wrench_cmd[k],
                *result.right_wrench_cmd[k],
                *result.left_tau[k],
                *result.right_tau[k],
            ])
    return path


def save_plot(result: RunResult, output_dir: Path) -> Path:
    path = output_dir / "dual_arm_shared_object.png"

    obj_err = 1000.0 * (result.object_position_des - result.object_position)
    rel_err = 1000.0 * np.linalg.norm(result.relative_position_error, axis=1)

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 11))

    axes[0].plot(
        result.time,
        1000.0 * (result.object_position_des[:, 2] - result.object_position_des[0, 2]),
        linestyle="--",
        label="object z desired",
    )
    axes[0].plot(
        result.time,
        1000.0 * (result.object_position[:, 2] - result.object_position[0, 2]),
        label="object z actual",
    )
    axes[0].set_ylabel("object lift [mm]")

    axes[1].plot(result.time, obj_err[:, 0], label="ex")
    axes[1].plot(result.time, obj_err[:, 1], label="ey")
    axes[1].plot(result.time, obj_err[:, 2], label="ez")
    axes[1].set_ylabel("object pos error [mm]")

    axes[2].plot(result.time, rel_err, label="TCP relative-position error norm")
    axes[2].set_ylabel("relative error [mm]")

    axes[3].plot(result.time, result.left_wrench_cmd[:, 2], label="left Fz cmd")
    axes[3].plot(result.time, result.right_wrench_cmd[:, 2], label="right Fz cmd")
    axes[3].set_ylabel("commanded Fz [N]")
    axes[3].set_xlabel("time [s]")

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


def show_viewer(scene_xml: Path) -> None:
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    left, right = make_adapters(model, data)
    set_home(left, right)

    controller_l = build_controller()
    controller_r = build_controller()

    p_obj0, r_obj0 = object_pose(model, data)
    p_l0, r_l0 = left.get_tcp_pose(frame="world")
    p_r0, r_r0 = right.get_tcp_pose(frame="world")

    rel_l = r_obj0.T @ (p_l0 - p_obj0)
    rel_r = r_obj0.T @ (p_r0 - p_obj0)
    r_obj_to_l = r_obj0.T @ r_l0
    r_obj_to_r = r_obj0.T @ r_r0
    nominal_half_weight = 0.5 * BOX_MASS * GRAVITY
    dt = float(model.opt.timestep)

    print("\n=== Task08 viewer ===")
    print("Orange box  : shared rigid object")
    print("Green sphere: desired object center")
    print("Two FR3 arms are rigidly connected to the box by weld constraints.")
    print("The desired object lifts +WORLD-Z by 40 mm.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = p_obj0
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -18.0

        while viewer.is_running() and data.time < DURATION:
            t = float(data.time)
            p_l, r_l = left.get_tcp_pose(frame="world")
            p_r, r_r = right.get_tcp_pose(frame="world")
            j_l = left.get_jacobian(frame="world")
            j_r = right.get_jacobian(frame="world")
            twist_l = j_l @ left.get_qdot()
            twist_r = j_r @ right.get_qdot()

            s, ds = smooth_step(t)
            p_obj_des = p_obj0 + np.array([0.0, 0.0, LIFT_DISTANCE * s])
            v_obj_des = np.array([0.0, 0.0, LIFT_DISTANCE * ds])

            p_l_des = p_obj_des + r_obj0 @ rel_l
            p_r_des = p_obj_des + r_obj0 @ rel_r
            r_l_des = r_obj0 @ r_obj_to_l
            r_r_des = r_obj0 @ r_obj_to_r

            w_l = controller_l.compute(
                p_l, r_l, twist_l[:3], twist_l[3:], p_l_des, r_l_des,
                desired_linear_velocity=v_obj_des,
            )
            w_r = controller_r.compute(
                p_r, r_r, twist_r[:3], twist_r[3:], p_r_des, r_r_des,
                desired_linear_velocity=v_obj_des,
            )
            w_l[2] += nominal_half_weight
            w_r[2] += nominal_half_weight

            tau_l = np.clip(j_l.T @ w_l + left.get_gravity(), -TORQUE_LIMITS, TORQUE_LIMITS)
            tau_r = np.clip(j_r.T @ w_r + right.get_gravity(), -TORQUE_LIMITS, TORQUE_LIMITS)
            left.set_joint_torque(tau_l)
            right.set_joint_torque(tau_r)
            mujoco.mj_step(model, data)

            viewer.user_scn.ngeom = 0
            _add_sphere(
                viewer,
                p_obj_des + np.array([0.0, 0.0, 0.07]),
                0.018,
                np.array([0.1, 1.0, 0.2, 1.0]),
            )
            viewer.sync()
            time.sleep(max(0.0, dt))


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    fr3_xml = resolve_fr3_xml(model_path)

    output_dir = ROOT / "outputs" / "task08"
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_xml = output_dir / "task08_dual_fr3_shared_object.xml"
    build_dual_scene(fr3_xml, scene_xml)

    result = simulate(scene_xml)
    m = metrics(result)

    print("\n=== Task08 dual-FR3 shared-object baseline ===")
    print("Grasp model            : ideal rigid weld constraints")
    print("Object command         : +WORLD-Z lift, 40 mm")
    print("Object mass            : 1.0 kg")
    print("Baseline load split    : fixed 50/50 gravity feedforward")
    print("Wrench logs            : commanded task-space wrenches, not measured grasp wrench")
    print("QP allocation          : NOT used in Task08")

    print("\n=== Task08 metrics ===")
    print(f"Object RMS pos error   : {m['object_rms_mm']:.3f} mm")
    print(f"Object max pos error   : {m['object_max_mm']:.3f} mm")
    print(f"Actual object Z travel : {m['object_final_z_mm']:.3f} mm")
    print(f"Left TCP RMS error     : {m['left_tcp_rms_mm']:.3f} mm")
    print(f"Right TCP RMS error    : {m['right_tcp_rms_mm']:.3f} mm")
    print(f"Relative grasp RMS err : {m['relative_rms_mm']:.4f} mm")
    print(f"Relative grasp max err : {m['relative_max_mm']:.4f} mm")
    print(f"Object max ori error   : {m['object_ori_max_deg']:.4f} deg")
    print(f"Left peak |F_cmd|      : {m['left_peak_force_cmd']:.3f} N")
    print(f"Right peak |F_cmd|     : {m['right_peak_force_cmd']:.3f} N")
    print(f"Left peak |tau|        : {m['left_peak_tau']:.3f} N m")
    print(f"Right peak |tau|       : {m['right_peak_tau']:.3f} N m")

    plot_path = save_plot(result, output_dir)
    csv_path = save_csv(result, output_dir)

    print(f"\nPlot : {plot_path}")
    print(f"CSV  : {csv_path}")
    print(f"Scene: {scene_xml}")
    print(
        "\nPASS means the dual-arm closed-chain baseline ran without NaN/Inf. "
        "Interpretation target: one object reference creates two consistent TCP "
        "references, but independent arm controllers still do not solve optimal "
        "wrench distribution or internal-force regulation."
    )

    if args.viewer:
        show_viewer(scene_xml)


if __name__ == "__main__":
    main()
