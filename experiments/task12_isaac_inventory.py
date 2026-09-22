"""Task12 Stage 0: inspect an existing Isaac Sim FR3 USD scene.

This script does not run any controller.  It loads a USD stage in Isaac Sim 4.5,
finds articulation roots, and prints the body / DOF ordering needed to build the
Isaac adapter safely.

Why this step exists:
- Task12 must not guess joint ordering;
- Jacobian body indexing depends on articulation body ordering;
- torque control must be attached to the correct seven FR3 joints.

Run this script with Isaac Sim's bundled python.sh, not the project's .venv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import UsdPhysics
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation


ARM_JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", required=True, help="Absolute path to the Isaac Sim USD scene")
    parser.add_argument(
        "--robot-prim",
        default=None,
        help="Optional articulation-root prim path. If omitted, all roots are listed.",
    )
    return parser.parse_args()


def articulation_roots():
    stage = stage_utils.get_current_stage()
    roots = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
    return roots


def main() -> None:
    args = parse_args()
    usd_path = Path(args.usd).expanduser().resolve()
    if not usd_path.exists():
        raise FileNotFoundError(usd_path)

    print("\n=== Task12 Stage 0: Isaac scene inventory ===")
    print("USD:", usd_path)

    stage_utils.open_stage(str(usd_path))
    while stage_utils.is_stage_loading():
        simulation_app.update()

    roots = articulation_roots()
    print("\nArticulation roots:")
    if not roots:
        print("  <none>")
        print("\nNo articulation root found. Check whether the USD contains/imports the FR3 physics articulation.")
        return

    for i, root in enumerate(roots):
        print(f"  [{i}] {root}")

    if args.robot_prim is None:
        print(
            "\nRe-run with --robot-prim <one path above> to inspect joint/body ordering."
        )
        return

    if args.robot_prim not in roots:
        raise RuntimeError(
            f"--robot-prim {args.robot_prim!r} is not one of the detected articulation roots"
        )

    world = World(stage_units_in_meters=1.0)
    robot = world.scene.add(
        SingleArticulation(
            prim_path=args.robot_prim,
            name="task12_inventory_robot",
        )
    )

    world.reset()
    robot.initialize()
    world.step(render=False)

    view = getattr(robot, "_articulation_view", None)
    if view is None:
        raise RuntimeError("SingleArticulation did not expose an articulation view")

    dof_names = list(view.dof_names)
    body_names = list(view.body_names)

    print("\nDOF ordering:")
    for i, name in enumerate(dof_names):
        marker = "  <ARM>" if name in ARM_JOINTS else ""
        print(f"  [{i:2d}] {name}{marker}")

    print("\nRigid-body ordering:")
    for i, name in enumerate(body_names):
        print(f"  [{i:2d}] {name}")

    print("\nFR3 arm-joint lookup:")
    missing = []
    for name in ARM_JOINTS:
        if name in dof_names:
            print(f"  {name:<12} -> DOF index {dof_names.index(name)}")
        else:
            missing.append(name)
            print(f"  {name:<12} -> MISSING")

    print("\nPhysics tensor shapes:")
    print("  Jacobian shape    :", tuple(view.get_jacobian_shape()))
    print("  Mass-matrix shape :", tuple(view.get_mass_matrix_shape()))

    if len(body_names) > 1:
        print(
            "\nCandidate TCP bodies (inspect these first):",
            ", ".join(
                name
                for name in body_names
                if any(key in name.lower() for key in ("link7", "hand", "tcp", "tool", "ee"))
            ) or "<none matched automatically>",
        )

    if missing:
        print(
            "\nSTATUS: NOT READY — expected FR3 arm joint names are missing. "
            "We must map the actual USD joint names before writing the adapter."
        )
    else:
        print(
            "\nSTATUS: JOINT ORDER READY — all seven fr3_joint1..7 were found. "
            "Send this output back before the Jacobian/TCP mapping step."
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
