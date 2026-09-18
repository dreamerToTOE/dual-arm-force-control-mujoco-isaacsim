"""Task06 Stage 2: same stabilized PI controller on different contact surfaces.

Only the MuJoCo contact softness is changed.  The force controller, filter,
command slew-rate limit, robot model, approach, target force, and duration are
kept identical so the effect of environment contact dynamics can be observed.

For positive MuJoCo solref = [timeconst, dampratio], a larger time constant
produces a softer/slower contact constraint and a smaller time constant produces
a firmer/faster one.  Stage 2 keeps dampratio=1.0 and sweeps only timeconst.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.task06_contact_force_control import (
    build_contact_scene,
    resolve_fr3_xml,
)
from experiments.task06_contact_force_control_stable import (
    DURATION,
    F_DES,
    STABILITY_TAIL,
    force_metrics,
    post_contact_mask,
    show_viewer,
    simulate_case,
)


# Same controller for every surface.  Only solref time constant changes.
SURFACES = {
    "soft": 0.050,
    "baseline": 0.015,
    "firm": 0.008,
}
DAMPING_RATIO = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to Menagerie FR3 scene.xml or fr3.xml")
    parser.add_argument("--viewer", action="store_true", help="Replay one selected surface in MuJoCo viewer")
    parser.add_argument(
        "--viewer-surface",
        choices=tuple(SURFACES.keys()),
        default="baseline",
        help="Surface replayed when --viewer is enabled",
    )
    return parser.parse_args()


def build_surface_scene(
    fr3_xml: Path,
    output_xml: Path,
    timeconst: float,
) -> tuple[float, float]:
    """Build Task06 scene and force both contact geoms to use one solref."""

    surface_z, home_tcp_z = build_contact_scene(fr3_xml, output_xml)

    tree = ET.parse(output_xml)
    root = tree.getroot()
    found = set()

    for geom in root.iter("geom"):
        name = geom.get("name")
        if name in {"task06_probe", "task06_surface"}:
            geom.set("solref", f"{timeconst:.6f} {DAMPING_RATIO:.6f}")
            geom.set("solimp", "0.9 0.95 0.001 0.5 2")
            found.add(name)

    missing = {"task06_probe", "task06_surface"} - found
    if missing:
        raise RuntimeError(f"Could not set Task06 contact parameters for: {sorted(missing)}")

    tree.write(output_xml, encoding="unicode")
    return surface_z, home_tcp_z


def tail_mask(result) -> np.ndarray:
    return result.time >= (result.time[-1] - STABILITY_TAIL)


def extra_metrics(result) -> dict[str, float]:
    mask = tail_mask(result)
    mean_clearance = float(np.mean(result.clearance[mask]))
    mean_command = float(np.mean(result.force_command[mask]))
    return {
        "penetration_mm": max(0.0, -1000.0 * mean_clearance),
        "mean_command": mean_command,
    }


def save_csv(name: str, timeconst: float, result, output_dir: Path) -> Path:
    path = output_dir / f"surface_{name}.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        headers = [
            "surface",
            "solref_timeconst",
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
                name,
                timeconst,
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


def save_response_plot(results: dict[str, object], output_dir: Path) -> Path:
    path = output_dir / "surface_contact_comparison.png"
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))

    max_rel = 0.0
    for name, result in results.items():
        mask = post_contact_mask(result)
        t = result.time[mask] - result.contact_time
        max_rel = max(max_rel, float(t[-1]))

        axes[0].plot(
            t,
            result.normal_force_filtered[mask],
            label=f"{name}: tc={SURFACES[name]:.3f}s",
        )
        axes[1].plot(t, result.force_command[mask], label=name)
        axes[2].plot(t, 1000.0 * result.clearance[mask], label=name)

    axes[0].axhline(F_DES, linestyle="--", label="F_des")
    axes[0].set_title("Task06 Stage 2: same PI controller, different contact softness")
    axes[0].set_ylabel("filtered normal force [N]")
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


def save_steady_plot(results: dict[str, object], output_dir: Path) -> Path:
    path = output_dir / "surface_steady_zoom.png"
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, result in results.items():
        start = float(result.time[-1] - STABILITY_TAIL)
        mask = result.time >= start
        t = result.time[mask] - result.contact_time
        ax.plot(t, result.normal_force_raw[mask], label=name)

    ax.axhline(F_DES, linestyle="--", label="F_des")
    ax.set_title(f"Task06 Stage 2: raw force during last {STABILITY_TAIL:.1f} s")
    ax.set_xlabel("time since contact [s]")
    ax.set_ylabel("raw normal force [N]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    fr3_xml = resolve_fr3_xml(model_path)

    output_dir = ROOT / "outputs" / "task06" / "stage2"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes: dict[str, Path] = {}
    surface_z_by_name: dict[str, float] = {}
    home_z = float("nan")

    for name, timeconst in SURFACES.items():
        scene = output_dir / f"task06_surface_{name}.xml"
        surface_z, home_z = build_surface_scene(fr3_xml, scene, timeconst)
        scenes[name] = scene
        surface_z_by_name[name] = surface_z

    # Same stabilized PI controller for every environment.
    results = {
        name: simulate_case(scenes[name], "pi", surface_z_by_name[name])
        for name in SURFACES
    }
    metrics = {name: force_metrics(result) for name, result in results.items()}
    extras = {name: extra_metrics(result) for name, result in results.items()}

    print("\n=== Task06 Stage 2: contact-surface sweep ===")
    print("Controller             : same stabilized PI for every surface")
    print("PI gains               : Kp=0.8, Ki=0.8")
    print("Desired normal force   : 10.00 N")
    print("Force filter tau       : 0.040 s")
    print("Command slew rate      : 20.0 N/s")
    print(f"Simulation duration    : {DURATION:.1f} s")
    print(f"HOME TCP z             : {home_z:.5f} m")
    print("MuJoCo solref form     : [timeconst, damping_ratio], damping_ratio=1.0")
    print("Interpretation         : larger timeconst = softer contact; smaller = firmer")

    print(
        "\nsurface    tc[s]   contact[s]   rise90[s]   settle[s]   "
        "overshoot[%]   steady err[N]   STD[N]   ripple[N]   loss[%]   penetration[mm]"
    )
    print("-" * 132)

    for name in SURFACES:
        m = metrics[name]
        e = extras[name]
        r = results[name]
        print(
            f"{name:<10} {SURFACES[name]:>6.3f}   {r.contact_time:>10.4f}   "
            f"{m['rise90']:>9.4f}   {m['settle']:>9.4f}   "
            f"{m['overshoot']:>12.3f}   {m['steady_error']:>13.4f}   "
            f"{m['force_std']:>7.4f}   {m['force_ripple']:>9.4f}   "
            f"{m['contact_loss_ratio']:>7.3f}   {e['penetration_mm']:>15.5f}"
        )

    response_plot = save_response_plot(results, output_dir)
    steady_plot = save_steady_plot(results, output_dir)
    csv_paths = [
        save_csv(name, SURFACES[name], results[name], output_dir)
        for name in SURFACES
    ]

    print(f"\nResponse plot : {response_plot}")
    print(f"Steady plot   : {steady_plot}")
    for path in csv_paths:
        print(f"CSV           : {path}")

    print(
        "\nPASS: all surface cases completed without NaN/Inf.\n"
        "Interpretation target: explain why the same PI gains produce different "
        "transients and penetration on different contact dynamics."
    )

    if args.viewer:
        print(
            f"\nViewer surface: {args.viewer_surface} "
            f"(timeconst={SURFACES[args.viewer_surface]:.3f}s)"
        )
        show_viewer(scenes[args.viewer_surface])


if __name__ == "__main__":
    main()
