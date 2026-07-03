from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_configs, resolve_project_path
from app.randomization import sample_reset
from app.terrain_generator import MoonTerrainGenerator


def _sphere(center: tuple[float, float, float], radius: float, samples: int = 18) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, samples)
    v = np.linspace(0.0, np.pi, samples)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _cylinder_z(
    center: tuple[float, float, float],
    radius: float,
    z_min: float,
    z_max: float,
    samples: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, samples)
    z = np.array([z_min, z_max], dtype=float)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x = center[0] + radius * np.cos(theta_grid)
    y = center[1] + radius * np.sin(theta_grid)
    return x, y, z_grid


def _cone_z(
    center: tuple[float, float, float],
    radius: float,
    z_base: float,
    z_tip: float,
    samples: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, samples)
    levels = np.array([0.0, 1.0], dtype=float)
    theta_grid, level_grid = np.meshgrid(theta, levels)
    r = radius * (1.0 - level_grid)
    x = center[0] + r * np.cos(theta_grid)
    y = center[1] + r * np.sin(theta_grid)
    z = z_base + (z_tip - z_base) * level_grid
    return x, y, z


def _terrain_grid(mesh_data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = mesh_data.heights.shape
    size_x, size_y = mesh_data.size_m
    x = np.linspace(-size_x / 2.0, size_x / 2.0, cols)
    y = np.linspace(-size_y / 2.0, size_y / 2.0, rows)
    xx, yy = np.meshgrid(x, y)
    return xx, yy, mesh_data.heights


def _height_at(mesh_data, x: float, y: float) -> float:
    size_x, size_y = mesh_data.size_m
    row = int(np.clip((y + size_y / 2.0) / size_y * (mesh_data.heights.shape[0] - 1), 0, mesh_data.heights.shape[0] - 1))
    col = int(np.clip((x + size_x / 2.0) / size_x * (mesh_data.heights.shape[1] - 1), 0, mesh_data.heights.shape[1] - 1))
    return float(mesh_data.heights[row, col])


def _draw_hopper(ax, base_z: float, scale: float, x: float, y: float) -> None:
    cx, cy = float(x), float(y)
    z0 = base_z + 0.12 * scale
    body_bottom = z0 + 0.00 * scale
    body_top = z0 + 0.72 * scale
    radius = 0.105 * scale

    ax.plot_surface(*_cylinder_z((cx, cy, 0.0), radius, body_bottom, body_top), color="#c7c7bf", linewidth=0, shade=True, zorder=10)
    ax.plot_surface(*_cone_z((cx, cy, 0.0), radius * 1.05, body_top, body_top + 0.22 * scale), color="#f0f0e8", linewidth=0, shade=True, zorder=10)
    ax.plot_surface(*_cylinder_z((cx, cy, 0.0), radius * 0.72, body_bottom - 0.09 * scale, body_bottom + 0.02 * scale), color="#8f3028", linewidth=0, shade=True, zorder=10)

    leg_targets = [(0.72, 0.0), (-0.72, 0.0), (0.0, 0.72), (0.0, -0.72)]
    for lx, ly in leg_targets:
        foot = (cx + lx * scale, cy + ly * scale, base_z + 0.02 * scale)
        hip = (cx, cy, body_bottom + 0.20 * scale)
        ax.plot([hip[0], foot[0]], [hip[1], foot[1]], [hip[2], foot[2]], color="#202020", linewidth=4.0)
        ax.plot_surface(*_sphere(foot, 0.045 * scale), color="#e6d958", linewidth=0, shade=True)

    flame_tip = body_bottom - 0.48 * scale
    ax.plot_surface(*_cone_z((cx, cy, 0.0), radius * 0.48, body_bottom - 0.08 * scale, flame_tip), color="#ff8128", linewidth=0, shade=True, alpha=0.86, zorder=10)
    ax.plot_surface(*_cone_z((cx, cy, 0.0), radius * 0.26, body_bottom - 0.08 * scale, flame_tip + 0.14 * scale), color="#ffd66a", linewidth=0, shade=True, alpha=0.95, zorder=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a still 3D presentation image from NASA DEM terrain and the MuJoCo hopper geometry")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output", default="outputs/preview/lunar_landing_still.png")
    parser.add_argument("--resolution", type=int, default=180)
    parser.add_argument("--terrain-size", type=float, default=18.0)
    parser.add_argument("--dem-scale", type=float, default=0.00008)
    parser.add_argument("--rocket-scale", type=float, default=2.3)
    parser.add_argument("--rocket-x", type=float, default=-2.2)
    parser.add_argument("--rocket-y", type=float, default=-3.8)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = load_configs(args.config_dir)
    terrain_config = dict(configs["terrain"])
    terrain_config["seed"] = int(args.seed)
    terrain_config["mesh_resolution"] = int(args.resolution)
    terrain_config["terrain_size_m"] = [float(args.terrain_size), float(args.terrain_size)]
    terrain_config["dem_vertical_scale"] = float(args.dem_scale)
    terrain_config["crater_radius_m_range"] = [0.35, 2.2]
    terrain_config["crater_depth_m_range"] = [0.05, 0.55]
    terrain_config["roughness_scale_range"] = [0.03, 0.16]
    terrain_config["slope_x_range"] = [-0.035, 0.035]
    terrain_config["slope_y_range"] = [-0.035, 0.035]

    rng = np.random.default_rng(int(args.seed))
    reset = sample_reset(terrain_config, configs["rocket"], rng)
    generator = MoonTerrainGenerator(terrain_config)
    terrain = generator.generate(reset)
    xx, yy, zz = _terrain_grid(terrain)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9), dpi=int(args.dpi), facecolor="#020202")
    ax = fig.add_subplot(111, projection="3d", facecolor="#020202")
    light = LightSource(azdeg=315, altdeg=28)
    shaded = light.shade(zz, cmap=plt.get_cmap("gray"), vert_exag=0.9, blend_mode="soft")
    ax.plot_surface(xx, yy, zz, facecolors=shaded, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

    base_z = _height_at(terrain, float(args.rocket_x), float(args.rocket_y))
    _draw_hopper(ax, base_z, float(args.rocket_scale), float(args.rocket_x), float(args.rocket_y))

    span = float(args.terrain_size) / 2.0
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_zlim(float(np.percentile(zz, 1)) - 0.4, base_z + 1.3 * float(args.rocket_scale))
    ax.view_init(elev=19, azim=-56, roll=0)
    ax.set_proj_type("persp", focal_length=0.55)
    ax.set_axis_off()
    ax.set_box_aspect((1.6, 1.0, 0.45))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[LunarRocket] still render written: {output}", flush=True)


if __name__ == "__main__":
    main()
