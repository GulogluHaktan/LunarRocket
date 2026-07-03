from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.config import load_configs, resolve_project_path
from app.randomization import sample_reset
from app.terrain_generator import MoonTerrainGenerator

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ModuleNotFoundError as exc:
    missing = exc.name or "matplotlib"
    raise SystemExit(
        f"Missing presentation video dependency: {missing}\n"
        "Install local video dependencies with:\n"
        "  python3 -m pip install -r requirements-presentation.txt\n"
        "Then rerun:\n"
        "  python3 app/presentation_video.py --frames 120 --fps 24 --dpi 100 --output outputs/videos/lunar_landing_demo.mp4"
    ) from exc


def _floats(value: str | None, default: Iterable[float]) -> np.ndarray:
    if value is None:
        return np.array(list(default), dtype=float)
    return np.array([float(part) for part in value.split()], dtype=float)


def _rotation_xyz(deg: np.ndarray) -> np.ndarray:
    rx, ry, rz = np.deg2rad(deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def _surface_at(heights: np.ndarray, size: tuple[float, float], x: float, y: float) -> float:
    sx, sy = size
    row = int(np.clip((y + sy / 2.0) / sy * (heights.shape[0] - 1), 0, heights.shape[0] - 1))
    col = int(np.clip((x + sx / 2.0) / sx * (heights.shape[1] - 1), 0, heights.shape[1] - 1))
    return float(heights[row, col])


def _parse_mjcf_geoms(xml_path: Path) -> list[dict[str, np.ndarray | str]]:
    tree = ET.parse(xml_path)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        return []

    geoms: list[dict[str, np.ndarray | str]] = []

    def walk(body: ET.Element, parent_pos: np.ndarray) -> None:
        body_pos = parent_pos + _floats(body.attrib.get("pos"), [0, 0, 0])
        for geom in body.findall("geom"):
            item: dict[str, np.ndarray | str] = {
                "type": geom.attrib.get("type", "sphere"),
                "name": geom.attrib.get("name", ""),
                "pos": body_pos + _floats(geom.attrib.get("pos"), [0, 0, 0]),
                "size": _floats(geom.attrib.get("size"), [0.03]),
                "material": geom.attrib.get("material", ""),
            }
            if "fromto" in geom.attrib:
                ft = _floats(geom.attrib["fromto"], [0, 0, 0, 0, 0, 1])
                item["from"] = body_pos + ft[:3]
                item["to"] = body_pos + ft[3:]
            geoms.append(item)
        for child in body.findall("body"):
            walk(child, body_pos)

    for body in worldbody.findall("body"):
        walk(body, np.zeros(3))
    return geoms


def _basis_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = float(np.linalg.norm(axis))
    if length == 0.0:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = axis / length
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return axis, u, v


def _add_cylinder(ax, start: np.ndarray, end: np.ndarray, radius: float, color: str, alpha: float = 1.0) -> None:
    axis, u, v = _basis_from_axis(end - start)
    length = float(np.linalg.norm(end - start))
    theta = np.linspace(0, 2 * np.pi, 20)
    z = np.linspace(0, length, 2)
    tt, zz = np.meshgrid(theta, z)
    points = start[:, None, None] + axis[:, None, None] * zz + radius * (
        u[:, None, None] * np.cos(tt) + v[:, None, None] * np.sin(tt)
    )
    ax.plot_surface(points[0], points[1], points[2], color=color, linewidth=0, antialiased=True, alpha=alpha, shade=True)


def _add_sphere(ax, center: np.ndarray, radius: float, color: str) -> None:
    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi, 10)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, linewidth=0, shade=True)


def _add_flame(ax, origin: np.ndarray, rotation: np.ndarray, throttle: float) -> None:
    length = 0.45 + 0.35 * throttle
    radius = 0.09
    tip = origin + rotation @ np.array([0.0, 0.0, -length])
    theta = np.linspace(0, 2 * np.pi, 24)
    ring = [origin + rotation @ np.array([radius * np.cos(t), radius * np.sin(t), -0.03]) for t in theta]
    faces = [[ring[i], ring[(i + 1) % len(ring)], tip] for i in range(len(ring))]
    flame = Poly3DCollection(faces, facecolor=(1.0, 0.45, 0.08, 0.6), edgecolor="none")
    ax.add_collection3d(flame)


def _draw_rocket(
    ax,
    geoms: list[dict[str, np.ndarray | str]],
    position: np.ndarray,
    euler_deg: np.ndarray,
    throttle: float,
    model_scale: float = 7.0,
) -> None:
    rotation = _rotation_xyz(euler_deg)
    colors = {
        "body_mat": "#c8c8c8",
        "leg_mat": "#1f2428",
        "foot_mat": "#e3c63a",
        "engine_mat": "#c43b32",
    }
    for geom in geoms:
        material = str(geom.get("material", ""))
        color = colors.get(material, "#d8d8d8")
        geom_type = str(geom["type"])
        size = np.asarray(geom["size"], dtype=float)
        if "from" in geom and "to" in geom:
            start = position + rotation @ (np.asarray(geom["from"], dtype=float) * model_scale)
            end = position + rotation @ (np.asarray(geom["to"], dtype=float) * model_scale)
            _add_cylinder(ax, start, end, float(size[0]) * model_scale, color)
        elif geom_type == "cylinder":
            center = position + rotation @ (np.asarray(geom["pos"], dtype=float) * model_scale)
            radius = float(size[0]) * model_scale
            half_len = (float(size[1]) if len(size) > 1 else float(size[0])) * model_scale
            start = center + rotation @ np.array([0.0, 0.0, -half_len])
            end = center + rotation @ np.array([0.0, 0.0, half_len])
            _add_cylinder(ax, start, end, radius, color)
        elif geom_type == "sphere":
            center = position + rotation @ (np.asarray(geom["pos"], dtype=float) * model_scale)
            _add_sphere(ax, center, float(size[0]) * model_scale, color)

    engine_origin = position + rotation @ (np.array([0.0, 0.0, 0.355 - 0.25]) * model_scale)
    _add_flame(ax, engine_origin, rotation, throttle)


def render_video(output: Path, frames: int, fps: int, dpi: int, config_dir: Path) -> None:
    configs = load_configs(config_dir)
    terrain_config = dict(configs["terrain"])
    terrain_config["mesh_resolution"] = max(140, int(terrain_config.get("mesh_resolution", 128)))
    terrain_config["terrain_size_m"] = [34.0, 34.0]

    rng = np.random.default_rng(int(configs["sim"].get("seed", 1234)))
    sample = sample_reset(terrain_config, configs["rocket"], rng)
    terrain = MoonTerrainGenerator(terrain_config).generate(sample)
    geoms = _parse_mjcf_geoms(resolve_project_path(configs["rocket"]["mjcf_path"]))

    heights = terrain.heights
    heights = (heights - float(np.median(heights))) * 0.28
    heights = np.clip(heights, -2.0, 2.0)
    sx, sy = terrain.size_m
    x = np.linspace(-sx / 2.0, sx / 2.0, heights.shape[1])
    y = np.linspace(-sy / 2.0, sy / 2.0, heights.shape[0])
    xx, yy = np.meshgrid(x, y)

    surface = _surface_at(heights, terrain.size_m, 0.0, 0.0)
    start_z = surface + 6.4
    end_z = surface + 3.2

    frame_dir = output.parent / "_frames_lunar_landing"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("frame_*.png"):
        old.unlink()

    for idx in range(frames):
        t = idx / max(1, frames - 1)
        ease = 1 - (1 - t) ** 2
        rocket_pos = np.array([0.25 * np.sin(t * np.pi), -6.9 + 0.2 * np.cos(t * np.pi), start_z * (1 - ease) + end_z * ease])
        rocket_att = np.array([3.0 * np.sin(t * np.pi * 1.7), 2.0 * np.cos(t * np.pi * 1.2), 14.0 * np.sin(t * np.pi)])
        throttle = max(0.18, 1.0 - 0.55 * t)

        fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#050505")
        fig.patch.set_facecolor("#050505")
        ax.plot_surface(
            xx,
            yy,
            heights,
            cmap="gray",
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=True,
            alpha=0.68,
        )
        _draw_rocket(ax, geoms, rocket_pos, rocket_att, throttle)

        ax.view_init(elev=18 + 4 * np.sin(t * np.pi), azim=205 + 50 * t)
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(-9.0, -1.2)
        ax.set_zlim(surface - 1.0, surface + 8.2)
        ax.set_box_aspect((1.0, 1.0, 0.65))
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(frame_dir / f"frame_{idx:04d}.png", facecolor=fig.get_facecolor())
        plt.close(fig)
        if idx % 20 == 0:
            print(f"[LunarRocket] rendered frame {idx + 1}/{frames}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    print(f"[LunarRocket] presentation video written: {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a local 3D presentation video from the lunar environment assets")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output", default="outputs/videos/lunar_landing_demo.mp4")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dpi", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_video(resolve_project_path(args.output), args.frames, args.fps, args.dpi, resolve_project_path(args.config_dir))


if __name__ == "__main__":
    main()
