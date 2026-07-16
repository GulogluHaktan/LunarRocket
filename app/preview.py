from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_configs, resolve_project_path
from app.randomization import sample_reset
from app.terrain_generator import MoonTerrainGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a presentation preview of the LunarRocket environment")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output", default="outputs/preview/lunar_landing_preview.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=1400)
    parser.add_argument("--mesh-resolution", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = load_configs(args.config_dir)
    rng = np.random.default_rng(args.seed)

    terrain_config = dict(configs["terrain"])
    terrain_config["mesh_resolution"] = args.mesh_resolution
    rocket_config = configs["rocket"]
    reset = sample_reset(terrain_config, rocket_config, rng)
    terrain = MoonTerrainGenerator(terrain_config).generate(reset)

    image = _compose_preview(terrain.heights, terrain.size_m, reset.rocket_position, reset.target_position, args.size)
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    clean_path = output_path.with_name(output_path.stem + "_clean" + output_path.suffix)
    _compose_clean_preview(terrain.heights, terrain.size_m, reset.rocket_position, reset.target_position, args.size).save(clean_path)
    print(f"[LunarRocket] preview saved: {output_path}")
    print(f"[LunarRocket] clean preview saved: {clean_path}")


def _compose_preview(
    heights: np.ndarray,
    terrain_size_m: tuple[float, float],
    rocket_position: tuple[float, float, float],
    target_position: tuple[float, float, float],
    size: int,
) -> Image.Image:
    title_h = 96
    footer_h = 120
    canvas = Image.new("RGB", (size, size + title_h + footer_h), (12, 13, 12))
    draw = ImageDraw.Draw(canvas)

    shaded = _shaded_relief(heights, contrast=0.92)
    terrain_image = Image.fromarray(shaded, mode="RGB").resize((size, size), Image.Resampling.LANCZOS)
    canvas.paste(terrain_image, (0, title_h))

    font_title = _font(40)
    font_body = _font(24)
    font_small = _font(18)

    draw.rectangle((0, 0, size, title_h), fill=(14, 15, 14))
    draw.text((34, 24), "LunarRocket Isaac Sim Environment", fill=(236, 235, 228), font=font_title)
    draw.text((34, 68), "NASA DEM terrain + procedural craters + MuJoCo hopper rocket", fill=(165, 166, 158), font=font_small)

    # 1. Draw target landing pad (crosshair)
    target_px = _world_to_pixel(target_position[0], target_position[1], terrain_size_m, size, title_h)
    _draw_landing_target(draw, target_px, scale=max(1.0, size / 1100.0))

    # 2. Draw rocket spawn position
    rocket_px = _world_to_pixel(rocket_position[0], rocket_position[1], terrain_size_m, size, title_h)
    _draw_hopper_rocket(draw, rocket_px, scale=max(1.0, size / 1100.0))

    scalebar_y = title_h + size - 54
    scalebar_w = int(size * 0.25)
    draw.line((34, scalebar_y, 34 + scalebar_w, scalebar_y), fill=(242, 241, 231), width=5)
    draw.text((34, scalebar_y + 10), f"{terrain_size_m[0] * 0.25:.0f} m", fill=(242, 241, 231), font=font_small)

    footer_y = title_h + size
    draw.rectangle((0, footer_y, size, footer_y + footer_h), fill=(14, 15, 14))
    stats = (
        f"terrain: {terrain_size_m[0]:.0f}m x {terrain_size_m[1]:.0f}m   "
        f"elevation: {float(np.min(heights)):.2f}m to {float(np.max(heights)):.2f}m   "
        f"target: x={target_position[0]:.2f}m y={target_position[1]:.2f}m   "
        f"rocket spawn: x={rocket_position[0]:.2f}m y={rocket_position[1]:.2f}m"
    )
    draw.text((34, footer_y + 28), stats, fill=(227, 226, 218), font=font_body)
    draw.text(
        (34, footer_y + 68),
        "Low-VRAM preview generated from the same reset/terrain pipeline used by the Isaac Sim runtime.",
        fill=(156, 158, 151),
        font=font_small,
    )
    return canvas


def _compose_clean_preview(
    heights: np.ndarray,
    terrain_size_m: tuple[float, float],
    rocket_position: tuple[float, float, float],
    target_position: tuple[float, float, float],
    size: int,
) -> Image.Image:
    shaded = _shaded_relief(heights, contrast=1.0)
    canvas = Image.fromarray(shaded, mode="RGB").resize((size, size), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    
    # 1. Draw target crosshair
    target_px = _world_to_pixel(target_position[0], target_position[1], terrain_size_m, size, 0)
    _draw_landing_target(draw, target_px, scale=max(1.2, size / 950.0))
    
    # 2. Draw rocket
    rocket_px = _world_to_pixel(rocket_position[0], rocket_position[1], terrain_size_m, size, 0)
    _draw_hopper_rocket(draw, rocket_px, scale=max(1.2, size / 950.0))
    return canvas


def _shaded_relief(heights: np.ndarray, contrast: float = 1.0) -> np.ndarray:
    normalized = heights - float(np.min(heights))
    span = float(np.max(normalized))
    if span > 0.0:
        normalized /= span

    smooth = _smooth(heights.astype(np.float32), passes=1)
    dy, dx = np.gradient(smooth)
    normal = np.dstack((-dx * 1.25, -dy * 1.25, np.ones_like(heights) * 2.1))
    normal /= np.linalg.norm(normal, axis=2, keepdims=True) + 1e-6
    light = np.array([-0.45, -0.35, 0.82], dtype=np.float32)
    light /= np.linalg.norm(light)
    shade = np.clip(np.sum(normal * light, axis=2), 0.0, 1.0) ** contrast

    base = np.array([60, 59, 55], dtype=np.float32)
    high = np.array([206, 202, 190], dtype=np.float32)
    color = base + (high - base) * normalized[..., None]
    color *= (0.36 + shade[..., None] * 0.82)
    color = _add_vignette(color)
    return np.clip(color, 0, 255).astype(np.uint8)


def _world_to_pixel(
    x: float,
    y: float,
    terrain_size_m: tuple[float, float],
    image_size: int,
    y_offset: int,
) -> tuple[int, int]:
    px = int((x / terrain_size_m[0] + 0.5) * image_size)
    py = int((0.5 - y / terrain_size_m[1]) * image_size) + y_offset
    return px, py


def _draw_hopper_rocket(draw: ImageDraw.ImageDraw, center: tuple[int, int], scale: float = 1.0) -> None:
    x, y = center
    s = scale
    shadow = (int(x - 54 * s), int(y + 38 * s), int(x + 64 * s), int(y + 72 * s))
    draw.ellipse(shadow, fill=(20, 20, 18))

    leg = (48, 49, 48)
    leg_w = max(3, int(4 * s))
    for dx in (-34, 34):
        draw.line((x, y + int(36 * s), x + int(dx * s), y + int(72 * s)), fill=leg, width=leg_w)
        draw.ellipse(
            (
                x + int(dx * s) - int(10 * s),
                y + int(70 * s) - int(5 * s),
                x + int(dx * s) + int(10 * s),
                y + int(70 * s) + int(5 * s),
            ),
            fill=(230, 214, 126),
        )

    body_w = int(36 * s)
    body_h = int(104 * s)
    body = (x - body_w // 2, y - body_h // 2, x + body_w // 2, y + body_h // 2)
    draw.rounded_rectangle(body, radius=max(8, int(10 * s)), fill=(214, 216, 208), outline=(44, 45, 43), width=max(2, int(3 * s)))
    draw.rectangle((x - body_w // 2 + int(5 * s), y - body_h // 2 + int(8 * s), x - int(1 * s), y + body_h // 2), fill=(176, 179, 172))
    draw.polygon(
        [
            (x - body_w // 2, y - body_h // 2 + int(8 * s)),
            (x, y - body_h // 2 - int(28 * s)),
            (x + body_w // 2, y - body_h // 2 + int(8 * s)),
        ],
        fill=(236, 238, 230),
        outline=(44, 45, 43),
    )

    engine = (
        x - int(18 * s),
        y + body_h // 2 - int(6 * s),
        x + int(18 * s),
        y + body_h // 2 + int(16 * s),
    )
    draw.ellipse(engine, fill=(126, 44, 42), outline=(42, 32, 30), width=max(2, int(2 * s)))
    draw.polygon(
        [
            (x - int(14 * s), y + body_h // 2 + int(12 * s)),
            (x, y + body_h // 2 + int(54 * s)),
            (x + int(14 * s), y + body_h // 2 + int(12 * s)),
        ],
        fill=(238, 147, 64),
    )
    draw.polygon(
        [
            (x - int(7 * s), y + body_h // 2 + int(12 * s)),
            (x, y + body_h // 2 + int(38 * s)),
            (x + int(7 * s), y + body_h // 2 + int(12 * s)),
        ],
        fill=(255, 222, 123),
    )

    ring_r = int(78 * s)
    draw.ellipse((x - ring_r, y - ring_r, x + ring_r, y + ring_r), outline=(255, 214, 94), width=max(3, int(4 * s)))
    draw.text((x + int(82 * s), y - int(46 * s)), "MuJoCo hopper", fill=(255, 214, 94), font=_font(max(18, int(20 * s))))


def _smooth(values: np.ndarray, passes: int) -> np.ndarray:
    result = values.copy()
    for _ in range(passes):
        result = (
            result
            + np.roll(result, 1, axis=0)
            + np.roll(result, -1, axis=0)
            + np.roll(result, 1, axis=1)
            + np.roll(result, -1, axis=1)
        ) / 5.0
    return result


def _add_vignette(color: np.ndarray) -> np.ndarray:
    h, w, _ = color.shape
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    vignette = np.clip(1.08 - 0.30 * np.sqrt(xx**2 + yy**2), 0.74, 1.0)
    return color * vignette[..., None]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_landing_target(draw: ImageDraw.ImageDraw, center: tuple[int, int], scale: float = 1.0) -> None:
    x, y = center
    s = scale
    color = (46, 204, 113) # Nice green
    width = max(2, int(3 * s))
    
    # Outer circle
    r_outer = int(24 * s)
    draw.ellipse((x - r_outer, y - r_outer, x + r_outer, y + r_outer), outline=color, width=width)
    
    # Inner circle
    r_inner = int(12 * s)
    draw.ellipse((x - r_inner, y - r_inner, x + r_inner, y + r_inner), outline=color, width=width)
    
    # Crosshair lines
    length = int(32 * s)
    draw.line((x - length, y, x + length, y), fill=color, width=width)
    draw.line((x, y - length, x, y + length), fill=color, width=width)


if __name__ == "__main__":
    main()
