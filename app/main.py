from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Sim lunar landing environment")
    parser.add_argument("--config-dir", default="configs", help="Directory containing sim/terrain/rocket YAML configs")
    parser.add_argument("--headless", action="store_true", help="Force Isaac Sim headless mode")
    parser.add_argument("--renderer", default=None, help="Override Isaac renderer, for example RayTracedLighting or None")
    parser.add_argument("--disable-camera", action="store_true", help="Skip RGB camera creation for low-VRAM smoke tests")
    parser.add_argument("--no-render-step", action="store_true", help="Step physics without render updates")
    parser.add_argument("--save-stage", default=None, help="Export the built 3D USD stage to this path")
    parser.add_argument("--steps", type=int, default=0, help="Number of simulation steps to run after reset")
    parser.add_argument("--throttle", type=float, default=0.0, help="Placeholder throttle applied during steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = load_configs(Path(args.config_dir))
    sim_config = configs["sim"]
    headless = bool(args.headless or sim_config.get("headless", False))
    if args.renderer is not None:
        sim_config["renderer"] = args.renderer
    if args.disable_camera:
        sim_config.setdefault("camera", {})["enabled"] = False
    if args.no_render_step:
        sim_config["render_on_step"] = False

    print(
        "[LunarRocket] config loaded "
        f"headless={headless} renderer={sim_config.get('renderer')} "
        f"camera_enabled={sim_config.get('camera', {}).get('enabled', True)} "
        f"steps={args.steps}",
        flush=True,
    )

    try:
        from isaacsim import SimulationApp
    except ImportError:
        try:
            from omni.isaac.kit import SimulationApp
        except ImportError as exc:
            raise RuntimeError("Run this entrypoint with Isaac Sim Python, not the system Python") from exc

    simulation_app = SimulationApp(
        {
            "headless": headless,
            "renderer": sim_config.get("renderer", "RayTracedLighting"),
            "width": int(sim_config.get("window_width", 1280)),
            "height": int(sim_config.get("window_height", 720)),
            "disable_viewport_updates": bool(sim_config.get("disable_viewport_updates", headless)),
        }
    )
    print("[LunarRocket] SimulationApp created", flush=True)

    try:
        from app.world_builder import LunarLandingWorld

        env = LunarLandingWorld(configs)
        print("[LunarRocket] building world", flush=True)
        try:
            env.build()
        except BaseException as exc:
            print(f"[LunarRocket] world build failed: {type(exc).__name__}: {exc!r}", flush=True)
            raise
        if env.state is not None:
            print(
                "[LunarRocket] reset complete "
                f"rocket={env.state.rocket.position} "
                f"terrain_min={env.state.terrain.min_height_m:.3f} "
                f"terrain_max={env.state.terrain.max_height_m:.3f}",
                flush=True,
            )
        if args.save_stage:
            env.save_stage(args.save_stage)
        for step_idx in range(max(0, int(args.steps))):
            env.step(throttle=args.throttle)
            if step_idx == 0 or step_idx == int(args.steps) - 1:
                print(f"[LunarRocket] step {step_idx + 1}/{args.steps}", flush=True)
        print("[LunarRocket] run complete", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
