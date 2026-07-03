from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_configs
from app.isaac_render_video import (
    _create_render_camera,
    _look_at_quat_wxyz,
    _set_camera_pose,
    _set_viewport_camera,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the real Isaac Sim lunar landing scene for screen recording")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--renderer", default="RayTracedLighting")
    parser.add_argument("--experience", default="/isaac-sim/apps/isaacsim.exp.full.kit")
    parser.add_argument("--terrain-resolution", type=int, default=64)
    parser.add_argument("--landing-resolution", type=int, default=512)
    parser.add_argument("--landing-size", type=float, default=24.0)
    parser.add_argument("--simulate-physics", action="store_true")
    return parser.parse_args()


def _freeze_rocket_for_demo(stage, prim_path: str, rigid_body_path: str) -> None:
    try:
        from pxr import Sdf

        for path in (rigid_body_path, prim_path):
            prim = stage.GetPrimAtPath(path)
            if prim:
                prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(True)
                prim.CreateAttribute("physxRigidBody:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        print("[LunarRocket] demo mode: rocket physics frozen for stable presentation", flush=True)
    except Exception as exc:
        print(f"[LunarRocket] could not freeze rocket physics attrs: {type(exc).__name__}: {exc}", flush=True)


def _wait_for_gui(simulation_app, timeout_s: float = 210.0):
    import omni.kit.viewport.utility as vu

    deadline = time.monotonic() + timeout_s
    last_report = 0.0
    viewport = None
    while time.monotonic() < deadline:
        simulation_app.update()
        viewport = vu.get_active_viewport()
        if viewport is not None:
            print("[LunarRocket] Isaac GUI viewport is ready", flush=True)
            return viewport
        now = time.monotonic()
        if now - last_report > 10.0:
            print("[LunarRocket] waiting for Isaac GUI viewport...", flush=True)
            last_report = now
        time.sleep(0.05)
    raise TimeoutError("Timed out waiting for Isaac GUI viewport/default window")


def main() -> None:
    args = parse_args()
    configs = load_configs(args.config_dir)
    configs["sim"]["headless"] = False
    configs["sim"]["renderer"] = args.renderer
    configs["sim"]["window_width"] = int(args.width)
    configs["sim"]["window_height"] = int(args.height)
    configs["sim"]["disable_viewport_updates"] = False
    configs["sim"]["render_on_step"] = True
    configs["sim"].setdefault("camera", {})["enabled"] = False
    if int(args.terrain_resolution) >= 256:
        configs["terrain"]["terrain_quality"] = "high_train"
    elif int(args.terrain_resolution) >= 192:
        configs["terrain"]["terrain_quality"] = "medium_train"
    else:
        configs["terrain"]["terrain_quality"] = "low_debug"
    configs["terrain"]["dem_vertical_scale"] = min(float(configs["terrain"].get("dem_vertical_scale", 0.001)), 0.00030)
    configs["terrain"]["dem_max_relief_m"] = 3.8
    configs["terrain"]["dem_feature_gain"] = 0.95
    configs["terrain"]["dem_base_smoothing_passes"] = 4
    configs["terrain"]["dem_final_smoothing_passes"] = 1
    configs["terrain"]["crater_depth_m_range"] = [0.06, 0.38]
    configs["terrain"]["crater_depth_to_radius_ratio"] = [0.06, 0.17]
    configs["terrain"]["crater_rim_height_ratio"] = [0.02, 0.06]
    configs["terrain"]["roughness_scale_range"] = [0.018, 0.050]
    configs["terrain"]["roughness_smoothing_passes"] = 2
    configs["terrain"]["base_slope_deg_range"] = [0.0, 3.5]
    configs["terrain"]["low_frequency_noise_scale"] = 0.16
    configs["terrain"]["low_frequency_noise_smoothing_passes"] = 10
    configs["terrain"].setdefault("local_detail_patch", {})["enabled"] = True
    min_landing_size = max(10.24, float(configs["rocket"].get("visual_scale", 1.0)) * 1.8)
    landing_size = max(float(args.landing_size), min_landing_size)
    if landing_size > float(args.landing_size):
        print(
            "[LunarRocket] landing patch enlarged for scaled rocket "
            f"requested={float(args.landing_size):.2f}m actual={landing_size:.2f}m",
            flush=True,
        )
    configs["terrain"]["local_detail_patch"]["resolution"] = int(args.landing_resolution)
    configs["terrain"]["local_detail_patch"]["patch_size_m"] = [landing_size, landing_size]
    configs["terrain"]["local_detail_patch"]["noise_scale"] = 0.022
    configs["terrain"]["local_detail_patch"]["noise_smoothing_passes"] = 1
    configs["terrain"]["local_detail_patch"]["regolith_grain_scale"] = 0.012
    configs["terrain"]["local_detail_patch"]["regolith_grain_smoothing_passes"] = 1
    configs["terrain"]["local_detail_patch"]["regolith_ripple_scale"] = 0.010
    configs["terrain"]["local_detail_patch"]["regolith_ripple_wavelength_m"] = [0.18, 0.42]
    configs["terrain"]["local_detail_patch"]["micro_feature_count"] = 650
    configs["terrain"]["local_detail_patch"]["micro_feature_radius_m_range"] = [0.030, 0.130]
    configs["terrain"]["local_detail_patch"]["micro_feature_height_m_range"] = [-0.030, 0.060]
    configs["terrain"]["local_detail_patch"]["micro_feature_sharpness"] = 0.70
    configs["terrain"]["local_detail_patch"]["micro_pit_probability"] = 0.62
    configs["terrain"]["local_detail_patch"]["micro_pit_rim_ratio"] = 0.28
    configs["terrain"]["local_detail_patch"]["footprint_detail_radius_m"] = min(landing_size * 0.42, 4.8)
    configs["terrain"]["local_detail_patch"]["footprint_micro_feature_count"] = 360
    configs["terrain"]["local_detail_patch"]["footprint_micro_feature_radius_m_range"] = [0.025, 0.100]
    configs["terrain"]["local_detail_patch"]["footprint_micro_feature_height_m_range"] = [-0.025, 0.055]
    configs["terrain"]["local_detail_patch"]["smooth_normals"] = False
    configs["terrain"]["local_detail_patch"]["max_slope_deg"] = 38.0
    configs["terrain"]["local_detail_patch"]["slope_limit_passes"] = 8

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": False,
            "renderer": args.renderer,
            "width": int(args.width),
            "height": int(args.height),
            "window_width": int(args.width),
            "window_height": int(args.height),
            "disable_viewport_updates": False,
        },
        experience=str(args.experience),
    )

    try:
        viewport = _wait_for_gui(simulation_app)
        from app.world_builder import LunarLandingWorld

        env = LunarLandingWorld(configs)
        print("[LunarRocket] building live Isaac scene", flush=True)
        env.build()
        if env.state is None:
            raise RuntimeError("Environment did not produce a reset state")
        if not bool(args.simulate_physics):
            _freeze_rocket_for_demo(env.stage, env.rocket.prim_path, env.rocket.rigid_body_path)

        print("[LunarRocket] creating live camera", flush=True)
        camera_path = _create_render_camera(env.stage, args.width, args.height)
        try:
            import omni.kit.viewport.utility as vu

            if viewport is None:
                window = vu.create_viewport_window("LunarRocketLive", width=int(args.width), height=int(args.height))
                viewport = getattr(window, "viewport_api", None) or vu.get_active_viewport()
            if viewport is not None:
                _set_viewport_camera(viewport, camera_path)
        except Exception as exc:
            print(f"[LunarRocket] could not bind live viewport camera: {type(exc).__name__}: {exc}", flush=True)
        rocket_start = np.array(env.state.rocket.position, dtype=float)
        target = rocket_start + np.array([0.0, 0.0, 0.65], dtype=float)
        print("[LunarRocket] live Isaac scene ready for screen recording", flush=True)
        total_steps = max(1, int(float(args.duration) * int(args.fps)))
        for idx in range(total_steps):
            phase = idx / max(1, total_steps - 1)
            angle = np.deg2rad(225.0 + 38.0 * phase)
            radius = 8.0 - 1.0 * phase
            height = 4.2 - 0.6 * phase
            camera_position = target + np.array(
                [radius * np.cos(angle), radius * np.sin(angle), height],
                dtype=float,
            )
            _set_camera_pose(env.stage, camera_path, camera_position, np.array(_look_at_quat_wxyz(camera_position, target)))
            if bool(args.simulate_physics):
                env.step(throttle=0.18 if phase < 0.75 else 0.0)
            else:
                env.rocket.reset_pose(env.state.rocket.position, env.state.rocket.euler_deg, log=False)
            simulation_app.update()
            time.sleep(1.0 / max(1, int(args.fps)))
        print("[LunarRocket] live Isaac demo complete", flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
