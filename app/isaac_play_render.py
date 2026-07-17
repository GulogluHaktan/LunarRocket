from __future__ import annotations

import argparse
import inspect
import sys
import os
from pathlib import Path
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_configs, resolve_project_path
from app.isaac_render_video import (
    _create_render_camera,
    _look_at_quat_wxyz,
    _set_camera_pose,
    _set_viewport_camera,
    _capture_viewport_frame,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture an MP4 video of the trained SAC policy in Isaac Sim")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--model-path", default="outputs/models/sac_lunar_landing.zip")
    parser.add_argument("--output", default="outputs/videos/trained_lunar_landing.mp4")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--renderer", default="RayTracedLighting")
    parser.add_argument("--terrain-resolution", type=int, default=96)
    parser.add_argument("--no-headless", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    model_path = resolve_project_path(args.model_path)
    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output.parent / "_frames_isaac_trained"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("frame_*.png"):
        old_frame.unlink()

    configs = load_configs(args.config_dir)
    headless = not bool(args.no_headless)
    configs["sim"]["headless"] = headless
    configs["sim"]["renderer"] = args.renderer
    configs["sim"]["window_width"] = int(args.width)
    configs["sim"]["window_height"] = int(args.height)
    configs["sim"]["disable_viewport_updates"] = False
    configs["sim"]["render_on_step"] = True
    configs["sim"].setdefault("camera", {})["enabled"] = False
    configs["terrain"]["mesh_resolution"] = int(args.terrain_resolution)

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": headless,
            "renderer": args.renderer,
            "width": int(args.width),
            "height": int(args.height),
            "disable_viewport_updates": False,
        }
    )

    try:
        from stable_baselines3 import SAC
        from app.world_builder import LunarLandingWorld
        from app.world_adapter import IsaacWorldAdapter
        from app.env_lunar_landing import LunarLandingIsaacEnv

        print("[LunarRocket] loading trained SAC policy", flush=True)
        world = LunarLandingWorld(configs)
        print("[LunarRocket] building Isaac-rendered scene", flush=True)
        world.build()

        rl_config = configs.get("rl", {})
        adapter_config = {
            "control_dt": float(configs["sim"].get("physics_dt", 1.0 / 120.0)),
            "max_lidar_distance_m": float(rl_config.get("observation", {}).get("max_lidar_distance_m", 30.0)),
        }
        adapter = IsaacWorldAdapter(world, adapter_config)
        env = LunarLandingIsaacEnv(
            adapter,
            {
                **rl_config,
                "max_episode_steps": int(args.frames),
                "control_dt": float(configs["sim"].get("physics_dt", 1.0 / 120.0)),
            },
        )

        print("[LunarRocket] loading policy weights...", flush=True)
        model = SAC.load(str(model_path), env=env)

        import omni.kit.viewport.utility as vu
        print("[LunarRocket] creating Isaac viewport render camera", flush=True)
        camera_path = _create_render_camera(world.stage, args.width, args.height)
        
        viewport = vu.get_active_viewport()
        if viewport is None:
            window = vu.create_viewport_window("LunarRocketCapture", width=int(args.width), height=int(args.height))
            viewport = getattr(window, "viewport_api", None) or vu.get_active_viewport()
        if viewport is None:
            raise RuntimeError("Isaac viewport API was not available for capture")
        _set_viewport_camera(viewport, camera_path)

        print("[LunarRocket] warming Isaac renderer", flush=True)
        for _ in range(36):
            world.world.step(render=True)
            simulation_app.update()

        print(f"[LunarRocket] running policy inference and writing frames to: {frame_dir}", flush=True)
        obs, _ = env.reset()
        
        for idx in range(max(1, int(args.frames))):
            # Predict action from trained policy
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            # Follow rocket with the viewport camera
            rocket_pos = np.array(world.state.rocket.position, dtype=float)
            target = rocket_pos + np.array([0.0, 0.0, 0.35], dtype=float)
            phase = idx / max(1, int(args.frames) - 1)
            angle = np.deg2rad(225.0 + 42.0 * phase)
            radius = 6.5 - 1.0 * phase
            height = 2.8 - 0.3 * phase
            camera_position = target + np.array(
                [radius * np.cos(angle), radius * np.sin(angle), height],
                dtype=float,
            )
            _set_camera_pose(world.stage, camera_path, camera_position, np.array(_look_at_quat_wxyz(camera_position, target)))

            # Capture viewport frame
            for _ in range(4):
                simulation_app.update()
            frame_path = frame_dir / f"frame_{idx:04d}.png"
            _capture_viewport_frame(vu, viewport, frame_path, simulation_app)
            try:
                os.chmod(frame_path, 0o666)
            except Exception:
                pass
            
            if idx % 20 == 0:
                print(f"[LunarRocket] captured frame {idx + 1}/{args.frames} | height: {rocket_pos[2]:.2f}m", flush=True)
            
            if done or truncated:
                print(f"[LunarRocket] episode finished at step {idx}, resetting environment...", flush=True)
                obs, _ = env.reset()

        print(f"[LunarRocket] Isaac Sim frames successfully written", flush=True)
    except BaseException as exc:
        print(f"[LunarRocket] Isaac render capture failed: {type(exc).__name__}: {exc!r}", flush=True)
        raise
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
