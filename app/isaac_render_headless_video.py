from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_configs, resolve_project_path
from app.isaac_render_video import _look_at_quat_wxyz

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a real Isaac Sim rendered MP4 from the lunar landing scene headlessly")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output", default="outputs/videos/isaac_lunar_landing_real.mp4")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--renderer", default="RayTracedLighting")
    parser.add_argument("--terrain-resolution", type=int, default=96)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output.parent / "_frames_isaac_real"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("frame_*.png"):
        old_frame.unlink()

    configs = load_configs(args.config_dir)
    configs["sim"]["headless"] = True
    configs["sim"]["renderer"] = args.renderer
    configs["sim"]["window_width"] = int(args.width)
    configs["sim"]["window_height"] = int(args.height)
    configs["sim"]["disable_viewport_updates"] = False
    configs["sim"]["render_on_step"] = True
    
    # Use a free-moving camera for clean orbit around the static rocket
    configs["sim"].setdefault("camera", {})
    configs["sim"]["camera"]["enabled"] = True
    configs["sim"]["camera"]["prim_path"] = "/World/IsaacRenderCamera"
    configs["sim"]["camera"]["resolution"] = [int(args.width), int(args.height)]
    
    configs["terrain"]["mesh_resolution"] = int(args.terrain_resolution)

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    print("[LunarHeadless] Creating SimulationApp...", flush=True)
    simulation_app = SimulationApp(
        {
            "headless": True,
            "renderer": args.renderer,
            "width": int(args.width),
            "height": int(args.height),
            "disable_viewport_updates": False,
        }
    )

    try:
        from app.world_builder import LunarLandingWorld
        import omni.replicator.core as rep
        
        try:
            from isaacsim.core.utils.xforms import get_world_pose
        except ImportError:
            from omni.isaac.core.utils.xforms import get_world_pose

        def get_terrain_height_at(terrain: Any, x: float, y: float) -> float:
            size_x, size_y = terrain.size_m
            heights = terrain.heights
            rows, cols = heights.shape
            col = int(np.clip((x + size_x / 2.0) / size_x * (cols - 1), 0, cols - 1))
            row = int(np.clip((y + size_y / 2.0) / size_y * (rows - 1), 0, rows - 1))
            return float(heights[row, col])

        env = LunarLandingWorld(configs)
        print("[LunarHeadless] Building Isaac-rendered scene...", flush=True)
        env.build()
        if env.state is None:
            raise RuntimeError("Environment did not produce a reset state")

        cam = env.sensors._camera
        if cam is None:
            raise RuntimeError("Camera sensor was not built successfully!")

        print("[LunarHeadless] Adding RGB sensor to frame...", flush=True)
        cam.add_rgb_to_frame()

        # Warm up the renderer and let rocket settle on the ground
        print("[LunarHeadless] Warming up renderer and letting rocket settle...", flush=True)
        for warm_step in range(60):
            env.world.step(render=True)
            try:
                # Keep camera behind the rocket temporarily during warmup settling
                r_pos, r_orient = get_world_pose("/World/Rocket")
                cam_pos = r_pos + np.array([0.0, -4.5, 1.8])
                # Ensure camera doesn't go underground during warmup
                g_z = get_terrain_height_at(env.state.terrain, cam_pos[0], cam_pos[1])
                cam_pos[2] = max(cam_pos[2], g_z + 1.6)

                target = r_pos + np.array([0.0, 0.0, 0.5])
                cam_orient = np.array(_look_at_quat_wxyz(cam_pos, target))
                cam.set_world_pose(position=cam_pos, orientation=cam_orient)
            except Exception:
                pass
            rep.orchestrator.step()
            simulation_app.update()

        # Get final settled rocket position
        r_pos, _ = get_world_pose("/World/Rocket")
        print(f"[LunarHeadless] Rocket settled on terrain at: {r_pos}", flush=True)
        target = r_pos + np.array([0.0, 0.0, 0.45], dtype=float)

        print(f"[LunarHeadless] Starting orbital capture of {args.frames} frames...", flush=True)
        for idx in range(max(1, int(args.frames))):
            phase = idx / max(1, int(args.frames) - 1)
            angle = np.deg2rad(225.0 + 360.0 * phase) # 360-degree orbit
            radius = 5.5
            height = 1.6
            
            camera_position = target + np.array(
                [radius * np.cos(angle), radius * np.sin(angle), height],
                dtype=float,
            )
            
            # Ground-following: raise camera Z to stay at least 1.6m above local terrain
            g_z = get_terrain_height_at(env.state.terrain, camera_position[0], camera_position[1])
            camera_position[2] = max(camera_position[2], g_z + 1.6)
            
            cam.set_world_pose(position=camera_position, orientation=np.array(_look_at_quat_wxyz(camera_position, target)))
            
            # Physics is NOT stepped during capture to keep the scene static
            rep.orchestrator.step()
            simulation_app.update()
            
            # Retrieve frame
            frame = cam.get_current_frame()
            rgb = frame.get("rgb")
            if rgb is None:
                raise RuntimeError(f"Failed to get RGB frame at step {idx}!")

            # Save frame (slice the first 3 channels in case it is RGBA)
            img_data = rgb[..., :3]
            img = Image.fromarray(img_data)
            img.save(frame_dir / f"frame_{idx:04d}.png")
            
            if idx % 20 == 0 or idx == args.frames - 1:
                print(f"[LunarHeadless] Captured frame {idx + 1}/{args.frames}", flush=True)

        print("[LunarHeadless] Headless rendering complete!", flush=True)
    except BaseException as exc:
        print(f"[LunarHeadless] Isaac render capture failed: {type(exc).__name__}: {exc!r}", flush=True)
        raise
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
