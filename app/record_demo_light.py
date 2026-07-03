from __future__ import annotations

import os
import sys
import time
import json
import subprocess
from pathlib import Path
import numpy as np
import yaml
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))))

def get_vram_info() -> dict:
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        parts = [p.strip() for p in res.strip().split(",")]
        return {
            "gpu_name": parts[0],
            "total_mb": float(parts[1]),
            "used_mb": float(parts[2]),
            "free_mb": float(parts[3])
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "gpu_name": "Unknown",
            "total_mb": 0.0,
            "used_mb": 0.0,
            "free_mb": 0.0
        }

def _look_at_quat_wxyz(camera_position: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    forward = target - camera_position
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-9:
        right = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        right /= right_norm
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1e-9)
    rotation = np.column_stack((right, up, -forward))
    
    m = rotation
    t = np.trace(m)
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
            
    quat = np.array([w, x, y, z], dtype=float)
    quat /= float(np.linalg.norm(quat))
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

def smooth_heightmap(heights: np.ndarray, passes: int = 2) -> np.ndarray:
    smoothed = heights.copy()
    for _ in range(passes):
        shifted = np.zeros_like(smoothed)
        shifted[1:-1, 1:-1] = (
            smoothed[1:-1, 1:-1] +
            smoothed[:-2, 1:-1] +
            smoothed[2:, 1:-1] +
            smoothed[1:-1, :-2] +
            smoothed[1:-1, 2:]
        ) / 5.0
        smoothed[1:-1, 1:-1] = shifted[1:-1, 1:-1]
    return smoothed

def generate_procedural_terrain(config: dict, seed: int = 7) -> np.ndarray:
    res = int(config.get("heightmap_resolution", 128))
    size_m = float(config.get("patch_size_m", 80.0))
    elev_scale = float(config.get("elevation_scale", 0.10))
    noise_amp = float(config.get("noise_amplitude", 0.18))
    
    rng = np.random.default_rng(seed)
    
    x = np.linspace(-size_m / 2.0, size_m / 2.0, res)
    y = np.linspace(-size_m / 2.0, size_m / 2.0, res)
    xx, yy = np.meshgrid(x, y)
    
    heights = elev_scale * (np.sin(xx * 0.08) + np.cos(yy * 0.08))
    heights += noise_amp * rng.standard_normal(size=(res, res))
    
    crater_count = int(config.get("crater_count", 14))
    radius_min, radius_max = config.get("crater_radius_range", [2.0, 10.0])
    depth_scale = float(config.get("crater_depth_scale", 0.20))
    
    for _ in range(crater_count):
        cx = float(rng.uniform(-size_m / 3.0, size_m / 3.0))
        cy = float(rng.uniform(-size_m / 3.0, size_m / 3.0))
        r = float(rng.uniform(radius_min, radius_max))
        depth = r * depth_scale
        
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        dist = np.sqrt(dist_sq)
        
        mask_inside = dist < r
        heights[mask_inside] -= depth * (1.0 - (dist[mask_inside] / r) ** 2)
        
        mask_rim = (dist >= r) & (dist < 1.3 * r)
        heights[mask_rim] += 0.22 * depth * (1.0 - ((dist[mask_rim] - r) / (0.3 * r))) ** 2

    return smooth_heightmap(heights, passes=3)

def build_terrain_mesh_data(heights: np.ndarray, size_m: float) -> tuple[list, list, list]:
    rows, cols = heights.shape
    x_values = np.linspace(-size_m / 2.0, size_m / 2.0, cols)
    y_values = np.linspace(-size_m / 2.0, size_m / 2.0, rows)
    
    vertices = []
    for r, y in enumerate(y_values):
        for c, x in enumerate(x_values):
            vertices.append((float(x), float(y), float(heights[r, c])))
            
    faces = []
    face_counts = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            idx0 = r * cols + c
            idx1 = r * cols + (c + 1)
            idx2 = (r + 1) * cols + c
            idx3 = (r + 1) * cols + (c + 1)
            
            faces.extend([idx0, idx2, idx1])
            face_counts.append(3)
            faces.extend([idx1, idx2, idx3])
            face_counts.append(3)
            
    return vertices, faces, face_counts

def terrain_height_near(heights: np.ndarray, size_m: float, x_m: float, y_m: float, radius_m: float = 1.0) -> float:
    rows, cols = heights.shape
    col = int(np.clip((x_m / size_m + 0.5) * (cols - 1), 0, cols - 1))
    row = int(np.clip((y_m / size_m + 0.5) * (rows - 1), 0, rows - 1))
    radius_cells = max(1, int(np.ceil(radius_m / (size_m / max(1, cols - 1)))))
    r0 = max(0, row - radius_cells)
    r1 = min(rows, row + radius_cells + 1)
    c0 = max(0, col - radius_cells)
    c1 = min(cols, col + radius_cells + 1)
    return float(np.max(heights[r0:r1, c0:c1]))

def main() -> None:
    app_start_time = time.time()
    
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    config_path = str(project_root / "configs" / "record_demo_light.yaml")
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)
        
    vram_stats = {}
    vram_stats["vram_before_scene_creation"] = get_vram_info().get("used_mb", 0.0)
    
    print("[RecordDemoLight] Creating SimulationApp...", flush=True)
    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp
        
    simulation_app = SimulationApp(
        {
            "headless": True,
            "width": int(configs["app"]["resolution"][0]),
            "height": int(configs["app"]["resolution"][1]),
            "renderer": "RayTracedLighting",
            "disable_viewport_updates": False
        }
    )
    
    try:
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics
        try:
            from isaacsim.core.api import World
        except ImportError:
            from omni.isaac.core import World
            
        print("[RecordDemoLight] Creating empty world...", flush=True)
        world = World(stage_units_in_meters=1.0)
        stage = world.stage
        
        # 1. Create Terrain
        terrain_gen_start = time.time()
        print("[RecordDemoLight] Generating procedural terrain...", flush=True)
        terrain_cfg = configs["terrain"]
        size_m = float(terrain_cfg["patch_size_m"])
        heights = generate_procedural_terrain(terrain_cfg, seed=int(configs.get("seed", 7)))
        vertices, faces, face_counts = build_terrain_mesh_data(heights, size_m)
        terrain_gen_time = time.time() - terrain_gen_start
        
        terrain_min_z = float(np.min(heights))
        terrain_max_z = float(np.max(heights))
        # Find elevation at center of terrain (0, 0)
        mid_row = heights.shape[0] // 2
        mid_col = heights.shape[1] // 2
        terrain_center_z = float(heights[mid_row, mid_col])
        terrain_landing_z = terrain_height_near(heights, size_m, 0.0, 0.0, radius_m=1.2)
        
        mesh = UsdGeom.Mesh.Define(stage, "/World/MoonTerrain")
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in vertices])
        mesh.CreateFaceVertexCountsAttr(face_counts)
        mesh.CreateFaceVertexIndicesAttr(faces)
        mesh.CreateSubdivisionSchemeAttr("none")
        # Medium lunar gray color: [0.38, 0.38, 0.36]
        mesh.CreateDisplayColorAttr([Gf.Vec3f(0.38, 0.38, 0.36)])
        
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        try:
            from pxr import PhysxSchema
            physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            if hasattr(physx_collision_api, "CreateApproximationAttr"):
                physx_collision_api.CreateApproximationAttr().Set("none")
        except ImportError:
            pass
        prim.CreateAttribute("physxCollision:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        
        vram_stats["vram_after_terrain_creation"] = get_vram_info().get("used_mb", 0.0)
        
        # 2. Create Placeholder Rocket
        print("[RecordDemoLight] Spawning placeholder rocket from primitives...", flush=True)
        rocket_cfg = configs["rocket"]
        rocket_z = float(terrain_landing_z + float(rocket_cfg["spawn_above_terrain_m"]) + float(rocket_cfg.get("ground_clearance_m", 0.35)))
        rocket_pos = (0.0, 0.0, rocket_z)
        
        rocket_xform = UsdGeom.Xform.Define(stage, "/World/Rocket")
        rocket_translate_op = rocket_xform.AddTranslateOp()
        rocket_translate_op.Set(Gf.Vec3d(*rocket_pos))
        
        # Body
        body = UsdGeom.Cylinder.Define(stage, "/World/Rocket/Body")
        body.CreateRadiusAttr(0.4)
        body.CreateHeightAttr(2.0)
        body.CreateAxisAttr("Z")
        body.CreateDisplayColorAttr([Gf.Vec3f(*rocket_cfg["body_color"])])
        body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
        
        # Nose
        nose = UsdGeom.Cone.Define(stage, "/World/Rocket/Nose")
        nose.CreateRadiusAttr(0.4)
        nose.CreateHeightAttr(0.8)
        nose.CreateAxisAttr("Z")
        nose.CreateDisplayColorAttr([Gf.Vec3f(*rocket_cfg["nose_color"])])
        nose.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 2.4))
        
        # Optional Flame Marker
        if bool(rocket_cfg.get("flame_marker", True)):
            flame = UsdGeom.Cone.Define(stage, "/World/Rocket/Flame")
            flame.CreateRadiusAttr(0.2)
            flame.CreateHeightAttr(0.4)
            flame.CreateAxisAttr("Z")
            flame.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.5, 0.1)])
            flame_xform = UsdGeom.Xformable(flame)
            flame_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.2))
            flame_xform.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
            
        if bool(rocket_cfg.get("shadow_marker", False)):
            shadow = UsdGeom.Cylinder.Define(stage, "/World/RocketShadowMarker")
            shadow.CreateRadiusAttr(0.8)
            shadow.CreateHeightAttr(0.02)
            shadow.CreateAxisAttr("Z")
            shadow.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.12, 0.12)])
            shadow.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(terrain_landing_z + 0.01)))
        
        # Add physics API to rocket root
        rocket_prim = rocket_xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(rocket_prim)
        rocket_prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(True)
        
        vram_stats["vram_after_rocket_creation"] = get_vram_info().get("used_mb", 0.0)
        
        # 3. Setup Camera and Lighting
        print("[RecordDemoLight] Configuring camera and lighting...", flush=True)
        camera_cfg = configs["camera"]
        pos_start = np.array([float(x) for x in camera_cfg["position_start"]], dtype=float)
        pos_end = np.array([float(x) for x in camera_cfg["position_end"]], dtype=float)
        cam_target = np.array([0.0, 0.0, float(terrain_center_z + float(camera_cfg["target_offset_z"]))], dtype=float)
        
        # Create DistantLight
        light_cfg = configs["light"]
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(float(light_cfg["intensity"]))
        sun.CreateAngleAttr(float(light_cfg["angle"]))
        # Set light rotation to orient the directional light correctly
        sun.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 0.0, 25.0))
        
        # 4. Setup Camera Sensor
        try:
            from isaacsim.sensors.camera import Camera
        except ImportError:
            from omni.isaac.sensor import Camera
            
        # Compute starting look-at to instantiate camera sensor cleanly
        cam_orient_start = _look_at_quat_wxyz(pos_start, cam_target)
        cam = Camera(
            prim_path=str(camera_cfg["path"]),
            resolution=[int(configs["app"]["resolution"][0]), int(configs["app"]["resolution"][1])],
            orientation=np.array(cam_orient_start),
        )
        cam.initialize()
        
        # Configure focal length and clipping range safely on the camera prim
        camera_geom = UsdGeom.Camera(cam.prim)
        camera_geom.GetFocalLengthAttr().Set(float(camera_cfg["focal_length"]))
        camera_geom.GetClippingRangeAttr().Set(Gf.Vec2f(float(camera_cfg["clipping_range"][0]), float(camera_cfg["clipping_range"][1])))
        
        cam.add_rgb_to_frame()
        
        # Reset the world to initialize everything
        print("[RecordDemoLight] Resetting world to initialize scene...", flush=True)
        world.reset()
        
        # 5. Capture Video Frames Loop
        print("[RecordDemoLight] Warming up renderer...", flush=True)
        
        # Initial pose before warmup
        xform = UsdGeom.Xformable(cam.prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(float(pos_start[0]), float(pos_start[1]), float(pos_start[2])))
        orient_op.Set(Gf.Quatd(float(cam_orient_start[0]), float(cam_orient_start[1]), float(cam_orient_start[2]), float(cam_orient_start[3])))
        
        # Warmup loop (45 frames is sufficient now that viewport is active)
        for warm_step in range(45):
            world.step(render=True)
            simulation_app.update()
            
        frames_dir = Path(project_root / configs["output"]["frames_dir"])
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear out any old frames in the directory
        for f_path in frames_dir.glob("frame_*.png"):
            f_path.unlink()
            
        frame_count = int(configs["app"]["frame_count"])
        print(f"[RecordDemoLight] Capturing {frame_count} frames...", flush=True)
        
        capture_times = []
        peak_vram = get_vram_info().get("used_mb", 0.0)
        
        video_capture_start = time.time()
        
        for idx in range(frame_count):
            frame_start = time.time()
            
            # Interpolate camera position
            fraction = idx / max(1, frame_count - 1)
            if bool(rocket_cfg.get("descent", True)):
                start_clearance = float(rocket_cfg["spawn_above_terrain_m"]) + float(rocket_cfg.get("ground_clearance_m", 0.35))
                final_clearance = float(rocket_cfg.get("final_clearance_m", 0.9))
                eased = 1.0 - (1.0 - fraction) ** 2
                rocket_z_frame = terrain_landing_z + start_clearance * (1.0 - eased) + final_clearance * eased
                rocket_translate_op.Set(Gf.Vec3d(0.0, 0.0, float(rocket_z_frame)))
            cam_pos = pos_start + fraction * (pos_end - pos_start)
            cam_orient = _look_at_quat_wxyz(cam_pos, cam_target)
            
            # Apply transform to stage directly
            translate_op.Set(Gf.Vec3d(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])))
            orient_op.Set(Gf.Quatd(float(cam_orient[0]), float(cam_orient[1]), float(cam_orient[2]), float(cam_orient[3])))
            
            # Step simulation & render
            world.step(render=True)
            simulation_app.update()
            
            # Capture frame
            frame = cam.get_current_frame()
            rgb = frame.get("rgb")
            if rgb is None:
                raise RuntimeError(f"Failed to capture RGB frame at index {idx}!")
                
            # Track VRAM during capture loop
            current_vram = get_vram_info().get("used_mb", 0.0)
            peak_vram = max(peak_vram, current_vram)
            
            if idx == 0:
                vram_stats["vram_after_first_frame"] = current_vram
                first_frame_capture_time = time.time() - frame_start
                
            # Save frame
            img_data = rgb[..., :3]
            img = Image.fromarray(img_data)
            img.save(frames_dir / f"frame_{idx:04d}.png")
            
            capture_times.append(time.time() - frame_start)
            
            if idx % 20 == 0 or idx == frame_count - 1:
                print(f"[RecordDemoLight] Captured frame {idx + 1}/{frame_count} | VRAM: {current_vram} MB", flush=True)
                
        vram_stats["vram_after_final_frame"] = get_vram_info().get("used_mb", 0.0)
        video_capture_time = time.time() - video_capture_start
        average_frame_capture_time = float(np.mean(capture_times))
        
        # 6. Encode video with ffmpeg (commented out inside container, will be run on the host)
        print("[RecordDemoLight] Frames saved. Video encoding will be executed on the host.", flush=True)
        video_path = Path(project_root / configs["output"]["video_path"])
        ffmpeg_encode_time = 0.0
        
        # Write stats.json
        stats_path = Path(project_root / configs["output"]["stats_path"])
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        stats = {
            "GPU name": get_vram_info().get("gpu_name", "Unknown"),
            "VRAM before scene creation (MB)": vram_stats["vram_before_scene_creation"],
            "VRAM after terrain creation (MB)": vram_stats["vram_after_terrain_creation"],
            "VRAM after rocket creation (MB)": vram_stats["vram_after_rocket_creation"],
            "VRAM after first frame (MB)": vram_stats["vram_after_first_frame"],
            "VRAM after final frame (MB)": vram_stats["vram_after_final_frame"],
            "peak VRAM (MB)": peak_vram,
            "app startup time (s)": float(time.time() - app_start_time - video_capture_time - ffmpeg_encode_time - terrain_gen_time),
            "terrain generation time (s)": terrain_gen_time,
            "first frame capture time (s)": first_frame_capture_time,
            "average frame capture time (s)": average_frame_capture_time,
            "total video capture time (s)": video_capture_time,
            "ffmpeg encode time (s)": ffmpeg_encode_time,
            "terrain vertex count": len(vertices),
            "terrain triangle count": len(faces) // 3,
            "terrain min elevation (m)": terrain_min_z,
            "terrain max elevation (m)": terrain_max_z,
            "terrain landing elevation (m)": terrain_landing_z,
            "rocket final clearance (m)": float(rocket_cfg.get("final_clearance_m", 0.9)),
            "rocket spawn position": list(rocket_pos),
            "camera initial position": list(pos_start),
            "camera target": list(cam_target),
            "output video path": str(video_path),
            "frame count": frame_count
        }
        
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"[RecordDemoLight] Performance stats saved to {stats_path}", flush=True)
        
    except BaseException as exc:
        print(f"[RecordDemoLight] Execution failed: {type(exc).__name__}: {exc!r}", flush=True)
        raise
    finally:
        print("[RecordDemoLight] Cleaning up and shutting down SimulationApp...", flush=True)
        simulation_app.close()
        
if __name__ == "__main__":
    main()
