from __future__ import annotations

import os
import sys
import time
import json
import subprocess
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
    res = int(config.get("heightmap_resolution", 64))
    size_m = float(config.get("patch_size_m", 40.0))
    elev_scale = float(config.get("elevation_scale", 0.08))
    noise_amp = float(config.get("noise_amplitude", 0.2))
    
    rng = np.random.default_rng(seed)
    
    x = np.linspace(-size_m / 2.0, size_m / 2.0, res)
    y = np.linspace(-size_m / 2.0, size_m / 2.0, res)
    xx, yy = np.meshgrid(x, y)
    
    heights = elev_scale * (np.sin(xx * 0.1) + np.cos(yy * 0.1))
    heights += noise_amp * rng.standard_normal(size=(res, res))
    
    crater_count = int(config.get("crater_count", 6))
    radius_min, radius_max = config.get("crater_radius_range", [1.0, 5.0])
    depth_scale = float(config.get("crater_depth_scale", 0.15))
    
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
        heights[mask_rim] += 0.25 * depth * (1.0 - ((dist[mask_rim] - r) / (0.3 * r))) ** 2

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

def main() -> None:
    start_time = time.time()
    
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    config_path = str(project_root / "configs" / "debug_min_vram.yaml")
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)
        
    vram_stats = {}
    vram_stats["VRAM usage at app startup"] = get_vram_info().get("used_mb", 0.0)
    
    print("[DebugVRAM] Creating SimulationApp...", flush=True)
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
    
    vram_stats["VRAM usage after empty world"] = get_vram_info().get("used_mb", 0.0)
    
    try:
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics
        try:
            from isaacsim.core.api import World
        except ImportError:
            from omni.isaac.core import World
            
        print("[DebugVRAM] Creating empty world...", flush=True)
        world = World(stage_units_in_meters=1.0)
        stage = world.stage
        
        # Disable expensive rendering features (commented out to prevent RTX pipeline compile crashes)
        # print("[DebugVRAM] Disabling expensive render features...", flush=True)
        # import carb
        # settings = carb.settings.get_settings()
        # settings.set_bool("/rtx/shadows/enabled", False)
        # settings.set_bool("/rtx/reflections/enabled", False)
        # settings.set_bool("/rtx/ambientOcclusion/enabled", False)
        # settings.set_bool("/rtx/post/bloom/enabled", False)
        # settings.set_bool("/rtx/post/motionblur/enabled", False)
        # settings.set_int("/rtx/post/aa/op", 0)
        
        # 1. Create Terrain
        print("[DebugVRAM] Creating procedural terrain...", flush=True)
        terrain_cfg = configs["terrain"]
        size_m = float(terrain_cfg["patch_size_m"])
        heights = generate_procedural_terrain(terrain_cfg, seed=int(configs.get("seed", 7)))
        vertices, faces, face_counts = build_terrain_mesh_data(heights, size_m)
        
        terrain_min_z = float(np.min(heights))
        terrain_max_z = float(np.max(heights))
        
        mesh = UsdGeom.Mesh.Define(stage, "/World/MoonTerrain")
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in vertices])
        mesh.CreateFaceVertexCountsAttr(face_counts)
        mesh.CreateFaceVertexIndicesAttr(faces)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr([Gf.Vec3f(0.4, 0.4, 0.4)])
        
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("none")
        prim.CreateAttribute("physxCollision:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        
        vram_stats["VRAM usage after terrain creation"] = get_vram_info().get("used_mb", 0.0)
        
        # 2. Create Placeholder Rocket
        print("[DebugVRAM] Spawning placeholder rocket from primitives...", flush=True)
        rocket_z = float(terrain_max_z + float(configs["rocket"]["spawn_above_terrain_m"]))
        rocket_pos = (0.0, 0.0, rocket_z)
        
        rocket_xform = UsdGeom.Xform.Define(stage, "/World/Rocket")
        rocket_xform.AddTranslateOp().Set(Gf.Vec3d(*rocket_pos))
        
        # Body
        body = UsdGeom.Cylinder.Define(stage, "/World/Rocket/Body")
        body.CreateRadiusAttr(0.4)
        body.CreateHeightAttr(2.0)
        body.CreateAxisAttr("Z")
        body.CreateDisplayColorAttr([Gf.Vec3f(0.7, 0.7, 0.7)])
        body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
        
        # Nose
        nose = UsdGeom.Cone.Define(stage, "/World/Rocket/Nose")
        nose.CreateRadiusAttr(0.4)
        nose.CreateHeightAttr(0.8)
        nose.CreateAxisAttr("Z")
        nose.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.2, 0.2)])
        nose.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 2.4))
        
        # Flame
        flame = UsdGeom.Cone.Define(stage, "/World/Rocket/Flame")
        flame.CreateRadiusAttr(0.2)
        flame.CreateHeightAttr(0.4)
        flame.CreateAxisAttr("Z")
        flame.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.5, 0.1)])
        flame_xform = UsdGeom.Xformable(flame)
        flame_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.2))
        flame_xform.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
        
        # Add physics API to rocket root
        rocket_prim = rocket_xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(rocket_prim)
        rocket_prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(True)
        
        vram_stats["VRAM usage after rocket creation"] = get_vram_info().get("used_mb", 0.0)
        
        # 3. Create Camera
        print("[DebugVRAM] Configuring camera and lighting...", flush=True)
        camera_cfg = configs["camera"]
        cam_pos = np.array([float(x) for x in camera_cfg["position"]], dtype=float)
        cam_target = np.array([0.0, 0.0, float(terrain_max_z + 2.0)], dtype=float)
        cam_orient = _look_at_quat_wxyz(cam_pos, cam_target)
        
        # Spawn DistantLight
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(8000.0)
        sun.CreateAngleAttr(0.53)
        sun.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 0.0, 25.0))
        
        # 4. Setup Camera Sensor
        try:
            from isaacsim.sensors.camera import Camera
        except ImportError:
            from omni.isaac.sensor import Camera
            
        cam = Camera(
            prim_path="/World/IsaacRenderCamera",
            resolution=[int(configs["app"]["resolution"][0]), int(configs["app"]["resolution"][1])],
            orientation=np.array(cam_orient),
        )
        cam.initialize()
        
        # Configure focal length and clipping range on the created camera prim safely
        camera_geom = UsdGeom.Camera(cam.prim)
        camera_geom.GetFocalLengthAttr().Set(float(camera_cfg["focal_length"]))
        camera_geom.GetClippingRangeAttr().Set(Gf.Vec2f(float(camera_cfg["clipping_range"][0]), float(camera_cfg["clipping_range"][1])))
        
        cam.add_rgb_to_frame()
        
        # Reset the world to initialize all sensors, replicator, and stage properties!
        print("[DebugVRAM] Resetting world to initialize scene...", flush=True)
        world.reset()
        
        # Set camera pose AFTER world reset!
        # Clear any existing transform operations to prevent conflicts, then set translate and orient directly on the USD stage.
        xform = UsdGeom.Xformable(cam.prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])))
        orient_op.Set(Gf.Quatd(float(cam_orient[0]), float(cam_orient[1]), float(cam_orient[2]), float(cam_orient[3])))
        
        # Warm up the renderer
        print("[DebugVRAM] Warming up renderer...", flush=True)
        import omni.replicator.core as rep
        for warm_step in range(50):
            world.step(render=True)
            rep.orchestrator.step()
            simulation_app.update()
            
        # Export USD stage for debugging
        usd_debug_path = str(project_root / "outputs" / "debug_min_vram_scene.usda")
        stage.Export(usd_debug_path)
        print(f"[DebugVRAM] Exported USDA scene to {usd_debug_path}", flush=True)
        
        # Capture frame
        frame = cam.get_current_frame()
        print("[DebugVRAM] frame keys:", list(frame.keys()) if frame else None, flush=True)
        rgb = frame.get("rgb")
        if rgb is not None:
            print("[DebugVRAM] rgb shape:", rgb.shape, "max:", rgb.max(), "mean:", rgb.mean(), flush=True)
        if rgb is None:
            raise RuntimeError("Failed to capture RGB frame!")
            
        # Save screenshot
        img_dir = project_root / "outputs"
        img_dir.mkdir(parents=True, exist_ok=True)
        img_data = rgb[..., :3]
        img = Image.fromarray(img_data)
        img_path = str(project_root / configs["output"]["screenshot_path"])
        img.save(img_path)
        print(f"[DebugVRAM] Screenshot successfully saved to {img_path}", flush=True)
        
        vram_stats["VRAM usage after screenshot"] = get_vram_info().get("used_mb", 0.0)
        
        # Calculate performance stats
        total_runtime = time.time() - start_time
        gpu_info = get_vram_info()
        
        stats = {
            "GPU name": gpu_info.get("gpu_name", "Unknown"),
            "VRAM usage at app startup (MB)": vram_stats["VRAM usage at app startup"],
            "VRAM usage after empty world (MB)": vram_stats["VRAM usage after empty world"],
            "VRAM usage after terrain creation (MB)": vram_stats["VRAM usage after terrain creation"],
            "VRAM usage after rocket creation (MB)": vram_stats["VRAM usage after rocket creation"],
            "VRAM usage after screenshot (MB)": vram_stats["VRAM usage after screenshot"],
            "total runtime (s)": total_runtime,
            "terrain vertex count": len(vertices),
            "terrain triangle count": len(faces) // 3,
            "terrain min elevation (m)": terrain_min_z,
            "terrain max elevation (m)": terrain_max_z,
            "rocket spawn position": list(rocket_pos),
            "camera position": list(cam_pos),
            "camera target": list(cam_target)
        }
        
        stats_path = str(project_root / configs["output"]["stats_path"])
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        print(f"[DebugVRAM] Performance stats saved to {stats_path}", flush=True)
        
    except BaseException as exc:
        print(f"[DebugVRAM] Execution failed: {type(exc).__name__}: {exc!r}", flush=True)
        raise
    finally:
        print("[DebugVRAM] Cleaning up and shutting down SimulationApp...", flush=True)
        simulation_app.close()
        
if __name__ == "__main__":
    main()
