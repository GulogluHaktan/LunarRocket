from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_configs, resolve_project_path


def _look_at_quat_wxyz(camera_position: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    forward = target - camera_position
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(forward, world_up)
    right /= float(np.linalg.norm(right))
    up = np.cross(right, forward)
    up /= float(np.linalg.norm(up))

    # USD camera convention: local -Z looks forward, local +Y is up.
    rotation = np.column_stack((right, up, -forward))
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        idx = int(np.argmax(np.diag(rotation)))
        if idx == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif idx == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    quat = np.array([w, x, y, z], dtype=float)
    quat /= float(np.linalg.norm(quat))
    return tuple(float(v) for v in quat)


def _create_render_camera(stage: Any, width: int, height: int) -> str:
    from pxr import Gf, UsdGeom

    camera_path = "/World/IsaacRenderCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(32.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(20.955 * float(height) / float(width))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000.0))
    return camera_path


def _set_camera_pose(stage: Any, camera_path: str, position: np.ndarray, orientation: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(camera_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    translate = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    orient = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    translate.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    orient.Set(Gf.Quatd(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])))


def _set_viewport_camera(viewport: Any, camera_path: str) -> None:
    for attr in ("camera_path", "camera_path_string"):
        if hasattr(viewport, attr):
            try:
                setattr(viewport, attr, camera_path)
                return
            except Exception:
                pass
    for method_name in ("set_active_camera", "set_camera_path"):
        method = getattr(viewport, method_name, None)
        if method is not None:
            method(camera_path)
            return
    raise RuntimeError("Could not bind Isaac viewport to render camera")


def _capture_viewport_frame(vu: Any, viewport: Any, path: Path, simulation_app: Any) -> None:
    future = vu.capture_viewport_to_file(viewport, str(path), is_hdr=False)
    wait_for_result = getattr(future, "wait_for_result", None)
    if wait_for_result is not None:
        result = wait_for_result()
        if inspect.isawaitable(result):
            import omni.kit.async_engine

            task = omni.kit.async_engine.run_coroutine(result)
            for _ in range(240):
                simulation_app.update()
                if task.done():
                    break
            if not task.done():
                raise TimeoutError(f"Timed out waiting for Isaac viewport capture: {path}")
            task.result()
    else:
        wait = getattr(future, "wait", None)
        if wait is not None:
            wait()

    for _ in range(4):
        simulation_app.update()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Isaac viewport capture did not write frame: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a real Isaac Sim rendered MP4 from the lunar landing scene")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output", default="outputs/videos/isaac_lunar_landing_real.mp4")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--renderer", default="RayTracedLighting")
    parser.add_argument("--terrain-resolution", type=int, default=96, help="Lower this if 8GB VRAM struggles")
    parser.add_argument("--no-headless", action="store_true", help="Open an Isaac viewport window for capture")
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
        from app.world_builder import LunarLandingWorld

        env = LunarLandingWorld(configs)
        print("[LunarRocket] building Isaac-rendered scene", flush=True)
        env.build()
        if env.state is None:
            raise RuntimeError("Environment did not produce a reset state")

        import omni.kit.viewport.utility as vu

        print("[LunarRocket] creating Isaac viewport render camera", flush=True)
        camera_path = _create_render_camera(env.stage, args.width, args.height)
        rocket_start = np.array(env.state.rocket.position, dtype=float)
        target = rocket_start + np.array([0.0, 0.0, 0.35], dtype=float)
        initial_angle = np.deg2rad(225.0)
        initial_position = target + np.array([7.0 * np.cos(initial_angle), 7.0 * np.sin(initial_angle), 3.2])
        _set_camera_pose(env.stage, camera_path, initial_position, np.array(_look_at_quat_wxyz(initial_position, target)))

        viewport = vu.get_active_viewport()
        if viewport is None:
            window = vu.create_viewport_window("LunarRocketCapture", width=int(args.width), height=int(args.height))
            viewport = getattr(window, "viewport_api", None) or vu.get_active_viewport()
        if viewport is None:
            raise RuntimeError("Isaac viewport API was not available for capture")
        _set_viewport_camera(viewport, camera_path)

        print("[LunarRocket] warming Isaac renderer", flush=True)
        for _ in range(36):
            env.world.step(render=True)
            simulation_app.update()

        print(f"[LunarRocket] writing real Isaac frames to: {frame_dir}", flush=True)
        for idx in range(max(1, int(args.frames))):
            phase = idx / max(1, int(args.frames) - 1)
            angle = np.deg2rad(225.0 + 42.0 * phase)
            radius = 7.0 - 1.2 * phase
            height = 3.2 - 0.5 * phase
            camera_position = target + np.array(
                [radius * np.cos(angle), radius * np.sin(angle), height],
                dtype=float,
            )
            _set_camera_pose(env.stage, camera_path, camera_position, np.array(_look_at_quat_wxyz(camera_position, target)))

            throttle = 0.18 if idx < int(args.frames) * 0.75 else 0.0
            env.step(throttle=throttle)
            for _ in range(4):
                simulation_app.update()
            _capture_viewport_frame(vu, viewport, frame_dir / f"frame_{idx:04d}.png", simulation_app)
            if idx % 20 == 0:
                print(f"[LunarRocket] captured Isaac frame {idx + 1}/{args.frames}", flush=True)

        print(f"[LunarRocket] real Isaac Sim frames written for: {output}", flush=True)
    except BaseException as exc:
        print(f"[LunarRocket] Isaac render capture failed: {type(exc).__name__}: {exc!r}", flush=True)
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
