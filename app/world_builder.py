from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.randomization import ResetSample, sample_reset
from app.rocket_asset import RocketAsset, RocketState
from app.sensors import CameraObservation, SensorSuite
from app.terrain_generator import MoonTerrainGenerator, TerrainMeshData


@dataclass
class EnvironmentState:
    reset_sample: ResetSample
    terrain: TerrainMeshData
    rocket: RocketState


class LunarLandingWorld:
    def __init__(self, configs: dict[str, dict[str, Any]]):
        self.configs = configs
        self.sim_config = configs["sim"]
        self.terrain_generator = MoonTerrainGenerator(configs["terrain"])
        self.rocket = RocketAsset(configs["rocket"])
        self.sensors = SensorSuite(self.sim_config) if self._camera_enabled() else None
        self.rng = np.random.default_rng(int(self.sim_config.get("seed", 1234)))
        self.world: Any | None = None
        self.stage: Any | None = None
        self.state: EnvironmentState | None = None

    def build(self) -> None:
        try:
            print("[LunarRocket] importing Isaac World API", flush=True)
            try:
                from isaacsim.core.api import World
                from isaacsim.core.utils.stage import get_current_stage
            except ImportError:
                from omni.isaac.core import World
                from omni.isaac.core.utils.stage import get_current_stage
        except ImportError as exc:
            raise RuntimeError("LunarLandingWorld.build() must run inside Isaac Sim Python") from exc

        print("[LunarRocket] creating World", flush=True)
        self.world = World(
            stage_units_in_meters=float(self.sim_config.get("stage_units_in_meters", 1.0)),
            physics_dt=float(self.sim_config.get("physics_dt", 1.0 / 120.0)),
            rendering_dt=float(self.sim_config.get("rendering_dt", 1.0 / 30.0)),
        )
        print("[LunarRocket] getting stage", flush=True)
        self.stage = get_current_stage()
        print("[LunarRocket] configuring physics", flush=True)
        self._configure_physics()
        print("[LunarRocket] adding lighting", flush=True)
        self._add_lighting()
        print("[LunarRocket] spawning rocket", flush=True)
        self.rocket.spawn(self.stage, self.world)
        if self.sensors is not None:
            print("[LunarRocket] building sensors", flush=True)
            self.sensors.build()
        print("[LunarRocket] resetting environment", flush=True)
        self.reset()
        self._add_presentation_camera()

    def reset(self) -> EnvironmentState:
        if self.stage is None or self.world is None:
            raise RuntimeError("LunarLandingWorld.build() must be called before reset()")

        self.terrain_generator.load_dem()
        reset_sample = sample_reset(self.configs["terrain"], self.configs["rocket"], self.rng)
        print("[LunarRocket] generating terrain mesh", flush=True)
        terrain = self.terrain_generator.generate(reset_sample)
        print("[LunarRocket] creating terrain USD mesh", flush=True)
        terrain_path = str(self.configs["terrain"].get("prim_path", "/World/MoonTerrain"))
        self.terrain_generator.create_or_update_usd_meshes(self.stage, terrain_path, terrain)

        rocket_position = self._rocket_position_above_surface(reset_sample.rocket_position, terrain)
        print("[LunarRocket] resetting Isaac world physics", flush=True)
        self.world.reset()
        print("[LunarRocket] resetting rocket pose", flush=True)
        rocket_state = self.rocket.reset_pose(rocket_position, reset_sample.rocket_euler_deg)
        self.state = EnvironmentState(reset_sample, terrain, rocket_state)
        
        # Log detailed terrain statistics
        self._log_terrain_stats(terrain)
        
        return self.state

    def step(self, throttle: float = 0.0) -> CameraObservation:
        if self.world is None:
            raise RuntimeError("LunarLandingWorld.build() must be called before step()")
        self.rocket.apply_thrust(throttle)
        self.world.step(render=bool(self.sim_config.get("render_on_step", self.sensors is not None)))
        if self.sensors is None:
            return CameraObservation(rgb=None)
        return self.sensors.get_observation()

    def save_stage(self, output_path: str) -> None:
        if self.stage is None:
            raise RuntimeError("LunarLandingWorld.build() must be called before save_stage()")
        output = str(output_path)
        try:
            from pathlib import Path

            Path(output).parent.mkdir(parents=True, exist_ok=True)
            self.stage.Export(output)
        except Exception as exc:
            raise RuntimeError(f"Failed to export USD stage to {output}") from exc
        print(f"[LunarRocket] USD stage exported: {output}", flush=True)

    def _camera_enabled(self) -> bool:
        camera_config = self.sim_config.get("camera", {})
        return bool(camera_config.get("enabled", True))

    def _configure_physics(self) -> None:
        gravity = float(self.sim_config.get("gravity_mps2", 1.62))
        if self.world is not None:
            physics_context = self.world.get_physics_context()
            try:
                physics_context.set_gravity(-gravity)
            except TypeError:
                physics_context.set_gravity(gravity)


    def _add_lighting(self) -> None:
        if self.stage is None:
            return
        try:
            from pxr import Gf, UsdLux

            sun = UsdLux.DistantLight.Define(self.stage, "/World/Sun")
            sun.CreateIntensityAttr(float(self.sim_config.get("sun_intensity", 3000.0)))
            sun.CreateAngleAttr(float(self.sim_config.get("sun_angle_deg", 0.53)))
            sun.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 0.0, 25.0))

            dome = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
            dome.CreateIntensityAttr(250.0)
        except Exception:
            pass

    def _add_presentation_camera(self) -> None:
        if self.stage is None:
            return
        try:
            from pxr import Gf, UsdGeom

            camera = UsdGeom.Camera.Define(self.stage, "/World/PresentationCamera")
            xform = UsdGeom.Xformable(camera.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(7.0, -13.0, 7.0))
            xform.AddRotateXYZOp().Set(Gf.Vec3f(60.0, 0.0, 28.0))
            camera.CreateFocalLengthAttr(24.0)
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000.0))
        except Exception:
            pass

    def _rocket_position_above_surface(
        self,
        requested: tuple[float, float, float],
        terrain: TerrainMeshData,
    ) -> tuple[float, float, float]:
        x, y, altitude = requested
        spawn_config = self.configs["rocket"].get("spawn", {})
        size_x, size_y = terrain.size_m
        rows, cols = terrain.heights.shape
        row = int(np.clip((y + size_y / 2.0) / size_y * (rows - 1), 0, rows - 1))
        col = int(np.clip((x + size_x / 2.0) / size_x * (cols - 1), 0, cols - 1))
        cell_size = min(size_x / max(1, cols - 1), size_y / max(1, rows - 1))
        radius_m = float(spawn_config.get("surface_sample_radius_m", 1.0))
        radius_cells = max(1, int(np.ceil(radius_m / max(cell_size, 1e-6))))
        r0 = max(0, row - radius_cells)
        r1 = min(rows, row + radius_cells + 1)
        c0 = max(0, col - radius_cells)
        c1 = min(cols, col + radius_cells + 1)
        surface_z = float(np.max(terrain.heights[r0:r1, c0:c1]))
        if terrain.local_detail_mesh is not None:
            local = terrain.local_detail_mesh
            local_size_x, local_size_y = local.size_m
            local_x_values = [vertex[0] for vertex in local.vertices]
            local_y_values = [vertex[1] for vertex in local.vertices]
            local_min_x = min(local_x_values)
            local_max_x = max(local_x_values)
            local_min_y = min(local_y_values)
            local_max_y = max(local_y_values)
            if local_min_x <= x <= local_max_x and local_min_y <= y <= local_max_y:
                local_rows, local_cols = local.heights.shape
                local_row = int(np.clip((y - local_min_y) / max(local_size_y, 1e-6) * (local_rows - 1), 0, local_rows - 1))
                local_col = int(np.clip((x - local_min_x) / max(local_size_x, 1e-6) * (local_cols - 1), 0, local_cols - 1))
                local_cell_size = min(
                    local_size_x / max(1, local_cols - 1),
                    local_size_y / max(1, local_rows - 1),
                )
                local_radius_cells = max(1, int(np.ceil(radius_m / max(local_cell_size, 1e-6))))
                lr0 = max(0, local_row - local_radius_cells)
                lr1 = min(local_rows, local_row + local_radius_cells + 1)
                lc0 = max(0, local_col - local_radius_cells)
                lc1 = min(local_cols, local_col + local_radius_cells + 1)
                surface_z = max(surface_z, float(np.max(local.heights[lr0:lr1, lc0:lc1])))
        clearance = float(spawn_config.get("ground_clearance_m", 0.0))
        return (x, y, surface_z + float(altitude) + clearance)

    def _log_terrain_stats(self, terrain: TerrainMeshData) -> None:
        """Log detailed terrain statistics for benchmarking and validation."""
        # Check if two-tier system
        is_two_tier = terrain.local_detail_mesh is not None
        
        if is_two_tier:
            local = terrain.local_detail_mesh
            assert local is not None
            base_triangles = terrain.triangle_count
            local_triangles = local.triangle_count
            effective_triangles = base_triangles + local_triangles
            
            print(
                "[LunarRocket] TERRAIN STATS (Two-Tier System):\n"
                f"  Base terrain:\n"
                f"    - resolution: {terrain.heightmap_resolution}×{terrain.heightmap_resolution}\n"
                f"    - triangles: {base_triangles:,}\n"
                f"    - grid_spacing_m: {terrain.grid_spacing_m:.4f}\n"
                f"  Landing zone detail:\n"
                f"    - resolution: {local.heightmap_resolution}×{local.heightmap_resolution}\n"
                f"    - triangles: {local_triangles:,}\n"
                f"    - grid_spacing_m: {local.grid_spacing_m:.4f}\n"
                f"    - size_m: {local.size_m[0]:.1f} x {local.size_m[1]:.1f}\n"
                f"  Combined:\n"
                f"    - actual_triangles: {effective_triangles:,}\n"
                f"    - elevation_range_m: {terrain.max_height_m - terrain.min_height_m:.4f}\n"
                f"    - mean_slope_deg: {terrain.mean_slope_deg:.2f}\n"
                f"    - max_slope_deg: {terrain.max_slope_deg:.2f}\n"
                f"  Physical properties:\n"
                f"    - size_m: {terrain.size_m[0]:.1f} x {terrain.size_m[1]:.1f}\n"
                f"    - min_elevation_m: {terrain.min_height_m:.4f}\n"
                f"    - max_elevation_m: {terrain.max_height_m:.4f}",
                flush=True,
            )
        else:
            print(
                "[LunarRocket] terrain stats:\n"
                f"  heightmap_resolution: {terrain.heightmap_resolution}\n"
                f"  collision_resolution: {terrain.collision_resolution}\n"
                f"  visual_resolution: {terrain.visual_resolution}\n"
                f"  vertex_count: {terrain.vertex_count}\n"
                f"  triangle_count: {terrain.triangle_count}\n"
                f"  grid_spacing_m: {terrain.grid_spacing_m:.4f}\n"
                f"  size_m: {terrain.size_m[0]:.1f} x {terrain.size_m[1]:.1f}\n"
                f"  min_elevation_m: {terrain.min_height_m:.4f}\n"
                f"  max_elevation_m: {terrain.max_height_m:.4f}\n"
                f"  elevation_range_m: {terrain.max_height_m - terrain.min_height_m:.4f}\n"
                f"  mean_slope_deg: {terrain.mean_slope_deg:.2f}\n"
                f"  max_slope_deg: {terrain.max_slope_deg:.2f}",
                flush=True,
            )
