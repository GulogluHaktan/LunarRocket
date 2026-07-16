from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs.direct_rl_env import DirectRLEnv
from isaaclab.envs.direct_rl_env_cfg import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass


def _project_root() -> Path:
    return Path(os.environ.get("LUNAR_ROCKET_ROOT", Path(__file__).resolve().parents[5])).resolve()


@configclass
class LunarLanderEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 2
    action_space = 3
    observation_space = 123
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -1.62),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.85,
            dynamic_friction=0.65,
            restitution=0.02,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=8.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    rocket: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Rocket",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_project_root() / "assets/rocket/lunar_rocket_single_body.usda"),
            scale=(6.0, 6.0, 6.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.63),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 25.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    rocket_mass_kg = 1.63
    max_thrust_n = 30.0
    max_gimbal_deg = 12.0
    thrust_lever_arm_m = 0.85
    spawn_altitude_m = 25.0
    spawn_xy_range_m = 6.0
    target_radius_m = 0.35
    max_xy_m = 18.0
    max_altitude_m = 60.0
    max_lidar_distance_m = 30.0
    lidar_ray_count = 64
    terrain_scan_count = 24
    landing_altitude_m = 0.20
    soft_vertical_speed_mps = 1.0
    soft_horizontal_speed_mps = 0.7
    soft_tilt_deg = 8.0
    terrain_height_scale_m = 0.45
    terrain_slope_scale = 0.035
    terrain_crater_count = 3
    terrain_rock_count = 5
    legacy_terrain_config_path = "configs/terrain_config.yaml"
    legacy_rocket_config_path = "configs/rocket_config.yaml"
    legacy_terrain_usd_max_envs = 16
    imu_accel_noise_std = 0.03
    imu_gyro_noise_std = 0.01
    altimeter_noise_std = 0.01
    lidar_noise_std = 0.015

    rew_progress = 2.0
    rew_target = -0.03
    rew_tilt = -1.5
    rew_angvel = -0.05
    rew_vxy = -0.08
    rew_vz = -0.25
    rew_fuel = -0.01
    rew_smooth = -0.03
    rew_time = -0.01
    rew_soft_landing = 80.0
    rew_harsh_landing = -30.0
    rew_crash = -80.0


class LunarLanderEnv(DirectRLEnv):
    cfg: LunarLanderEnvCfg

    def __init__(self, cfg: LunarLanderEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_distance_xy = torch.zeros(self.num_envs, device=self.device)
        self._body_ids = self._rocket.find_bodies(".*")[0]
        self._terrain_slope = torch.zeros(self.num_envs, 2, device=self.device)
        self._terrain_phase = torch.zeros(self.num_envs, 2, device=self.device)
        self._terrain_amp = torch.full((self.num_envs,), 0.12, device=self.device)
        self._crater_xy = torch.zeros(self.num_envs, self.cfg.terrain_crater_count, 2, device=self.device)
        self._crater_radius = torch.ones(self.num_envs, self.cfg.terrain_crater_count, device=self.device)
        self._crater_depth = torch.zeros(self.num_envs, self.cfg.terrain_crater_count, device=self.device)
        self._rock_xy = torch.zeros(self.num_envs, self.cfg.terrain_rock_count, 2, device=self.device)
        self._rock_radius = torch.ones(self.num_envs, self.cfg.terrain_rock_count, device=self.device)
        self._rock_height = torch.zeros(self.num_envs, self.cfg.terrain_rock_count, device=self.device)
        self._legacy_terrain = getattr(self, "_legacy_terrain", None)
        self._legacy_terrain_heights = getattr(self, "_legacy_terrain_heights", None)
        self._legacy_local_heights = getattr(self, "_legacy_local_heights", None)
        self._legacy_local_min_xy = getattr(self, "_legacy_local_min_xy", None)
        self._legacy_local_max_xy = getattr(self, "_legacy_local_max_xy", None)
        self._legacy_terrain_size = getattr(self, "_legacy_terrain_size", (80.0, 80.0))
        self._legacy_terrain_min_xy = getattr(self, "_legacy_terrain_min_xy", (-40.0, -40.0))
        self._legacy_terrain_max_xy = getattr(self, "_legacy_terrain_max_xy", (40.0, 40.0))

        # Run a single dummy physics step to allocate PhysX GPU tensor views before any reset() is called
        print("[LunarRocket] performing dummy physics step to initialize PhysX backend...", flush=True)
        self.sim.step(render=False)
        print("[LunarRocket] PhysX backend initialization check complete", flush=True)

    def _setup_scene(self):
        from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

        rocket_cfg = self.cfg.rocket.copy()
        if rocket_cfg.spawn is not None:
            rocket_cfg.spawn = rocket_cfg.spawn.copy()
            rocket_cfg.spawn.spawn_path = "/World/envs/env_0/Rocket"
        self._rocket = RigidObject(rocket_cfg)
        legacy_terrain_ready = self._setup_legacy_terrain()
        if not legacy_terrain_ready:
            spawn_ground_plane(
                prim_path="/World/ground",
                cfg=GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="multiply",
                        restitution_combine_mode="multiply",
                        static_friction=0.85,
                        dynamic_friction=0.65,
                        restitution=0.02,
                    )
                ),
            )
        self._ensure_viewport_visuals()
        self._ensure_root_preview_visuals()
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        plan = cloner.ClonePlan.from_env_0(src, dest, self.scene.num_envs, self.device, pos)
        cloner.replicate(plan, stage=self.scene.stage)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.rigid_objects["rocket"] = self._rocket
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _ensure_root_preview_visuals(self) -> None:
        try:
            from pxr import Gf, UsdGeom, UsdLux
        except Exception as exc:
            print(f"[LunarRocket] root preview visual fallback unavailable: {exc}", flush=True)
            return

        self._define_preview_material("/World/Looks/PreviewRocketWhite", (0.82, 0.84, 0.82))
        self._define_preview_material("/World/Looks/PreviewRocketRed", (0.72, 0.12, 0.08))
        self._define_preview_material("/World/Looks/PreviewTerrainDark", (0.16, 0.15, 0.13))
        self._define_preview_material("/World/Looks/PreviewLandingGreen", (0.05, 0.75, 0.42))

        self._define_box_mesh(
            "/World/PreviewLandingPad_VISIBLE",
            center=(0.0, 0.0, 0.12),
            size=(14.0, 14.0, 0.18),
            material_path="/World/Looks/PreviewTerrainDark",
        )
        self._define_cylinder_mesh(
            "/World/PreviewLandingTarget_VISIBLE",
            center=(0.0, 0.0, 0.32),
            radius=0.9,
            height=0.10,
            material_path="/World/Looks/PreviewLandingGreen",
        )
        self._define_cylinder_mesh(
            "/World/PreviewRocket_VISIBLE/Body",
            center=(0.0, 0.0, 2.35),
            radius=0.38,
            height=2.6,
            material_path="/World/Looks/PreviewRocketWhite",
        )
        self._define_cone_mesh(
            "/World/PreviewRocket_VISIBLE/Nose",
            center=(0.0, 0.0, 3.95),
            radius=0.40,
            height=0.9,
            material_path="/World/Looks/PreviewRocketWhite",
        )
        self._define_cylinder_mesh(
            "/World/PreviewRocket_VISIBLE/Engine",
            center=(0.0, 0.0, 0.88),
            radius=0.23,
            height=0.35,
            material_path="/World/Looks/PreviewRocketRed",
        )

        if not self.scene.stage.GetPrimAtPath("/World/PreviewSun_VISIBLE"):
            sun = UsdLux.DistantLight.Define(self.scene.stage, "/World/PreviewSun_VISIBLE")
            sun.CreateIntensityAttr(6000.0)
            sun.CreateAngleAttr(1.0)
            sun.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))

    def _define_preview_material(self, material_path: str, color: tuple[float, float, float]) -> None:
        from pxr import Gf, Sdf, UsdShade

        material = UsdShade.Material.Define(self.scene.stage, material_path)
        shader = UsdShade.Shader.Define(self.scene.stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    def _bind_preview_material(self, prim, material_path: str) -> None:
        from pxr import UsdShade

        material = UsdShade.Material(self.scene.stage.GetPrimAtPath(material_path))
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    def _define_box_mesh(
        self,
        prim_path: str,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        material_path: str,
    ) -> None:
        from pxr import Gf, UsdGeom

        if self.scene.stage.GetPrimAtPath(prim_path):
            return
        cx, cy, cz = center
        sx, sy, sz = (0.5 * value for value in size)
        points = [
            (cx - sx, cy - sy, cz - sz),
            (cx + sx, cy - sy, cz - sz),
            (cx + sx, cy + sy, cz - sz),
            (cx - sx, cy + sy, cz - sz),
            (cx - sx, cy - sy, cz + sz),
            (cx + sx, cy - sy, cz + sz),
            (cx + sx, cy + sy, cz + sz),
            (cx - sx, cy + sy, cz + sz),
        ]
        counts = [4, 4, 4, 4, 4, 4]
        indices = [0, 1, 2, 3, 4, 7, 6, 5, 0, 4, 5, 1, 1, 5, 6, 2, 2, 6, 7, 3, 3, 7, 4, 0]
        self._define_mesh_prim(prim_path, points, counts, indices, material_path)

    def _define_cylinder_mesh(
        self,
        prim_path: str,
        center: tuple[float, float, float],
        radius: float,
        height: float,
        material_path: str,
        sides: int = 32,
    ) -> None:
        if self.scene.stage.GetPrimAtPath(prim_path):
            return
        cx, cy, cz = center
        z0 = cz - 0.5 * height
        z1 = cz + 0.5 * height
        points = []
        for z in (z0, z1):
            for index in range(sides):
                angle = 2.0 * math.pi * index / sides
                points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle), z))
        counts = []
        indices = []
        for index in range(sides):
            next_index = (index + 1) % sides
            counts.append(4)
            indices.extend([index, next_index, sides + next_index, sides + index])
        counts.extend([sides, sides])
        indices.extend(range(sides - 1, -1, -1))
        indices.extend(range(sides, 2 * sides))
        self._define_mesh_prim(prim_path, points, counts, indices, material_path)

    def _define_cone_mesh(
        self,
        prim_path: str,
        center: tuple[float, float, float],
        radius: float,
        height: float,
        material_path: str,
        sides: int = 32,
    ) -> None:
        if self.scene.stage.GetPrimAtPath(prim_path):
            return
        cx, cy, cz = center
        z0 = cz - 0.5 * height
        z1 = cz + 0.5 * height
        points = [(cx, cy, z1)]
        for index in range(sides):
            angle = 2.0 * math.pi * index / sides
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle), z0))
        counts = []
        indices = []
        for index in range(sides):
            next_index = 1 + (index + 1) % sides
            counts.append(3)
            indices.extend([0, 1 + index, next_index])
        counts.append(sides)
        indices.extend(range(sides, 0, -1))
        self._define_mesh_prim(prim_path, points, counts, indices, material_path)

    def _define_mesh_prim(
        self,
        prim_path: str,
        points: list[tuple[float, float, float]],
        counts: list[int],
        indices: list[int],
        material_path: str,
    ) -> None:
        from pxr import Gf, UsdGeom

        mesh = UsdGeom.Mesh.Define(self.scene.stage, prim_path)
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateVisibilityAttr("inherited")
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        zs = [point[2] for point in points]
        mesh.CreateExtentAttr(
            [
                Gf.Vec3f(min(xs), min(ys), min(zs)),
                Gf.Vec3f(max(xs), max(ys), max(zs)),
            ]
        )
        self._bind_preview_material(mesh.GetPrim(), material_path)

    def _ensure_viewport_visuals(self) -> None:
        try:
            from pxr import Gf, UsdGeom
        except Exception as exc:
            print(f"[LunarRocket] viewport visual fallback unavailable: {exc}", flush=True)
            return

        root_path = "/World/envs/env_0/Rocket"
        if self.scene.stage.GetPrimAtPath(root_path):
            self._define_rocket_preview_visual(root_path)
        else:
            print(f"[LunarRocket] rocket source prim missing before clone: {root_path}", flush=True)

        marker_path = "/World/envs/env_0/LandingZonePreview"
        if not self.scene.stage.GetPrimAtPath(marker_path):
            marker = UsdGeom.Cylinder.Define(self.scene.stage, marker_path)
            marker.CreateAxisAttr("Z")
            marker.CreateRadiusAttr(0.65)
            marker.CreateHeightAttr(0.035)
            marker.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.72, 0.45)])
            marker.CreateVisibilityAttr("inherited")
            marker.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.20))

        patch_path = "/World/envs/env_0/ViewportSurfacePatch"
        if not self.scene.stage.GetPrimAtPath(patch_path):
            patch = UsdGeom.Cube.Define(self.scene.stage, patch_path)
            patch.CreateSizeAttr(1.0)
            patch.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.17, 0.15)])
            patch.CreateVisibilityAttr("inherited")
            patch.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.08))
            patch.AddScaleOp().Set(Gf.Vec3f(9.0, 9.0, 0.08))

    def _define_rocket_preview_visual(self, root_path: str) -> None:
        from pxr import Gf, UsdGeom

        body_path = f"{root_path}/ViewportBody"
        if not self.scene.stage.GetPrimAtPath(body_path):
            body = UsdGeom.Cylinder.Define(self.scene.stage, body_path)
            body.CreateAxisAttr("Z")
            body.CreateRadiusAttr(0.32)
            body.CreateHeightAttr(3.0)
            body.CreateDisplayColorAttr([Gf.Vec3f(0.78, 0.80, 0.82)])
            body.CreateVisibilityAttr("inherited")
            body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.55))

        nose_path = f"{root_path}/ViewportNose"
        if not self.scene.stage.GetPrimAtPath(nose_path):
            nose = UsdGeom.Cone.Define(self.scene.stage, nose_path)
            nose.CreateAxisAttr("Z")
            nose.CreateRadiusAttr(0.34)
            nose.CreateHeightAttr(0.9)
            nose.CreateDisplayColorAttr([Gf.Vec3f(0.92, 0.93, 0.91)])
            nose.CreateVisibilityAttr("inherited")
            nose.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 3.5))

        engine_path = f"{root_path}/ViewportEngine"
        if not self.scene.stage.GetPrimAtPath(engine_path):
            engine = UsdGeom.Cylinder.Define(self.scene.stage, engine_path)
            engine.CreateAxisAttr("Z")
            engine.CreateRadiusAttr(0.20)
            engine.CreateHeightAttr(0.35)
            engine.CreateDisplayColorAttr([Gf.Vec3f(0.58, 0.12, 0.08)])
            engine.CreateVisibilityAttr("inherited")
            engine.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.10))

        for leg_name, x, y, rot_y, rot_x in (
            ("LegPX", 0.58, 0.0, 38.0, 0.0),
            ("LegNX", -0.58, 0.0, -38.0, 0.0),
            ("LegPY", 0.0, 0.58, 0.0, -38.0),
            ("LegNY", 0.0, -0.58, 0.0, 38.0),
        ):
            leg_path = f"{root_path}/Viewport{leg_name}"
            if self.scene.stage.GetPrimAtPath(leg_path):
                continue
            leg = UsdGeom.Capsule.Define(self.scene.stage, leg_path)
            leg.CreateAxisAttr("Z")
            leg.CreateRadiusAttr(0.055)
            leg.CreateHeightAttr(1.15)
            leg.CreateDisplayColorAttr([Gf.Vec3f(0.32, 0.32, 0.32)])
            leg.CreateVisibilityAttr("inherited")
            leg.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.15))
            leg.AddRotateXYZOp().Set(Gf.Vec3f(rot_x, rot_y, 0.0))

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions.copy_(self._actions)
        self._actions = actions.clone()
        self._actions[:, 0] = 0.5 * (torch.clamp(self._actions[:, 0], -1.0, 1.0) + 1.0)
        self._actions[:, 1:] = torch.clamp(self._actions[:, 1:], -1.0, 1.0)

        max_gimbal = torch.deg2rad(torch.tensor(self.cfg.max_gimbal_deg, device=self.device))
        gx = self._actions[:, 1] * max_gimbal
        gy = self._actions[:, 2] * max_gimbal
        thrust_dir_b = torch.stack(
            (
                torch.sin(gx),
                torch.sin(gy),
                torch.cos(gx) * torch.cos(gy),
            ),
            dim=-1,
        )
        thrust_dir_b = thrust_dir_b / torch.clamp(torch.linalg.norm(thrust_dir_b, dim=-1, keepdim=True), min=1e-6)
        quat = self._rocket.data.root_quat_w
        thrust_dir_w = _quat_rotate(quat, thrust_dir_b)
        force_w = thrust_dir_w * (self._actions[:, 0:1] * self.cfg.max_thrust_n)
        lever_b = torch.zeros_like(thrust_dir_b)
        lever_b[:, 2] = -float(self.cfg.thrust_lever_arm_m)
        lever_w = _quat_rotate(quat, lever_b)
        self._forces[:, 0, :] = force_w
        self._torques[:, 0, :] = torch.cross(lever_w, force_w, dim=-1)

    def _apply_action(self) -> None:
        self._rocket.permanent_wrench_composer.set_forces_and_torques(
            forces=self._forces,
            torques=self._torques,
            body_ids=self._body_ids,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        pos = self._rocket.data.root_pos_w
        quat = self._rocket.data.root_quat_w
        lin_vel = self._rocket.data.root_lin_vel_w
        ang_vel = self._rocket.data.root_ang_vel_w
        terrain_height = self._terrain_height(pos[:, :2]).unsqueeze(-1)
        altitude = torch.clamp(pos[:, 2:3] - terrain_height, min=0.0)
        delta = self._target_pos_w - pos
        terrain_metrics = self._terrain_metrics(self._target_pos_w[:, :2])

        state = torch.cat(
            (
                torch.clamp(delta / self.cfg.max_xy_m, -1.0, 1.0),
                torch.clamp(lin_vel / 20.0, -1.0, 1.0),
                quat,
                torch.clamp(ang_vel / 5.0, -1.0, 1.0),
                torch.clamp(_quat_rotate_inverse(quat, self._forces[:, 0, :] / self.cfg.rocket_mass_kg) / 30.0, -1.0, 1.0),
                torch.clamp(altitude / self.cfg.max_altitude_m, 0.0, 1.0),
                self._actions,
                self._prev_actions,
                terrain_metrics,
            ),
            dim=-1,
        )
        lidar = self._lidar_ranges(pos[:, :2], pos[:, 2])
        terrain_scan = self._terrain_scan(pos[:, :2], terrain_height.squeeze(-1))
        sensor_features = self._sensor_features(quat, ang_vel, altitude.squeeze(-1), lidar)
        return {"policy": torch.cat((state, lidar, terrain_scan, sensor_features), dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        pos = self._rocket.data.root_pos_w
        quat = self._rocket.data.root_quat_w
        lin_vel = self._rocket.data.root_lin_vel_w
        ang_vel = self._rocket.data.root_ang_vel_w

        delta_xy = self._target_pos_w[:, :2] - pos[:, :2]
        distance_xy = torch.linalg.norm(delta_xy, dim=-1)
        progress = self._previous_distance_xy - distance_xy
        self._previous_distance_xy = distance_xy.detach()

        terrain_height = self._terrain_height(pos[:, :2])
        altitude = torch.clamp(pos[:, 2] - terrain_height, min=0.0)
        near_ground = torch.exp(-altitude / 5.0)
        horizontal_speed = torch.linalg.norm(lin_vel[:, :2], dim=-1)
        vertical_speed = torch.abs(lin_vel[:, 2])
        angular_speed = torch.linalg.norm(ang_vel, dim=-1)
        up_alignment = 1.0 - 2.0 * (quat[:, 1] * quat[:, 1] + quat[:, 2] * quat[:, 2])
        tilt_penalty = 1.0 - torch.clamp(up_alignment, -1.0, 1.0)
        action_delta = torch.linalg.norm(self._actions - self._prev_actions, dim=-1)
        terrain_metrics = self._terrain_metrics(self._target_pos_w[:, :2])
        terrain_hazard = (
            0.45 * terrain_metrics[:, 0].clamp(0.0, 1.0)
            + 0.35 * terrain_metrics[:, 1].clamp(0.0, 1.0)
            + 0.20 * (1.0 - terrain_metrics[:, 2].clamp(0.0, 1.0))
        )

        reward = (
            self.cfg.rew_progress * progress
            + self.cfg.rew_target * distance_xy
            + self.cfg.rew_tilt * tilt_penalty
            + self.cfg.rew_angvel * angular_speed
            + self.cfg.rew_vxy * horizontal_speed
            + self.cfg.rew_vz * vertical_speed * near_ground
            + self.cfg.rew_fuel * self._actions[:, 0] * self._actions[:, 0]
            + self.cfg.rew_smooth * action_delta
            - terrain_hazard * torch.exp(-altitude / 10.0)
            + self.cfg.rew_time
        )

        landed = altitude < self.cfg.landing_altitude_m
        soft = landed & (distance_xy < self.cfg.target_radius_m) & (
            horizontal_speed < self.cfg.soft_horizontal_speed_mps
        ) & (vertical_speed < self.cfg.soft_vertical_speed_mps) & (
            tilt_penalty < 1.0 - torch.cos(torch.deg2rad(torch.tensor(self.cfg.soft_tilt_deg, device=self.device)))
        ) & (terrain_metrics[:, 2] > 0.65)
        self._last_landing_metrics = {
            "soft": soft.detach(),
            "landed": landed.detach(),
            "distance_xy": distance_xy.detach(),
            "horizontal_speed": horizontal_speed.detach(),
            "vertical_speed": vertical_speed.detach(),
            "tilt_penalty": tilt_penalty.detach(),
            "safe_zone_score": terrain_metrics[:, 2].detach(),
        }
        reward = reward + soft.float() * self.cfg.rew_soft_landing
        reward = reward + (landed & ~soft).float() * self.cfg.rew_harsh_landing
        reward = reward + ((pos[:, 2] - terrain_height) < -0.25).float() * self.cfg.rew_crash
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self._rocket.data.root_pos_w
        terrain_height = self._terrain_height(pos[:, :2])
        altitude = pos[:, 2] - terrain_height
        distance_xy = torch.linalg.norm(self._target_pos_w[:, :2] - pos[:, :2], dim=-1)
        out_of_bounds = torch.linalg.norm(pos[:, :2] - self.scene.env_origins[:, :2], dim=-1) > self.cfg.max_xy_m
        landed = altitude < self.cfg.landing_altitude_m
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        terminated = (
            landed
            | out_of_bounds
            | (altitude > self.cfg.max_altitude_m)
            | (distance_xy > self.cfg.max_xy_m * 1.5)
        )
        metrics = getattr(self, "_last_landing_metrics", None)
        if metrics is not None:
            log = self.extras.setdefault("log", {})
            log["Metrics/success_rate"] = metrics["soft"].float().mean().item()
            log["Metrics/landed_rate"] = metrics["landed"].float().mean().item()
            log["Metrics/crash_rate"] = (landed & ~metrics["soft"]).float().mean().item()
            log["Metrics/out_of_bounds_rate"] = out_of_bounds.float().mean().item()
            log["Metrics/timeout_rate"] = timeout.float().mean().item()
            log["Metrics/mean_landing_speed"] = metrics["vertical_speed"].mean().item()
            log["Metrics/mean_horizontal_speed"] = metrics["horizontal_speed"].mean().item()
            log["Metrics/mean_distance_xy"] = metrics["distance_xy"].mean().item()
            log["Metrics/mean_safe_zone_score"] = metrics["safe_zone_score"].mean().item()
        return terminated, timeout

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._rocket._ALL_INDICES
        super()._reset_idx(env_ids)
        self._rocket.reset(env_ids)

        if getattr(self, "_legacy_terrain", None) is not None:
            import numpy as np
            from app.randomization import sample_reset
            from dataclasses import replace

            for env_id in env_ids.tolist():
                # Generate new randomized parameters
                rng = np.random.default_rng()
                res_sample = sample_reset(self._legacy_terrain_cfg, self._legacy_rocket_cfg, rng)
                # Keep target and rocket local offset centered
                res_sample = replace(
                    res_sample,
                    rocket_position=(0.0, 0.0, 0.0),
                    target_position=(0.0, 0.0, 0.0),
                )
                
                # Generate new heights
                new_terrain = self._legacy_terrain_generator.generate(res_sample)
                
                # Update 3D PyTorch heights tensors (observations/rewards will use these)
                self._legacy_terrain_heights[env_id] = torch.as_tensor(new_terrain.heights, dtype=torch.float32, device=self.device)
                if new_terrain.local_detail_mesh is not None:
                    self._legacy_local_heights[env_id] = torch.as_tensor(new_terrain.local_detail_mesh.heights, dtype=torch.float32, device=self.device)

        count = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        root_state = self._rocket.data.default_root_state[env_ids].clone()
        xy_offset = torch.empty(count, 2, device=self.device).uniform_(
            -self.cfg.spawn_xy_range_m,
            self.cfg.spawn_xy_range_m,
        )
        self._randomize_terrain(env_ids, origins[:, :2])
        root_state[:, :2] = origins[:, :2] + xy_offset
        root_state[:, 2] = self._terrain_height(root_state[:, :2]) + self.cfg.spawn_altitude_m
        root_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)
        root_state[:, 7:] = 0.0
        self._rocket.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._rocket.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        self._target_pos_w[env_ids] = origins
        self._target_pos_w[env_ids, 2] = self._terrain_height(origins[:, :2])
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._previous_distance_xy[env_ids] = torch.linalg.norm(
            self._target_pos_w[env_ids, :2] - root_state[:, :2],
            dim=-1,
        )

    def _randomize_terrain(self, env_ids: torch.Tensor, origins_xy: torch.Tensor) -> None:
        count = len(env_ids)
        self._terrain_slope[env_ids] = torch.empty(count, 2, device=self.device).uniform_(
            -self.cfg.terrain_slope_scale,
            self.cfg.terrain_slope_scale,
        )
        self._terrain_phase[env_ids] = torch.empty(count, 2, device=self.device).uniform_(0.0, 2.0 * torch.pi)
        self._terrain_amp[env_ids] = torch.empty(count, device=self.device).uniform_(
            0.05,
            self.cfg.terrain_height_scale_m,
        )
        self._crater_xy[env_ids] = origins_xy[:, None, :] + torch.empty(
            count,
            self.cfg.terrain_crater_count,
            2,
            device=self.device,
        ).uniform_(-7.0, 7.0)
        self._crater_radius[env_ids] = torch.empty(
            count,
            self.cfg.terrain_crater_count,
            device=self.device,
        ).uniform_(1.2, 4.5)
        self._crater_depth[env_ids] = torch.empty(
            count,
            self.cfg.terrain_crater_count,
            device=self.device,
        ).uniform_(0.08, 0.35)
        self._rock_xy[env_ids] = origins_xy[:, None, :] + torch.empty(
            count,
            self.cfg.terrain_rock_count,
            2,
            device=self.device,
        ).uniform_(-6.0, 6.0)
        self._rock_radius[env_ids] = torch.empty(
            count,
            self.cfg.terrain_rock_count,
            device=self.device,
        ).uniform_(0.25, 0.65)
        self._rock_height[env_ids] = torch.empty(
            count,
            self.cfg.terrain_rock_count,
            device=self.device,
        ).uniform_(0.05, 0.22)

    def _terrain_height(self, xy_w: torch.Tensor) -> torch.Tensor:
        legacy_height = self._legacy_terrain_height(xy_w)
        if legacy_height is not None:
            return legacy_height
        env_ids = self._env_ids_from_xy(xy_w)
        origin_xy = self.scene.env_origins[env_ids, :2]
        local_xy = xy_w - origin_xy
        slope = torch.sum(self._terrain_slope[env_ids] * local_xy, dim=-1)
        ripple = self._terrain_amp[env_ids] * (
            0.55 * torch.sin(0.55 * local_xy[:, 0] + self._terrain_phase[env_ids, 0])
            + 0.45 * torch.sin(0.45 * local_xy[:, 1] + self._terrain_phase[env_ids, 1])
        )
        crater_delta = xy_w[:, None, :] - self._crater_xy[env_ids]
        crater_dist = torch.linalg.norm(crater_delta, dim=-1)
        crater_sigma = torch.clamp(self._crater_radius[env_ids] * 0.55, min=1e-3)
        crater_bowl = -self._crater_depth[env_ids] * torch.exp(-0.5 * (crater_dist / crater_sigma) ** 2)
        rim_sigma = torch.clamp(self._crater_radius[env_ids] * 0.18, min=1e-3)
        crater_rim = 0.35 * self._crater_depth[env_ids] * torch.exp(
            -0.5 * ((crater_dist - self._crater_radius[env_ids]) / rim_sigma) ** 2
        )
        rock_delta = xy_w[:, None, :] - self._rock_xy[env_ids]
        rock_dist = torch.linalg.norm(rock_delta, dim=-1)
        rock_radius = torch.clamp(self._rock_radius[env_ids], min=1e-3)
        rock_profile = self._rock_height[env_ids] * torch.clamp(1.0 - rock_dist / rock_radius, min=0.0) ** 2
        return slope + ripple + torch.sum(crater_bowl + crater_rim, dim=-1) + torch.sum(rock_profile, dim=-1)

    def _terrain_metrics(self, xy_w: torch.Tensor) -> torch.Tensor:
        eps = 0.35
        dx = torch.tensor((eps, 0.0), device=self.device).expand_as(xy_w)
        dy = torch.tensor((0.0, eps), device=self.device).expand_as(xy_w)
        h0 = self._terrain_height(xy_w)
        hx = self._terrain_height(xy_w + dx)
        hy = self._terrain_height(xy_w + dy)
        grad_x = (hx - h0) / eps
        grad_y = (hy - h0) / eps
        slope = torch.atan(torch.linalg.norm(torch.stack((grad_x, grad_y), dim=-1), dim=-1))
        roughness = torch.abs(hx - h0) + torch.abs(hy - h0)
        slope_norm = torch.clamp(torch.rad2deg(slope) / 25.0, 0.0, 1.0)
        roughness_norm = torch.clamp(roughness / 0.35, 0.0, 1.0)
        safe_zone = torch.clamp(1.0 - 0.55 * slope_norm - 0.45 * roughness_norm, 0.0, 1.0)
        target_height = torch.clamp(h0 / self.cfg.max_altitude_m, -1.0, 1.0)
        return torch.stack((slope_norm, roughness_norm, safe_zone, target_height), dim=-1)

    def _lidar_ranges(self, xy_w: torch.Tensor, z_w: torch.Tensor) -> torch.Tensor:
        angles = torch.linspace(0.0, 2.0 * torch.pi, self.cfg.lidar_ray_count + 1, device=self.device)[:-1]
        radius_scale = 1.0 + 0.35 * torch.sin(3.0 * angles)
        offsets = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1) * (2.5 * radius_scale[:, None])
        sample_xy = xy_w[:, None, :] + offsets[None, :, :]
        sample_height = self._terrain_height(sample_xy.reshape(-1, 2)).reshape(xy_w.shape[0], -1)
        ranges = (z_w[:, None] - sample_height) / self.cfg.max_lidar_distance_m
        if self.cfg.lidar_noise_std > 0.0:
            ranges = ranges + torch.randn_like(ranges) * self.cfg.lidar_noise_std
        return torch.clamp(ranges, 0.0, 1.0)

    def _terrain_scan(self, xy_w: torch.Tensor, center_height: torch.Tensor) -> torch.Tensor:
        angles = torch.linspace(0.0, 2.0 * torch.pi, self.cfg.terrain_scan_count + 1, device=self.device)[:-1]
        radius = 5.0
        offsets = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1) * radius
        sample_xy = xy_w[:, None, :] + offsets[None, :, :]
        sample_height = self._terrain_height(sample_xy.reshape(-1, 2)).reshape(xy_w.shape[0], -1)
        return torch.clamp((sample_height - center_height[:, None]) / 2.0, -1.0, 1.0)

    def _sensor_features(
        self,
        quat: torch.Tensor,
        ang_vel: torch.Tensor,
        altitude: torch.Tensor,
        lidar: torch.Tensor,
    ) -> torch.Tensor:
        mass = float(self.cfg.rocket_mass_kg)
        accel_body = _quat_rotate_inverse(quat, self._forces[:, 0, :] / max(mass, 1e-6)) / 30.0
        gyro = ang_vel / 5.0
        altimeter = altitude[:, None] / self.cfg.max_lidar_distance_m
        lidar_min = torch.amin(lidar, dim=-1, keepdim=True)
        if self.cfg.imu_accel_noise_std > 0.0:
            accel_body = accel_body + torch.randn_like(accel_body) * self.cfg.imu_accel_noise_std
        if self.cfg.imu_gyro_noise_std > 0.0:
            gyro = gyro + torch.randn_like(gyro) * self.cfg.imu_gyro_noise_std
        if self.cfg.altimeter_noise_std > 0.0:
            altimeter = altimeter + torch.randn_like(altimeter) * self.cfg.altimeter_noise_std
        return torch.cat(
            (
                torch.clamp(accel_body, -1.0, 1.0),
                torch.clamp(gyro, -1.0, 1.0),
                torch.clamp(altimeter, 0.0, 1.0),
                torch.clamp(lidar_min, 0.0, 1.0),
            ),
            dim=-1,
        )

    def _env_ids_from_xy(self, xy_w: torch.Tensor) -> torch.Tensor:
        if self.scene.env_origins.shape[0] == xy_w.shape[0]:
            return torch.arange(xy_w.shape[0], device=self.device)
        distance = torch.cdist(xy_w, self.scene.env_origins[:, :2])
        return torch.argmin(distance, dim=-1)

    def _setup_legacy_terrain(self) -> bool:
        try:
            import numpy as np

            from app.config import load_yaml
            from app.randomization import sample_reset
            from app.terrain_generator import MoonTerrainGenerator
        except Exception as exc:
            print(f"[LunarRocket] legacy terrain unavailable, using procedural fallback: {exc}", flush=True)
            return False

        try:
            root = _project_root()
            terrain_cfg = load_yaml(root / self.cfg.legacy_terrain_config_path)
            rocket_cfg = load_yaml(root / self.cfg.legacy_rocket_config_path)
            terrain_cfg = dict(terrain_cfg)
            terrain_cfg["terrain_quality"] = os.environ.get(
                "ISAACLAB_TERRAIN_QUALITY",
                terrain_cfg.get("terrain_quality", "high_detail_landing"),
            )
            local_detail_cfg = dict(terrain_cfg.get("local_detail_patch", {}))
            local_detail_cfg["resolution"] = int(os.environ.get("ISAACLAB_TERRAIN_LOCAL_RESOLUTION", 512))
            if "ISAACLAB_TERRAIN_LOCAL_DETAIL" in os.environ:
                local_detail_cfg["enabled"] = os.environ["ISAACLAB_TERRAIN_LOCAL_DETAIL"].lower() not in {"0", "false", "no"}
            terrain_cfg["local_detail_patch"] = local_detail_cfg

            rng = np.random.default_rng(int(terrain_cfg.get("seed", 7)))
            reset_sample = sample_reset(terrain_cfg, rocket_cfg, rng)
            reset_sample = replace(
                reset_sample,
                rocket_position=(0.0, 0.0, 0.0),
                target_position=(0.0, 0.0, 0.0),
            )
            generator = MoonTerrainGenerator(terrain_cfg)
            terrain = generator.generate(reset_sample)
            self._legacy_terrain_generator = generator
            self._legacy_terrain_cfg = terrain_cfg
            self._legacy_rocket_cfg = rocket_cfg
            self._legacy_terrain = terrain
            
            # Setup 3D tensor to support per-env dynamic heightmaps
            self._legacy_terrain_heights = torch.zeros(
                (self.num_envs, terrain.heights.shape[0], terrain.heights.shape[1]),
                dtype=torch.float32,
                device=self.device
            )
            self._legacy_terrain_heights[:] = torch.as_tensor(terrain.heights, dtype=torch.float32, device=self.device)
            self._legacy_terrain_size = terrain.size_m
            min_x, max_x, min_y, max_y = _mesh_bounds(terrain.vertices)
            self._legacy_terrain_min_xy = (min_x, min_y)
            self._legacy_terrain_max_xy = (max_x, max_y)
            if terrain.local_detail_mesh is not None:
                local = terrain.local_detail_mesh
                self._legacy_local_heights = torch.zeros(
                    (self.num_envs, local.heights.shape[0], local.heights.shape[1]),
                    dtype=torch.float32,
                    device=self.device
                )
                self._legacy_local_heights[:] = torch.as_tensor(local.heights, dtype=torch.float32, device=self.device)
                local_min_x, local_max_x, local_min_y, local_max_y = _mesh_bounds(local.vertices)
                self._legacy_local_min_xy = (local_min_x, local_min_y)
                self._legacy_local_max_xy = (local_max_x, local_max_y)
            else:
                self._legacy_local_heights = None
                self._legacy_local_min_xy = None
                self._legacy_local_max_xy = None

            if self.num_envs <= int(self.cfg.legacy_terrain_usd_max_envs):
                generator.create_or_update_usd_meshes(
                    self.scene.stage,
                    "/World/envs/env_0/MoonTerrain",
                    terrain,
                )
                self._validate_legacy_visual_prims()
                self._spawn_legacy_rocks(terrain_cfg, reset_sample)
                print(
                    "[LunarRocket] IsaacLab terrain mesh created from app.MoonTerrainGenerator "
                    f"({terrain.heightmap_resolution}x{terrain.heightmap_resolution}, "
                    f"{terrain.triangle_count:,} base triangles, "
                    f"local_detail={terrain.local_detail_mesh.heightmap_resolution if terrain.local_detail_mesh else 0})",
                    flush=True,
                )
                return True

            if os.environ.get("ISAACLAB_FAST_APPROX_TERRAIN", "0").lower() not in {"1", "true", "yes"}:
                raise RuntimeError(
                    "Real USD terrain/collision is enabled only up to "
                    f"{self.cfg.legacy_terrain_usd_max_envs} envs. "
                    "Use ISAACLAB_NUM_ENVS<=16 for real terrain training, or set "
                    "ISAACLAB_FAST_APPROX_TERRAIN=1 to opt into the fast approximate mode "
                    "where observations/rewards sample the DEM but physical collision falls back to a plane."
                )

            print(
                "[LunarRocket] loaded MoonTerrainGenerator heightfield for observations/reward; "
                "USD terrain mesh disabled for high env count.",
                flush=True,
            )
            return False
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[LunarRocket] legacy terrain setup failed, using procedural fallback: {exc}", flush=True)
            self._legacy_terrain = None
            self._legacy_terrain_heights = None
            return False

    def _validate_legacy_visual_prims(self) -> None:
        try:
            from pxr import UsdGeom
        except Exception as exc:
            raise RuntimeError("USD geometry validation requires pxr.UsdGeom") from exc

        required = ["/World/envs/env_0/MoonTerrain"]
        if getattr(self._legacy_terrain, "local_detail_mesh", None) is not None:
            required.append("/World/envs/env_0/MoonTerrain_LandingDetail")

        for prim_path in required:
            prim = self.scene.stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"Required visual terrain prim is missing: {prim_path}")
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            face_counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if not points or not face_counts or not indices:
                raise RuntimeError(f"Terrain mesh has empty geometry arrays: {prim_path}")
            extent = mesh.GetExtentAttr().Get()
            print(
                f"[LunarRocket] validated visual mesh {prim_path}: "
                f"points={len(points):,}, faces={len(face_counts):,}, indices={len(indices):,}, extent={extent}",
                flush=True,
            )

    def _spawn_legacy_rocks(self, terrain_cfg: dict, reset_sample) -> None:
        try:
            import random

            from pxr import Gf, Sdf, UsdGeom, UsdPhysics
        except Exception:
            return

        rocks_cfg = terrain_cfg.get("rocks", {})
        if not bool(rocks_cfg.get("enabled", True)):
            return

        rocks_dir = _project_root() / "assets/rocks/small_rocks"
        rock_files = sorted(rocks_dir.glob("rock_*.usd"))
        if not rock_files:
            return

        total_count = int(rocks_cfg.get("total_count", 35))
        landing_zone_percentage = float(rocks_cfg.get("landing_zone_percentage", 0.75))
        min_distance = float(rocks_cfg.get("min_distance_from_center_m", 2.0))
        landing_radius = float(rocks_cfg.get("landing_zone_radius_m", 12.0))
        outer_radius = float(rocks_cfg.get("outer_radius_m", 35.0))
        scale_min, scale_max = [float(value) for value in rocks_cfg.get("scale_range", [0.12, 0.55])]
        collision_enabled = bool(rocks_cfg.get("collision_enabled", True))
        rng = random.Random(int(reset_sample.seed))

        group_path = "/World/envs/env_0/Rocks"
        UsdGeom.Xform.Define(self.scene.stage, group_path)
        center_x, center_y = reset_sample.target_position[:2]

        for rock_id in range(total_count):
            if rng.random() < landing_zone_percentage:
                radius = rng.uniform(min_distance, landing_radius)
            else:
                radius = rng.uniform(landing_radius, outer_radius)
            angle = rng.uniform(0.0, 2.0 * math.pi)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            z = float(self._legacy_height_at_local_xy(x, y)) + 0.015
            scale = rng.uniform(scale_min, scale_max)

            rock_path = f"{group_path}/Rock_{rock_id}"
            rock_prim = self.scene.stage.DefinePrim(rock_path)
            rock_prim.GetReferences().AddReference(str(rng.choice(rock_files)))
            xformable = UsdGeom.Xformable(rock_prim)
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
            xformable.AddRotateXYZOp().Set(
                Gf.Vec3f(
                    rng.uniform(-8.0, 8.0),
                    rng.uniform(-8.0, 8.0),
                    rng.uniform(0.0, 360.0),
                )
            )
            xformable.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
            if collision_enabled:
                UsdPhysics.CollisionAPI.Apply(rock_prim)
                rock_prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token).Set("convexHull")
                rock_prim.CreateAttribute("physxCollision:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)

    def _legacy_height_at_local_xy(self, x: float, y: float) -> float:
        heights = self._legacy_heights_for_local_xy(x, y)
        if heights is None:
            return 0.0

        min_xy, max_xy = self._legacy_bounds_for_local_xy(x, y)
        min_x, min_y = min_xy
        max_x, max_y = max_xy
        
        heights_2d = heights[0] if heights.ndim == 3 else heights
        rows, cols = heights_2d.shape
        col_f = torch.clamp(
            torch.tensor((x - min_x) / max(max_x - min_x, 1e-6) * (cols - 1), device=self.device),
            0.0,
            cols - 1,
        )
        row_f = torch.clamp(
            torch.tensor((y - min_y) / max(max_y - min_y, 1e-6) * (rows - 1), device=self.device),
            0.0,
            rows - 1,
        )
        c0 = int(torch.floor(col_f).item())
        r0 = int(torch.floor(row_f).item())
        c1 = min(cols - 1, c0 + 1)
        r1 = min(rows - 1, r0 + 1)
        cf = float((col_f - c0).item())
        rf = float((row_f - r0).item())
        top = float(heights_2d[r0, c0].item()) * (1.0 - cf) + float(heights_2d[r0, c1].item()) * cf
        bottom = float(heights_2d[r1, c0].item()) * (1.0 - cf) + float(heights_2d[r1, c1].item()) * cf
        return top * (1.0 - rf) + bottom * rf

    def _legacy_heights_for_local_xy(self, x: float, y: float) -> torch.Tensor | None:
        local_heights = getattr(self, "_legacy_local_heights", None)
        local_min = getattr(self, "_legacy_local_min_xy", None)
        local_max = getattr(self, "_legacy_local_max_xy", None)
        if local_heights is not None and local_min is not None and local_max is not None:
            if local_min[0] <= x <= local_max[0] and local_min[1] <= y <= local_max[1]:
                return local_heights
        return getattr(self, "_legacy_terrain_heights", None)

    def _legacy_bounds_for_local_xy(self, x: float, y: float) -> tuple[tuple[float, float], tuple[float, float]]:
        local_heights = getattr(self, "_legacy_local_heights", None)
        local_min = getattr(self, "_legacy_local_min_xy", None)
        local_max = getattr(self, "_legacy_local_max_xy", None)
        if local_heights is not None and local_min is not None and local_max is not None:
            if local_min[0] <= x <= local_max[0] and local_min[1] <= y <= local_max[1]:
                return local_min, local_max
        return self._legacy_terrain_min_xy, self._legacy_terrain_max_xy

    def _legacy_terrain_height(self, xy_w: torch.Tensor) -> torch.Tensor | None:
        heights = getattr(self, "_legacy_terrain_heights", None)
        if heights is None:
            return None

        env_ids = self._env_ids_from_xy(xy_w)
        local_xy = xy_w - self.scene.env_origins[env_ids, :2]
        base_height = self._sample_height_tensor(
            heights,
            local_xy,
            self._legacy_terrain_min_xy,
            self._legacy_terrain_max_xy,
            env_ids,
        )

        local_heights = getattr(self, "_legacy_local_heights", None)
        local_min = getattr(self, "_legacy_local_min_xy", None)
        local_max = getattr(self, "_legacy_local_max_xy", None)
        if local_heights is None or local_min is None or local_max is None:
            return base_height

        in_local = (
            (local_xy[:, 0] >= local_min[0])
            & (local_xy[:, 0] <= local_max[0])
            & (local_xy[:, 1] >= local_min[1])
            & (local_xy[:, 1] <= local_max[1])
        )
        if not torch.any(in_local):
            return base_height

        local_height = self._sample_height_tensor(local_heights, local_xy, local_min, local_max, env_ids)
        return torch.where(in_local, local_height, base_height)

    def _sample_height_tensor(
        self,
        heights: torch.Tensor,
        local_xy: torch.Tensor,
        min_xy: tuple[float, float],
        max_xy: tuple[float, float],
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        min_x, min_y = min_xy
        max_x, max_y = max_xy
        if heights.ndim == 3:
            num_envs, rows, cols = heights.shape
        else:
            rows, cols = heights.shape
        col_f = torch.clamp((local_xy[:, 0] - min_x) / max(max_x - min_x, 1e-6) * (cols - 1), 0.0, cols - 1)
        row_f = torch.clamp((local_xy[:, 1] - min_y) / max(max_y - min_y, 1e-6) * (rows - 1), 0.0, rows - 1)
        c0 = torch.floor(col_f).long()
        r0 = torch.floor(row_f).long()
        c1 = torch.clamp(c0 + 1, max=cols - 1)
        r1 = torch.clamp(r0 + 1, max=rows - 1)
        cf = (col_f - c0.float()).clamp(0.0, 1.0)
        rf = (row_f - r0.float()).clamp(0.0, 1.0)

        if heights.ndim == 3:
            top = heights[env_ids, r0, c0] * (1.0 - cf) + heights[env_ids, r0, c1] * cf
            bottom = heights[env_ids, r1, c0] * (1.0 - cf) + heights[env_ids, r1, c1] * cf
        else:
            top = heights[r0, c0] * (1.0 - cf) + heights[r0, c1] * cf
            bottom = heights[r1, c0] * (1.0 - cf) + heights[r1, c1] * cf
        return top * (1.0 - rf) + bottom * rf


def _quat_rotate(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    q_vec = quat_wxyz[:, 1:4]
    q_w = quat_wxyz[:, 0:1]
    t = 2.0 * torch.cross(q_vec, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_vec, t, dim=-1)


def _quat_rotate_inverse(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    inv = torch.cat((quat_wxyz[:, 0:1], -quat_wxyz[:, 1:4]), dim=-1)
    return _quat_rotate(inv, vec)


def _mesh_bounds(vertices: list[tuple[float, float, float]]) -> tuple[float, float, float, float]:
    xs = [vertex[0] for vertex in vertices]
    ys = [vertex[1] for vertex in vertices]
    return min(xs), max(xs), min(ys), max(ys)
