from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace

import math
import numpy as np
import torch

from app.terrain_pool import generate_terrain_pool_batch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs.direct_rl_env import DirectRLEnv

# Isaac Lab task convention: the *_env_cfg module holds only Cfg dataclasses
# (isaaclab.assets.ArticulationCfg, isaaclab.envs.DirectRLEnvCfg, etc.), never
# the runtime env class. gym.register's env_cfg_entry_point resolves this
# module alone (see __init__.py), which must succeed before SimulationApp
# exists. This module (lunar_lander_env), by contrast, imports DirectRLEnv
# and Articulation -- runtime classes that pull in isaaclab.sim.SimulationContext
# and therefore pxr -- and must stay out of the env_cfg_entry_point's import
# chain, or resolving the cfg crashes with "No module named 'pxr'" /
# "must take place after SimulationApp has been instantiated" before Kit
# ever starts.
from .lunar_lander_env_cfg import LunarLanderEnvCfg, _project_root


class LunarLanderEnv(DirectRLEnv):
    cfg: LunarLanderEnvCfg

    def __init__(self, cfg: LunarLanderEnvCfg, render_mode: str | None = None, **kwargs):
        # DirectRLEnv performs the first reset inside super().__init__, so these
        # curriculum counters must exist before entering the base constructor.
        self._curriculum_level = float(cfg.curriculum_initial_difficulty)
        self._curriculum_window_count = 0
        self._curriculum_window_successes = 0
        # Scale the window with num_envs: cfg.curriculum_window_episodes alone
        # is satisfied by a handful of reset calls at high env counts (see the
        # cfg field's comment), which let difficulty race far ahead of what the
        # policy had actually learned.
        self._curriculum_window_target = max(
            int(cfg.curriculum_window_episodes), 4 * int(cfg.scene.num_envs)
        )
        # B2 ablation switch (see _get_observations). A plain bool with no
        # dependency on self.num_envs/self.device, so it costs nothing to set
        # here rather than after super().__init__() alongside the tensors below.
        self._ablate_lidar = os.environ.get("ISAACLAB_ABLATE_LIDAR", "0").lower() not in {"0", "false", "no", ""}
        super().__init__(cfg, render_mode, **kwargs)
        if getattr(self, "_video_camera", None) is not None:
            self._video_camera.set_world_poses_from_view(
                np.asarray([[6.0, -7.0, 5.0]], dtype=np.float32),
                np.asarray([[0.0, 0.0, 1.5]], dtype=np.float32),
            )
        self._actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._difficulty = torch.zeros(self.num_envs, device=self.device)
        self._rocket_mass_kg = torch.full((self.num_envs,), float(cfg.rocket_mass_kg), device=self.device)
        self._max_thrust_n = torch.full((self.num_envs,), float(cfg.max_thrust_n), device=self.device)
        self._previous_distance_xy = torch.zeros(self.num_envs, device=self.device)
        self._previous_altitude = torch.zeros(self.num_envs, device=self.device)
        self._body_ids = self._rocket.find_bodies("hopper")[0]
        if len(self._body_ids) != 1:
            raise RuntimeError(f"Expected one XML hopper root body, found body ids: {self._body_ids}")
        self._joint_position_targets = torch.zeros(
            self.num_envs,
            self._rocket.num_joints,
            dtype=torch.float32,
            device=self.device,
        )
        self._yaw_joint_ids = self._rocket.find_joints("tvc_yaw_joint")[0]
        self._pitch_joint_ids = self._rocket.find_joints("tvc_pitch_joint")[0]
        self._foot_offsets_b = torch.tensor(
            self.cfg.rocket_foot_offsets_m,
            dtype=torch.float32,
            device=self.device,
        )
        self._terrain_slope = torch.zeros(self.num_envs, 2, device=self.device)
        self._terrain_phase = torch.zeros(self.num_envs, 2, device=self.device)
        self._terrain_amp = torch.full((self.num_envs,), 0.12, device=self.device)
        self._terrain_detail_phase = torch.zeros(self.num_envs, 4, device=self.device)
        self._terrain_detail_amp = torch.zeros(self.num_envs, device=self.device)
        self._terrain_center_height = torch.zeros(self.num_envs, device=self.device)
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
        self._terrain_blend_method = os.environ.get("ISAACLAB_TERRAIN_BLEND_METHOD", self.cfg.terrain_blend_method).lower()
        if self._terrain_blend_method not in {"smoothstep", "gaussian", "hybrid"}:
            raise ValueError(
                f"ISAACLAB_TERRAIN_BLEND_METHOD must be one of smoothstep|gaussian|hybrid, got {self._terrain_blend_method!r}"
            )
        self._terrain_hybrid_contact_radius_m = float(
            os.environ.get("ISAACLAB_TERRAIN_HYBRID_CONTACT_RADIUS_M", self.cfg.terrain_hybrid_contact_radius_m)
        )
        # Recorded back onto cfg so params/env.yaml (dumped by train_sac.py)
        # captures which blend method this run actually used, not just the
        # class default -- needed to tell the 3 comparison runs' logs apart.
        self.cfg.terrain_blend_method = self._terrain_blend_method
        self.cfg.terrain_hybrid_contact_radius_m = self._terrain_hybrid_contact_radius_m
        print(f"[LunarRocket] terrain fine-detail blend method: {self._terrain_blend_method}", flush=True)
        self._terrain_pool_size = max(2, int(os.environ.get("ISAACLAB_TERRAIN_POOL_SIZE", self.cfg.terrain_pool_size)))
        self._terrain_pool_refresh_assignments = max(
            self._terrain_pool_size,
            int(
                os.environ.get(
                    "ISAACLAB_TERRAIN_POOL_REFRESH_ASSIGNMENTS",
                    self.cfg.terrain_pool_refresh_assignments,
                )
            ),
        )
        self._terrain_pool_generation = 0
        self._terrain_pool_assignments = 0
        self._terrain_pool_slot_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        base_seed = int(self.cfg.seed if self.cfg.seed is not None else 42)
        self._terrain_pool_next_seed = base_seed + 1
        self._terrain_pool_dem_generator = None
        self._terrain_pool_dem_cfg = None
        self._terrain_pool_rocket_cfg = None
        self._terrain_pool_base_min_xy = (-40.0, -40.0)
        self._terrain_pool_base_max_xy = (40.0, 40.0)
        self._setup_terrain_pool_dem()
        self._terrain_pool_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lunar-terrain-refill")
        initial_pool = self._generate_terrain_pool(base_seed)
        self._terrain_pool = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in initial_pool.items()
        }
        base_rows, base_cols = initial_pool["base_height"].shape[-2:]
        self._terrain_base_heights = torch.zeros(
            self.num_envs,
            base_rows,
            base_cols,
            dtype=torch.float32,
            device=self.device,
        )
        self._terrain_pool_future: Future[dict[str, np.ndarray]] | None = None
        self._schedule_terrain_pool_refill()
        print(
            f"[LunarRocket] preloaded {self._terrain_pool_size} GPU terrain slots; "
            f"NASA_DEM={'on' if self._terrain_pool_dem_generator is not None else 'off'}, "
            "background refill enabled",
            flush=True,
        )
        self._episode_reward_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "xy_progress",
                "altitude_progress",
                "stability",
                "velocity",
                "guidance",
                "control",
                "terrain",
                "time",
                "terminal",
                "total",
            )
        }

    def close(self):
        executor = getattr(self, "_terrain_pool_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            self._terrain_pool_executor = None
        return super().close()

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Return the dedicated Isaac camera frame for reliable headless video."""
        camera = getattr(self, "_video_camera", None)
        if self.render_mode == "rgb_array" and camera is not None:
            rgb = camera.data.output.get("rgb")
            if rgb is not None and rgb.numel() > 0:
                frame = rgb[0, :, :, :3].detach().cpu().numpy()
                if int(frame.max()) - int(frame.min()) > 8:
                    self._last_video_frame = frame.copy()
                cached = getattr(self, "_last_video_frame", None)
                if cached is not None:
                    return cached
                return frame
        return super().render(recompute=recompute)

    def _setup_scene(self):
        from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

        rocket_cfg = self.cfg.rocket.copy()
        if rocket_cfg.spawn is not None:
            rocket_cfg.spawn = rocket_cfg.spawn.copy()
            rocket_cfg.spawn.spawn_path = "/World/envs/env_0/Rocket"
        self._rocket = Articulation(rocket_cfg)
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
                translation=(0.0, 0.0, -100.0),
            )
        self._ensure_viewport_visuals()
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        # Homogeneous (single-prototype) clone: matches the pattern InteractiveScene
        # itself uses for its own env_0 -> env_N replication (isaaclab.scene.
        # interactive_scene.InteractiveScene.__init__). ClonePlan.from_env_0 /
        # cloner.replicate(plan, stage=...) were removed upstream; usd_replicate
        # takes the flat args directly instead of a ClonePlan.
        env_ids = torch.arange(self.scene.num_envs, dtype=torch.long, device=self.device)
        pos, quat = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)
        homo_mask = torch.ones((1, self.scene.num_envs), device=self.device, dtype=torch.bool)
        with cloner.disabled_fabric_change_notifies(self.scene.stage, restore=False):
            cloner.usd_replicate(self.scene.stage, [src], [dest], env_ids, homo_mask, pos, quat)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["rocket"] = self._rocket
        self._video_camera = None
        self._last_video_frame = None
        if os.environ.get("ISAACLAB_ENABLE_VIDEO_CAMERA", "0").lower() in {"1", "true", "yes"}:
            from isaaclab.sensors import Camera, CameraCfg

            width = int(os.environ.get("ISAACLAB_VIDEO_WIDTH", "854"))
            height = int(os.environ.get("ISAACLAB_VIDEO_HEIGHT", "480"))
            capture_stride = max(1, int(os.environ.get("ISAACLAB_VIDEO_CAPTURE_STRIDE", "8")))
            camera_cfg = CameraCfg(
                prim_path="/World/LunarRocketVideoCamera",
                update_period=capture_stride * self.cfg.decimation * self.cfg.sim.dt,
                height=height,
                width=width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    horizontal_aperture=32.0,
                    clipping_range=(0.1, 200.0),
                ),
            )
            self._video_camera = Camera(camera_cfg)
            self.scene.sensors["video_camera"] = self._video_camera
        # Opt-in low-angle "sun" (directional light with real shadows) for
        # visual capture only -- the dome light below is shadow-free by
        # design (uniform training signal, not photographed), so screenshots
        # look flat/washed-out without it. Off by default; never affects
        # training since nobody sets this env var during a training run.
        # When enabled, the dome light is cut way down (fill only) so the
        # sun -- not ambient light -- sets the image's dynamic range;
        # otherwise the dome alone blows out highlights (measured: >80% of
        # pixels above 240/255 with the training-default dome intensity),
        # and no amount of post-hoc contrast stretching recovers detail
        # that was clipped at render time.
        add_sun = bool(os.environ.get("ISAACLAB_ADD_SUN_LIGHT"))
        dome_intensity = float(os.environ.get("ISAACLAB_DOME_LIGHT_INTENSITY") or ("150.0" if add_sun else "2000.0"))
        light_cfg = sim_utils.DomeLightCfg(intensity=dome_intensity, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        if add_sun:
            import math as _math

            sun_intensity = float(os.environ.get("ISAACLAB_SUN_INTENSITY") or "2500.0")
            sun_cfg = sim_utils.DistantLightCfg(intensity=sun_intensity, color=(1.0, 0.96, 0.88), angle=0.53)
            half = _math.radians(float(os.environ.get("ISAACLAB_SUN_TILT_DEG") or "65.0")) / 2.0
            sun_cfg.func("/World/SunLight", sun_cfg, orientation=(_math.sin(half), 0.0, 0.0, _math.cos(half)))

    def _ensure_viewport_visuals(self) -> None:
        root_path = "/World/envs/env_0/Rocket"
        body_path = f"{root_path}/Geometry/hopper/main_body"
        if not self.scene.stage.GetPrimAtPath(body_path):
            raise RuntimeError(
                "The hopper_lunar.xml-derived USD body is missing from the scene: "
                f"{body_path}"
            )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self.cfg.use_residual_action:
            raise NotImplementedError(
                "use_residual_action=True has no analytic guidance term implemented yet "
                "(see the cfg field's comment) -- keep it False until that lands and has "
                "been smoke-tested on GPU."
            )
        self._prev_actions.copy_(self._actions)
        self._actions = actions.clone()
        raw_throttle = torch.clamp(self._actions[:, 0], -1.0, 1.0)
        # Per-env mass/thrust (see physics_randomization_enabled): hover_throttle
        # is now a [num_envs] tensor so a heavier or weaker-engined draw still
        # hovers at raw_throttle == 0 instead of drifting off the hover point.
        hover_throttle = self._rocket_mass_kg * abs(float(self.cfg.sim.gravity[2])) / self._max_thrust_n
        # SAC policies initialize around zero.  Make zero command hover instead
        # of mapping it to 50% thrust (5.7x lunar hover thrust for this rocket).
        self._actions[:, 0] = torch.where(
            raw_throttle <= 0.0,
            hover_throttle * (raw_throttle + 1.0),
            hover_throttle
            + (self.cfg.max_commanded_throttle - hover_throttle) * raw_throttle,
        )
        self._actions[:, 1:] = torch.clamp(self._actions[:, 1:], -1.0, 1.0)

        # The external thrust follows the *measured* two-axis TVC pose from
        # hopper_lunar.xml, rather than teleporting to the requested action.
        # This retains the authored servo stiffness/damping and keeps the
        # rendered nozzle aligned with the force used by physics.
        yaw_x = self._rocket.data.joint_pos[:, self._yaw_joint_ids].squeeze(-1)
        pitch_y = self._rocket.data.joint_pos[:, self._pitch_joint_ids].squeeze(-1)
        thrust_dir_b = torch.stack(
            (
                torch.sin(pitch_y),
                -torch.sin(yaw_x) * torch.cos(pitch_y),
                torch.cos(yaw_x) * torch.cos(pitch_y),
            ),
            dim=-1,
        )
        thrust_dir_b = thrust_dir_b / torch.clamp(torch.linalg.norm(thrust_dir_b, dim=-1, keepdim=True), min=1e-6)
        quat = self._rocket.data.root_quat_w
        thrust_dir_w = _quat_rotate(quat, thrust_dir_b)
        force_w = thrust_dir_w * (self._actions[:, 0:1] * self._max_thrust_n.unsqueeze(-1))
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
            is_global=True,
        )
        gimbal_frac = torch.clamp(
            self._difficulty / max(self.cfg.curriculum_full_gimbal_difficulty, 1e-6), 0.0, 1.0
        )
        max_gimbal = torch.deg2rad(
            self.cfg.policy_min_gimbal_deg
            + gimbal_frac * (self.cfg.policy_max_gimbal_deg - self.cfg.policy_min_gimbal_deg)
        ).unsqueeze(-1)
        self._joint_position_targets.zero_()
        self._joint_position_targets[:, self._yaw_joint_ids] = (
            self._actions[:, 1:2] * max_gimbal
        )
        self._joint_position_targets[:, self._pitch_joint_ids] = (
            self._actions[:, 2:3] * max_gimbal
        )
        self._rocket.set_joint_position_target_index(target=self._joint_position_targets)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        pos = self._rocket.data.root_pos_w
        quat = self._rocket.data.root_quat_w
        lin_vel = self._rocket.data.root_lin_vel_w
        ang_vel = self._rocket.data.root_ang_vel_w
        foot_clearances = self._foot_clearances(pos, quat)
        altitude = torch.clamp(torch.amin(foot_clearances, dim=-1, keepdim=True), min=0.0)
        delta = self._target_pos_w - pos
        terrain_metrics = self._terrain_metrics(self._target_pos_w[:, :2])

        state = torch.cat(
            (
                torch.clamp(delta / self.cfg.max_xy_m, -1.0, 1.0),
                torch.clamp(lin_vel / 20.0, -1.0, 1.0),
                quat,
                torch.clamp(ang_vel / 5.0, -1.0, 1.0),
                torch.clamp(
                    _quat_rotate_inverse(quat, self._forces[:, 0, :] / self._rocket_mass_kg.unsqueeze(-1)) / 30.0,
                    -1.0,
                    1.0,
                ),
                torch.clamp(altitude / self.cfg.max_altitude_m, 0.0, 1.0),
                self._actions,
                self._prev_actions,
                terrain_metrics,
            ),
            dim=-1,
        )
        lidar = self._lidar_ranges(pos[:, :2], pos[:, 2])
        if self._ablate_lidar:
            # B2 ablation run (ISAACLAB_ABLATE_LIDAR=1): zero the lidar channel
            # instead of shrinking observation_space, so the architecture stays
            # identical to the baseline and only the information content
            # changes. sensor_features derives lidar_min from this same
            # tensor, so zeroing here removes the signal from both places.
            lidar = torch.zeros_like(lidar)
        center_height = self._terrain_height(pos[:, :2])
        terrain_scan = self._terrain_scan(pos[:, :2], center_height)
        sensor_features = self._sensor_features(quat, ang_vel, altitude.squeeze(-1), lidar)
        return {"policy": torch.cat((state, lidar, terrain_scan, sensor_features), dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        pos = self._rocket.data.root_pos_w
        quat = self._rocket.data.root_quat_w
        lin_vel = self._rocket.data.root_lin_vel_w
        ang_vel = self._rocket.data.root_ang_vel_w

        delta_xy = self._target_pos_w[:, :2] - pos[:, :2]
        distance_xy = torch.linalg.norm(delta_xy, dim=-1)
        xy_progress = self._previous_distance_xy - distance_xy
        self._previous_distance_xy = distance_xy.detach()

        foot_clearances = self._foot_clearances(pos, quat)
        altitude = torch.clamp(torch.amin(foot_clearances, dim=-1), min=0.0)
        foot_clearance_spread = torch.amax(foot_clearances, dim=-1) - torch.amin(
            foot_clearances, dim=-1
        )
        altitude_progress = self._previous_altitude - altitude
        self._previous_altitude = altitude.detach()
        near_ground = torch.exp(-altitude / 5.0)
        horizontal_speed = torch.linalg.norm(lin_vel[:, :2], dim=-1)
        desired_horizontal_speed = torch.clamp(
            0.25 + 0.15 * altitude,
            max=1.5,
        )
        desired_horizontal_velocity = (
            delta_xy
            / torch.clamp(distance_xy.unsqueeze(-1), min=1e-4)
            * torch.minimum(0.35 * distance_xy, desired_horizontal_speed).unsqueeze(-1)
        )
        horizontal_velocity_error = torch.linalg.norm(
            lin_vel[:, :2] - desired_horizontal_velocity,
            dim=-1,
        )
        vertical_speed = torch.abs(lin_vel[:, 2])
        desired_vertical_speed = -torch.clamp(0.35 + 0.11 * altitude, max=3.0)
        vertical_speed_error = torch.abs(lin_vel[:, 2] - desired_vertical_speed)
        angular_speed = torch.linalg.norm(ang_vel, dim=-1)
        tilt_penalty, _ = self._tilt_against_terrain(quat, near_ground)
        action_delta = torch.linalg.norm(self._actions - self._prev_actions, dim=-1)
        desired_net_acceleration = torch.zeros_like(lin_vel)
        desired_net_acceleration[:, :2] = torch.clamp(
            0.20 * delta_xy - 0.60 * lin_vel[:, :2],
            min=-1.0,
            max=1.0,
        )
        desired_net_acceleration[:, 2] = torch.clamp(
            desired_vertical_speed - lin_vel[:, 2],
            min=-3.0,
            max=3.0,
        )
        desired_thrust_acceleration = desired_net_acceleration.clone()
        desired_thrust_acceleration[:, 2] -= float(self.cfg.sim.gravity[2])
        actual_thrust_acceleration = self._forces[:, 0, :] / self._rocket_mass_kg.unsqueeze(-1)
        # Horizontal gimbal force initially points opposite to the body tilt it
        # creates (the engine is below the CoM).  Only supervise vertical
        # braking here; horizontal credit comes from target progress.
        guidance_error = torch.abs(
            actual_thrust_acceleration[:, 2] - desired_thrust_acceleration[:, 2]
        )
        dt = float(self.step_dt)
        terms = {
            "xy_progress": self.cfg.rew_progress * xy_progress,
            "altitude_progress": self.cfg.rew_altitude_progress * altitude_progress,
            "stability": dt * (
                self.cfg.rew_tilt * tilt_penalty * (0.2 + 0.8 * near_ground)
                + self.cfg.rew_angvel * angular_speed
            ),
            "velocity": dt * (
                self.cfg.rew_vxy
                * horizontal_velocity_error
                * (0.25 + 0.75 * near_ground)
                + self.cfg.rew_vz * vertical_speed_error
            ),
            "guidance": dt * self.cfg.rew_guidance * guidance_error,
            "control": (
                dt * self.cfg.rew_fuel * self._actions[:, 0] * self._actions[:, 0]
                + dt
                * self.cfg.rew_gimbal
                * torch.sum(self._actions[:, 1:] * self._actions[:, 1:], dim=-1)
                + self.cfg.rew_smooth * action_delta
            ),
            "terrain": (
                dt
                * self.cfg.rew_terrain
                * foot_clearance_spread
                * torch.exp(-altitude / 10.0)
            ),
            "time": torch.full_like(altitude, dt * self.cfg.rew_time),
        }

        termination = self._last_termination_metrics
        terminal = (
            termination["soft"].float() * self.cfg.rew_soft_landing
            + termination["harsh"].float()
            * (
                self.cfg.rew_harsh_landing
                + self.cfg.rew_landing_quality * termination["landing_quality"]
            )
            + termination["failed"].float() * self.cfg.rew_crash
            + termination["timeout_only"].float() * self.cfg.rew_timeout
        )
        terms["terminal"] = terminal
        reward = sum(terms.values())
        terms["total"] = reward
        for name, value in terms.items():
            self._episode_reward_sums[name] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self._rocket.data.root_pos_w
        quat = self._rocket.data.root_quat_w
        lin_vel = self._rocket.data.root_lin_vel_w
        ang_vel = self._rocket.data.root_ang_vel_w
        foot_clearances = self._foot_clearances(pos, quat)
        altitude = torch.amin(foot_clearances, dim=-1)
        max_foot_clearance = torch.amax(foot_clearances, dim=-1)
        foot_clearance_spread = max_foot_clearance - altitude
        distance_xy = torch.linalg.norm(self._target_pos_w[:, :2] - pos[:, :2], dim=-1)
        horizontal_speed = torch.linalg.norm(lin_vel[:, :2], dim=-1)
        vertical_speed = torch.abs(lin_vel[:, 2])
        angular_speed = torch.linalg.norm(ang_vel, dim=-1)
        near_ground = torch.exp(-altitude / 5.0)
        tilt_penalty, tilt_angle = self._tilt_against_terrain(quat, near_ground)
        out_of_bounds = torch.linalg.norm(pos[:, :2] - self.scene.env_origins[:, :2], dim=-1) > self.cfg.max_xy_m
        landed = altitude <= self.cfg.landing_altitude_m
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        escaped = (altitude > self.cfg.max_altitude_m) | (distance_xy > self.cfg.max_xy_m * 1.5)
        tilt_limit = 1.0 - math.cos(math.radians(self.cfg.soft_tilt_deg))
        soft = (
            landed
            & (distance_xy <= self.cfg.target_radius_m)
            & (horizontal_speed <= self.cfg.soft_horizontal_speed_mps)
            & (vertical_speed <= self.cfg.soft_vertical_speed_mps)
            & (angular_speed <= self.cfg.soft_angular_speed_rps)
            # tilt_penalty is judged against the local terrain normal near the
            # ground (see _tilt_against_terrain), so this gates the residual
            # attitude error after banking to match the slope -- not raw
            # world-vertical tilt, which real terrain often can't satisfy.
            & (tilt_penalty <= tilt_limit)
            # On a natural slope the first foot touches before the highest
            # foot. Judge whether the footprint fits the terrain by the spread,
            # not by absolute clearance (which also includes touchdown margin).
            # A rocket correctly banked to the local slope keeps this small
            # regardless of the slope's magnitude; see
            # tests/test_terrain_landability.py for the curvature-residual proof.
            & (foot_clearance_spread <= self.cfg.landing_max_foot_clearance_m)
        )
        landing_quality = (
            0.45 * torch.exp(-distance_xy / 2.0)
            + 0.35 * torch.exp(-vertical_speed / 0.75)
            + 0.10 * torch.exp(-horizontal_speed / 0.75)
            + 0.10 * torch.exp(-tilt_angle / math.radians(8.0))
        )
        failed = (out_of_bounds | escaped) & ~landed
        harsh = landed & ~soft
        terminated = landed | failed
        self._last_termination_metrics = {
            "soft": soft.detach(),
            "harsh": harsh.detach(),
            "failed": failed.detach(),
            "landed": landed.detach(),
            "out_of_bounds": out_of_bounds.detach(),
            "timeout_only": (timeout & ~terminated).detach(),
            "distance_xy": distance_xy.detach(),
            "horizontal_speed": horizontal_speed.detach(),
            "vertical_speed": vertical_speed.detach(),
            "angular_speed": angular_speed.detach(),
            "tilt_penalty": tilt_penalty.detach(),
            "landing_quality": landing_quality.detach(),
            "foot_clearance_spread": foot_clearance_spread.detach(),
        }
        return terminated, timeout

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._rocket._ALL_INDICES
        completed = self.episode_length_buf[env_ids] > 0
        completed_env_ids = env_ids[completed]
        if len(completed_env_ids) > 0 and hasattr(self, "_episode_reward_sums"):
            log = self.extras.setdefault("log", {})
            for name, values in self._episode_reward_sums.items():
                log[f"Episode_Reward/{name}"] = values[completed_env_ids].mean().item()
            metrics = self._last_termination_metrics
            log["Metrics/success_rate"] = metrics["soft"][completed_env_ids].float().mean().item()
            log["Metrics/harsh_landing_rate"] = metrics["harsh"][completed_env_ids].float().mean().item()
            log["Metrics/failure_rate"] = metrics["failed"][completed_env_ids].float().mean().item()
            log["Metrics/timeout_rate"] = metrics["timeout_only"][completed_env_ids].float().mean().item()
            log["Metrics/mean_landing_speed"] = metrics["vertical_speed"][completed_env_ids].mean().item()
            log["Metrics/mean_horizontal_speed"] = metrics["horizontal_speed"][completed_env_ids].mean().item()
            log["Metrics/mean_distance_xy"] = metrics["distance_xy"][completed_env_ids].mean().item()
            log["Metrics/mean_landing_quality"] = (
                metrics["landing_quality"][completed_env_ids].mean().item()
            )
            log["Metrics/mean_episode_length_s"] = (
                self.episode_length_buf[completed_env_ids].float().mean().item() * self.step_dt
            )
            if self.cfg.curriculum_enabled:
                self._curriculum_window_count += len(completed_env_ids)
                self._curriculum_window_successes += int(
                    metrics["soft"][completed_env_ids].sum().item()
                )
                if self._curriculum_window_count >= self._curriculum_window_target:
                    window_success_rate = (
                        self._curriculum_window_successes
                        / max(self._curriculum_window_count, 1)
                    )
                    if window_success_rate >= self.cfg.curriculum_success_threshold:
                        self._curriculum_level = min(
                            1.0,
                            self._curriculum_level + self.cfg.curriculum_increment,
                        )
                    # A full step back, not just a token half-step, whenever
                    # performance is meaningfully below the bar that would
                    # advance it -- not only in the near-total-failure case.
                    # The old (0.05 flat, half-increment) version let a run
                    # coast at ~5-25% success for a long time before backing
                    # off, and even then only barely.
                    elif window_success_rate < 0.5 * self.cfg.curriculum_success_threshold:
                        self._curriculum_level = max(
                            self.cfg.curriculum_initial_difficulty,
                            self._curriculum_level - self.cfg.curriculum_increment,
                        )
                    self._curriculum_window_count = 0
                    self._curriculum_window_successes = 0
                log["Curriculum/difficulty"] = self._curriculum_level
            for values in self._episode_reward_sums.values():
                values[env_ids] = 0.0
        super()._reset_idx(env_ids)
        self._rocket.reset(env_ids)

        count = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        root_state = self._rocket.data.default_root_state[env_ids].clone()
        if self.cfg.curriculum_enabled:
            difficulty = torch.rand(count, device=self.device) * self._curriculum_level
        else:
            difficulty = torch.ones(count, device=self.device)
        if hasattr(self, "_difficulty"):
            self._difficulty[env_ids] = difficulty
        # _rocket_mass_kg/_max_thrust_n (like _difficulty above) don't exist
        # yet during the very first reset, which DirectRLEnv triggers from
        # inside super().__init__() before this constructor has created them.
        if hasattr(self, "_rocket_mass_kg"):
            if self.cfg.physics_randomization_enabled:
                mass_lo, mass_hi = self.cfg.mass_randomization_range
                thrust_lo, thrust_hi = self.cfg.thrust_randomization_range
                self._rocket_mass_kg[env_ids] = self.cfg.rocket_mass_kg * torch.empty(
                    count, device=self.device
                ).uniform_(mass_lo, mass_hi)
                self._max_thrust_n[env_ids] = self.cfg.max_thrust_n * torch.empty(
                    count, device=self.device
                ).uniform_(thrust_lo, thrust_hi)
            else:
                self._rocket_mass_kg[env_ids] = self.cfg.rocket_mass_kg
                self._max_thrust_n[env_ids] = self.cfg.max_thrust_n
        target_xy_range = self.cfg.target_xy_min_range_m + difficulty * (
            self.cfg.target_xy_range_m - self.cfg.target_xy_min_range_m
        )
        target_offset = (
            torch.empty(count, 2, device=self.device).uniform_(-1.0, 1.0)
            * target_xy_range.unsqueeze(-1)
        )
        target_xy = origins[:, :2] + target_offset
        spawn_xy_range = self.cfg.spawn_xy_min_range_m + difficulty * (
            self.cfg.spawn_xy_range_m - self.cfg.spawn_xy_min_range_m
        )
        spawn_offset = (
            torch.empty(count, 2, device=self.device).uniform_(-1.0, 1.0)
            * spawn_xy_range.unsqueeze(-1)
        )
        spawn_altitudes = self.cfg.spawn_altitude_min_m + difficulty * (
            self.cfg.spawn_altitude_m - self.cfg.spawn_altitude_min_m
        )
        self._randomize_terrain(env_ids, origins[:, :2])
        root_state[:, :2] = target_xy + spawn_offset
        root_state[:, 2] = self._upright_root_height(root_state[:, :2]) + spawn_altitudes
        root_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)
        root_state[:, 7:] = 0.0
        self._rocket.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._rocket.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        joint_pos = self._rocket.data.default_joint_pos[env_ids].clone()
        joint_vel = self._rocket.data.default_joint_vel[env_ids].clone()
        joint_pos.zero_()
        joint_vel.zero_()
        self._rocket.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._joint_position_targets[env_ids] = 0.0

        self._target_pos_w[env_ids, :2] = target_xy
        self._target_pos_w[env_ids, 2] = self._upright_root_height(target_xy)
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._forces[env_ids] = 0.0
        self._torques[env_ids] = 0.0
        self._previous_distance_xy[env_ids] = torch.linalg.norm(
            self._target_pos_w[env_ids, :2] - root_state[:, :2],
            dim=-1,
        )
        self._previous_altitude[env_ids] = spawn_altitudes

    def _randomize_terrain(self, env_ids: torch.Tensor, origins_xy: torch.Tensor) -> None:
        """Assign cached terrain slots without generating terrain on the reset path."""
        count = len(env_ids)
        self._maybe_swap_terrain_pool()
        permutation = torch.randperm(self._terrain_pool_size, device=self.device)
        repeats = math.ceil(count / self._terrain_pool_size)
        slots = permutation.repeat(repeats)[:count]
        self._terrain_pool_slot_ids[env_ids] = slots

        self._terrain_slope[env_ids] = self._terrain_pool["slope"][slots]
        self._terrain_phase[env_ids] = self._terrain_pool["phase"][slots]
        self._terrain_amp[env_ids] = self._terrain_pool["amplitude"][slots]
        self._terrain_detail_phase[env_ids] = self._terrain_pool["detail_phase"][slots]
        self._terrain_detail_amp[env_ids] = self._terrain_pool["detail_amplitude"][slots]
        self._terrain_base_heights[env_ids] = self._terrain_pool["base_height"][slots]
        self._crater_xy[env_ids] = origins_xy[:, None, :] + self._terrain_pool["crater_xy"][slots]
        self._crater_radius[env_ids] = self._terrain_pool["crater_radius"][slots]
        self._crater_depth[env_ids] = self._terrain_pool["crater_depth"][slots]
        self._rock_xy[env_ids] = origins_xy[:, None, :] + self._terrain_pool["rock_xy"][slots]
        self._rock_radius[env_ids] = self._terrain_pool["rock_radius"][slots]
        self._rock_height[env_ids] = self._terrain_pool["rock_height"][slots]
        self._terrain_center_height[env_ids] = self._procedural_height_raw(origins_xy, env_ids)
        self._terrain_pool_assignments += count

        log = self.extras.setdefault("log", {})
        log["Terrain/pool_generation"] = self._terrain_pool_generation
        log["Terrain/pool_size"] = self._terrain_pool_size

    def _generate_terrain_pool(self, seed: int) -> dict[str, np.ndarray]:
        batch = generate_terrain_pool_batch(
            seed=seed,
            pool_size=self._terrain_pool_size,
            crater_count=self.cfg.terrain_crater_count,
            rock_count=self.cfg.terrain_rock_count,
            slope_scale=self.cfg.terrain_slope_scale,
            height_scale=self.cfg.terrain_height_scale_m,
            safe_feature_radius=0.0,
            detail_amplitude_range=self.cfg.terrain_detail_amplitude_range_m,
        )
        if self._terrain_pool_dem_generator is None:
            batch["base_height"] = np.zeros((self._terrain_pool_size, 2, 2), dtype=np.float32)
            return batch

        from app.randomization import sample_reset

        rng = np.random.default_rng(seed)
        heightmaps = []
        for _ in range(self._terrain_pool_size):
            reset_sample = sample_reset(self._terrain_pool_dem_cfg, self._terrain_pool_rocket_cfg, rng)
            reset_sample = replace(
                reset_sample,
                rocket_position=(0.0, 0.0, 0.0),
                target_position=(0.0, 0.0, 0.0),
            )
            heightmaps.append(self._terrain_pool_dem_generator.generate(reset_sample).heights)
        batch["base_height"] = np.stack(heightmaps).astype(np.float32)
        return batch

    def _setup_terrain_pool_dem(self) -> None:
        if os.environ.get("ISAACLAB_TERRAIN_POOL_USE_DEM", "1").lower() in {"0", "false", "no"}:
            return
        try:
            from app.config import load_yaml
            from app.terrain_generator import MoonTerrainGenerator

            terrain_cfg = dict(load_yaml(_project_root() / self.cfg.legacy_terrain_config_path))
            terrain_cfg["terrain_quality"] = os.environ.get(
                "ISAACLAB_TERRAIN_POOL_DEM_QUALITY",
                self.cfg.terrain_pool_dem_quality,
            )
            local_detail_cfg = dict(terrain_cfg.get("local_detail_patch", {}))
            local_detail_cfg["enabled"] = False
            terrain_cfg["local_detail_patch"] = local_detail_cfg
            generator = MoonTerrainGenerator(terrain_cfg)
            generator.load_dem()
            size_x, size_y = (float(value) for value in terrain_cfg.get("terrain_size_m", (80.0, 80.0)))
            self._terrain_pool_base_min_xy = (-0.5 * size_x, -0.5 * size_y)
            self._terrain_pool_base_max_xy = (0.5 * size_x, 0.5 * size_y)
            self._terrain_pool_dem_generator = generator
            self._terrain_pool_dem_cfg = terrain_cfg
            self._terrain_pool_rocket_cfg = load_yaml(_project_root() / self.cfg.legacy_rocket_config_path)
        except Exception as exc:
            print(f"[LunarRocket] NASA DEM terrain pool unavailable; using procedural macro terrain: {exc}", flush=True)
            self._terrain_pool_dem_generator = None

    def _schedule_terrain_pool_refill(self) -> None:
        if self._terrain_pool_executor is None or self._terrain_pool_future is not None:
            return
        seed = self._terrain_pool_next_seed
        self._terrain_pool_next_seed += 1
        self._terrain_pool_future = self._terrain_pool_executor.submit(self._generate_terrain_pool, seed)

    def _maybe_swap_terrain_pool(self) -> None:
        """Atomically publish a staged pool only between episode assignments."""
        future = self._terrain_pool_future
        if self._terrain_pool_assignments < self._terrain_pool_refresh_assignments:
            return
        if future is None or not future.done():
            return
        try:
            staged_pool = future.result()
        except Exception as exc:
            print(f"[LunarRocket] terrain pool refill failed; retaining active pool: {exc}", flush=True)
            self._terrain_pool_future = None
            self._schedule_terrain_pool_refill()
            return

        for key, value in staged_pool.items():
            self._terrain_pool[key].copy_(torch.as_tensor(value, dtype=torch.float32, device=self.device))
        self._terrain_pool_generation += 1
        self._terrain_pool_assignments = 0
        self._terrain_pool_future = None
        self._schedule_terrain_pool_refill()

    def _terrain_height(self, xy_w: torch.Tensor) -> torch.Tensor:
        legacy_height = self._legacy_terrain_height(xy_w)
        if legacy_height is not None:
            return legacy_height
        env_ids = self._env_ids_from_xy(xy_w)
        origin_xy = self.scene.env_origins[env_ids, :2]
        raw_height = self._procedural_height_raw(xy_w, env_ids)
        radius = torch.linalg.norm(xy_w - origin_xy, dim=-1)
        # Never flatten or replace the target region. The policy lands on the
        # same cratered height field that its lidar and touchdown checks read;
        # see _terrain_normal for how attitude is judged against real slope.
        return raw_height + self._local_detail_height(xy_w, env_ids, origin_xy, radius)

    def _terrain_normal(self, xy_w: torch.Tensor) -> torch.Tensor:
        """Outward surface normal of the real (unflattened) terrain at xy_w."""
        eps = 0.35
        dx = torch.tensor((eps, 0.0), device=self.device).expand_as(xy_w)
        dy = torch.tensor((0.0, eps), device=self.device).expand_as(xy_w)
        h0 = self._terrain_height(xy_w)
        hx = self._terrain_height(xy_w + dx)
        hy = self._terrain_height(xy_w + dy)
        grad_x = (hx - h0) / eps
        grad_y = (hy - h0) / eps
        normal = torch.stack((-grad_x, -grad_y, torch.ones_like(grad_x)), dim=-1)
        return normal / torch.clamp(torch.linalg.norm(normal, dim=-1, keepdim=True), min=1e-6)

    def _tilt_against_terrain(
        self, quat: torch.Tensor, near_ground: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tilt of the rocket's up axis against a target-aware reference.

        The reference is world-vertical in flight, blending into the real
        (unflattened) terrain's surface normal at the target as the rocket
        nears the ground (near_ground = exp(-altitude/5)). A rocket landing on
        a natural slope must bank to match it -- judging tilt against world
        vertical regardless of the ground actually being touched made the
        soft-landing gate nearly unreachable on real terrain (measured 30-50%
        of random targets have >10 deg local slope). Judging it against this
        blended reference instead makes the residual attitude error, not the
        raw slope, the thing being gated.
        """
        target_normal = self._terrain_normal(self._target_pos_w[:, :2])
        world_up = torch.zeros_like(target_normal)
        world_up[:, 2] = 1.0
        desired_up = world_up + near_ground.unsqueeze(-1) * (target_normal - world_up)
        desired_up = desired_up / torch.clamp(torch.linalg.norm(desired_up, dim=-1, keepdim=True), min=1e-6)
        body_up_w = _quat_rotate(quat, world_up)
        tilt_cos = torch.clamp(torch.sum(body_up_w * desired_up, dim=-1), -1.0, 1.0)
        return 1.0 - tilt_cos, torch.acos(tilt_cos)

    def _local_detail_height(
        self,
        xy_w: torch.Tensor,
        env_ids: torch.Tensor,
        origin_xy: torch.Tensor,
        radius: torch.Tensor,
    ) -> torch.Tensor:
        """Centimeter-scale regolith detail, faded into the coarse global terrain."""
        local_xy = xy_w - origin_xy
        phase = self._terrain_detail_phase[env_ids]
        detail = self._terrain_detail_amp[env_ids] * (
            0.55
            * torch.sin(3.2 * local_xy[:, 0] + phase[:, 0])
            * torch.sin(2.7 * local_xy[:, 1] + phase[:, 1])
            + 0.30
            * torch.sin(6.4 * local_xy[:, 0] + phase[:, 2])
            * torch.sin(5.8 * local_xy[:, 1] + phase[:, 3])
            + 0.15 * torch.sin(9.5 * (local_xy[:, 0] + local_xy[:, 1]) + phase[:, 0])
        )
        detail_at_target = self._terrain_detail_amp[env_ids] * (
            0.55 * torch.sin(phase[:, 0]) * torch.sin(phase[:, 1])
            + 0.30 * torch.sin(phase[:, 2]) * torch.sin(phase[:, 3])
            + 0.15 * torch.sin(phase[:, 0])
        )
        fade = self._terrain_fade(radius)
        return (detail - detail_at_target) * fade

    def _terrain_fade_smoothstep(self, radius: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(1.0 - radius / max(self.cfg.terrain_detail_radius_m, 1e-6), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _terrain_fade_gaussian(self, radius: torch.Tensor) -> torch.Tensor:
        r_over_R = radius / max(self.cfg.terrain_detail_radius_m, 1e-6)
        return torch.exp(-self.cfg.terrain_gaussian_k * r_over_R * r_over_R)

    def _terrain_fade(self, radius: torch.Tensor) -> torch.Tensor:
        """Fine-detail fade weight, dispatched on ISAACLAB_TERRAIN_BLEND_METHOD /
        cfg.terrain_blend_method. All three shapes hard-truncate to exactly 0
        at radius >= terrain_detail_radius_m, so the comparison between them
        is a same-transition-budget comparison (see
        experiments/terrain_transition/README.md "Finding" #1 for why the
        naive Gaussian shape needs this truncation and smoothstep doesn't)."""
        R = max(self.cfg.terrain_detail_radius_m, 1e-6)
        if self._terrain_blend_method == "smoothstep":
            return self._terrain_fade_smoothstep(radius)
        if self._terrain_blend_method == "gaussian":
            fade = self._terrain_fade_gaussian(radius)
            return torch.where(radius < R, fade, torch.zeros_like(fade))
        # hybrid: Gaussian near the landing-gear contact zone, smoothstep
        # beyond it, joined by a smoothstep-weighted switch so no new seam
        # is introduced at the contact radius.
        rc = self._terrain_hybrid_contact_radius_m
        width = max(self.cfg.terrain_hybrid_switch_width_m, 1e-6)
        lo, hi = rc - width / 2.0, rc + width / 2.0
        t = torch.clamp((hi - radius) / max(hi - lo, 1e-9), 0.0, 1.0)
        switch = t * t * (3.0 - 2.0 * t)
        gaussian = self._terrain_fade_gaussian(radius)
        smoothstep = self._terrain_fade_smoothstep(radius)
        fade = switch * gaussian + (1.0 - switch) * smoothstep
        return torch.where(radius < R, fade, torch.zeros_like(fade))

    def _procedural_height_raw(self, xy_w: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        origin_xy = self.scene.env_origins[env_ids, :2]
        local_xy = xy_w - origin_xy
        base_height = self._sample_height_tensor(
            self._terrain_base_heights,
            local_xy,
            self._terrain_pool_base_min_xy,
            self._terrain_pool_base_max_xy,
            env_ids,
        )
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
        return (
            base_height
            + slope
            + ripple
            + torch.sum(crater_bowl + crater_rim, dim=-1)
            + torch.sum(rock_profile, dim=-1)
        )

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
        # Direction, not just magnitude: the policy must know which way to bank
        # to align with the target's slope (see _tilt_against_terrain), and
        # slope_norm alone can't express that.
        grad_x_norm = torch.clamp(grad_x / 0.5, -1.0, 1.0)
        grad_y_norm = torch.clamp(grad_y / 0.5, -1.0, 1.0)
        return torch.stack(
            (slope_norm, roughness_norm, safe_zone, target_height, grad_x_norm, grad_y_norm), dim=-1
        )

    def _foot_positions_w(self, pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
        """World positions of the four foot centers authored in hopper_lunar.xml."""
        count = pos_w.shape[0]
        offsets = self._foot_offsets_b.unsqueeze(0).expand(count, -1, -1)
        quats = quat_w.unsqueeze(1).expand(-1, offsets.shape[1], -1)
        rotated = _quat_rotate(quats.reshape(-1, 4), offsets.reshape(-1, 3))
        return pos_w.unsqueeze(1) + rotated.reshape(count, offsets.shape[1], 3)

    def _foot_clearances(self, pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
        """Signed clearance from every foot sphere to the actual terrain mesh height."""
        foot_positions = self._foot_positions_w(pos_w, quat_w)
        terrain_height = self._terrain_height(foot_positions[:, :, :2].reshape(-1, 2))
        terrain_height = terrain_height.reshape(pos_w.shape[0], -1)
        return foot_positions[:, :, 2] - float(self.cfg.rocket_foot_radius_m) - terrain_height

    def _upright_root_height(self, xy_w: torch.Tensor) -> torch.Tensor:
        """Root height that places all four XML feet on/above the local surface."""
        foot_xy = xy_w[:, None, :] + self._foot_offsets_b[None, :, :2]
        terrain_height = self._terrain_height(foot_xy.reshape(-1, 2)).reshape(xy_w.shape[0], -1)
        required_root_height = (
            terrain_height
            - self._foot_offsets_b[None, :, 2]
            + float(self.cfg.rocket_foot_radius_m)
        )
        return torch.amax(required_root_height, dim=-1)

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
        mass = torch.clamp(self._rocket_mass_kg, min=1e-6).unsqueeze(-1)
        accel_body = _quat_rotate_inverse(quat, self._forces[:, 0, :] / mass) / 30.0
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
        num_envs = self.scene.env_origins.shape[0]
        query_count = xy_w.shape[0]
        if query_count == num_envs:
            return torch.arange(num_envs, device=self.device)
        # All batched terrain sensors build [num_envs, samples, 2] and then
        # flatten it. Preserve that ordering directly instead of allocating an
        # O(query_count * num_envs) torch.cdist matrix every simulation step.
        if query_count % num_envs == 0:
            samples_per_env = query_count // num_envs
            return torch.arange(num_envs, device=self.device).repeat_interleave(samples_per_env)
        distance = torch.cdist(xy_w, self.scene.env_origins[:, :2])
        return torch.argmin(distance, dim=-1)

    def _setup_legacy_terrain(self) -> bool:
        if os.environ.get("ISAACLAB_USE_DEM_TERRAIN", "0").lower() not in {"1", "true", "yes"}:
            print(
                "[LunarRocket] using reset-randomized GPU terrain envelope "
                "(set ISAACLAB_USE_DEM_TERRAIN=1 for the static DEM/USD demo terrain)",
                flush=True,
            )
            return False
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
