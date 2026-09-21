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
from isaaclab.assets import RigidObject
from isaaclab.envs.direct_rl_env import DirectRLEnv

# Isaac Lab task convention: the *_env_cfg module holds only Cfg dataclasses
# (isaaclab.assets.RigidObjectCfg, isaaclab.envs.DirectRLEnvCfg, etc.), never
# the runtime env class. gym.register's env_cfg_entry_point resolves this
# module alone (see __init__.py), which must succeed before SimulationApp
# exists. This module (lunar_lander_env), by contrast, imports DirectRLEnv
# and RigidObject -- runtime classes that pull in isaaclab.sim.SimulationContext
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
            # Opt-in override for the dedicated RTX video camera's fixed pose
            # (default: a 3/4 side angle) -- e.g. ISAACLAB_VIDEO_CAMERA_EYE=
            # "0,0,6" with ISAACLAB_VIDEO_CAMERA_TARGET="0,0,0.5" for a
            # straight-overhead shot. Off by default, so normal training/play
            # video capture is unaffected.
            eye_raw = os.environ.get("ISAACLAB_VIDEO_CAMERA_EYE", "6.0,-7.0,5.0")
            target_raw = os.environ.get("ISAACLAB_VIDEO_CAMERA_TARGET", "0.0,0.0,1.5")
            self._video_camera.set_world_poses_from_view(
                np.asarray([[float(v) for v in eye_raw.split(",")]], dtype=np.float32),
                np.asarray([[float(v) for v in target_raw.split(",")]], dtype=np.float32),
            )
        self._actions = torch.zeros(self.num_envs, 1 + len(cfg.rcs_thruster_layout), device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._difficulty = torch.zeros(self.num_envs, device=self.device)
        self._rocket_mass_kg = torch.full((self.num_envs,), float(cfg.rocket_mass_kg), device=self.device)
        self._max_thrust_n = torch.full((self.num_envs,), float(cfg.max_thrust_n), device=self.device)
        self._previous_distance_xy = torch.zeros(self.num_envs, device=self.device)
        self._previous_altitude = torch.zeros(self.num_envs, device=self.device)
        # Per-env spawn altitude, set at each reset -- used by _get_rewards
        # to size the loiter grace period to THIS episode's own physically
        # -implied minimum descent time (see rew_wb_loiter_altitude_grace_frac).
        self._spawn_altitude_m = torch.zeros(self.num_envs, device=self.device)
        self._body_ids = self._rocket.find_bodies("hopper")[0]
        if len(self._body_ids) != 1:
            raise RuntimeError(f"Expected one XML hopper root body, found body ids: {self._body_ids}")
        # RCS thruster ring + fixed main engine (replaces the old TVC gimbal
        # joints): fixed body-frame positions/directions, consumed directly
        # in _pre_physics_step to compute forces/torques as wrenches -- no
        # joints or actuators involved (see rcs_thruster_layout's cfg
        # comment).
        rcs_pos_b, rcs_dir_b = zip(*cfg.rcs_thruster_layout)
        self._rcs_pos_b = torch.tensor(rcs_pos_b, dtype=torch.float32, device=self.device)
        rcs_dir_b = torch.tensor(rcs_dir_b, dtype=torch.float32, device=self.device)
        self._rcs_dir_b = rcs_dir_b / torch.linalg.norm(rcs_dir_b, dim=-1, keepdim=True)
        self._engine_thrust_dir_b = torch.tensor(
            cfg.engine_thrust_dir_b, dtype=torch.float32, device=self.device
        )
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
        # Debug/watch-only override: the trained-and-tuned default (0.006-0.020 m,
        # see CLAUDE.md/experiments/terrain_transition) is a physical micro-roughness
        # term feeding the foot-clearance landing gate, not meant to be visible from
        # a normal camera distance -- deliberately NOT the cfg default so a real
        # training run never picks this up by accident.
        detail_amp_min = os.environ.get("ISAACLAB_TERRAIN_DETAIL_AMPLITUDE_MIN_M")
        detail_amp_max = os.environ.get("ISAACLAB_TERRAIN_DETAIL_AMPLITUDE_MAX_M")
        if detail_amp_min or detail_amp_max:
            lo, hi = self.cfg.terrain_detail_amplitude_range_m
            lo = float(detail_amp_min) if detail_amp_min else lo
            hi = float(detail_amp_max) if detail_amp_max else hi
            self.cfg.terrain_detail_amplitude_range_m = (lo, hi)
            print(
                f"[LunarRocket] terrain_detail_amplitude_range_m overridden to ({lo}, {hi}) -- "
                "watch/debug only, do not use for real training runs",
                flush=True,
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
            # One key per whiteboard reward term (see _get_rewards).
            for name in (
                "proximity",
                "tilt",
                "foot_load",
                "throttle",
                "altitude",
                "rcs",
                "rel_xy",
                "velocity",
                "readiness",
                "loiter",
                "terminal",
                "total",
            )
        }
        # Opt-in, per-step CSV telemetry (position/thrust/RCS duty/attitude),
        # off by default (zero overhead) unless ISAACLAB_TELEMETRY_CSV_DIR is
        # set -- added to diagnose the timeout/gate bottlenecks in
        # TRAINING_STATUS.md by eye instead of only from aggregated
        # Metrics/* scalars. Limited to a handful of env ids
        # (ISAACLAB_TELEMETRY_ENV_IDS, default just env 0) since writing
        # every one of e.g. 128 envs every step would dominate wall-clock and
        # disk. Chunked into multiple files (ISAACLAB_TELEMETRY_CHUNK_STEPS
        # rows/file) so a long run doesn't produce one unbounded CSV.
        self._telemetry_csv_dir = os.environ.get("ISAACLAB_TELEMETRY_CSV_DIR", "").strip()
        self._telemetry_env_ids: list[int] = []
        self._telemetry_writer = None
        self._telemetry_file = None
        self._telemetry_rows_in_chunk = 0
        self._telemetry_chunk_index = 0
        self._telemetry_step_counter = 0
        self._telemetry_chunk_steps = max(1, int(os.environ.get("ISAACLAB_TELEMETRY_CHUNK_STEPS", "5000")))
        if self._telemetry_csv_dir:
            ids_raw = os.environ.get("ISAACLAB_TELEMETRY_ENV_IDS", "0")
            self._telemetry_env_ids = sorted(
                {int(x) for x in ids_raw.split(",") if x.strip() != "" and int(x) < self.num_envs}
            )
            os.makedirs(self._telemetry_csv_dir, exist_ok=True)
            self._telemetry_open_new_chunk()
            print(
                f"[LunarRocket] step telemetry ON: envs={self._telemetry_env_ids} -> "
                f"{self._telemetry_csv_dir} (chunk={self._telemetry_chunk_steps} rows/file)",
                flush=True,
            )

    def _telemetry_open_new_chunk(self) -> None:
        import csv

        if self._telemetry_file is not None:
            self._telemetry_file.close()
        path = os.path.join(
            self._telemetry_csv_dir,
            f"telemetry_chunk_{self._telemetry_chunk_index:05d}.csv",
        )
        self._telemetry_file = open(path, "w", newline="")
        self._telemetry_writer = csv.writer(self._telemetry_file)
        self._telemetry_writer.writerow(
            [
                "sim_step",
                "env_id",
                "difficulty",
                "pos_x",
                "pos_y",
                "pos_z",
                "quat_w",
                "quat_x",
                "quat_y",
                "quat_z",
                "lin_vel_x",
                "lin_vel_y",
                "lin_vel_z",
                "ang_vel_x",
                "ang_vel_y",
                "ang_vel_z",
                "throttle_cmd",
                *[f"rcs_{i}_duty" for i in range(len(self.cfg.rcs_thruster_layout))],
                "target_x",
                "target_y",
                "distance_xy_m",
                "altitude_m",
            ]
        )
        self._telemetry_chunk_index += 1
        self._telemetry_rows_in_chunk = 0

    def _telemetry_log_step(
        self,
        pos: torch.Tensor,
        quat: torch.Tensor,
        lin_vel: torch.Tensor,
        ang_vel: torch.Tensor,
        distance_xy: torch.Tensor,
        altitude: torch.Tensor,
    ) -> None:
        if not self._telemetry_env_ids:
            return
        rcs_duty = torch.clamp(self._actions[:, 1:], 0.0, 1.0)
        self._telemetry_step_counter += 1
        for env_id in self._telemetry_env_ids:
            self._telemetry_writer.writerow(
                [
                    self._telemetry_step_counter,
                    env_id,
                    float(self._difficulty[env_id].item()),
                    *[float(v) for v in pos[env_id].tolist()],
                    *[float(v) for v in quat[env_id].tolist()],
                    *[float(v) for v in lin_vel[env_id].tolist()],
                    *[float(v) for v in ang_vel[env_id].tolist()],
                    float(self._actions[env_id, 0].item()),
                    *[float(v) for v in rcs_duty[env_id].tolist()],
                    float(self._target_pos_w[env_id, 0].item()),
                    float(self._target_pos_w[env_id, 1].item()),
                    float(distance_xy[env_id].item()),
                    float(altitude[env_id].item()),
                ]
            )
        self._telemetry_rows_in_chunk += len(self._telemetry_env_ids)
        self._telemetry_file.flush()
        if self._telemetry_rows_in_chunk >= self._telemetry_chunk_steps:
            self._telemetry_open_new_chunk()

    def close(self):
        executor = getattr(self, "_terrain_pool_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            self._terrain_pool_executor = None
        telemetry_file = getattr(self, "_telemetry_file", None)
        if telemetry_file is not None:
            telemetry_file.close()
            self._telemetry_file = None
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
        # RigidObject, not Articulation: the RCS/fixed-engine design (see
        # rcs_thruster_layout) has zero joints left -- Isaac Lab's
        # Articulation wrapper requires an ArticulationRootAPI prim, which
        # the MJCF->USD converter only authors when the body has at least
        # one internal joint. A joint-free floating body is a RigidObject.
        self._rocket = RigidObject(rocket_cfg)
        self._contact_sensor = None
        if self.cfg.contact_sensor_enabled:
            from isaaclab.sensors import ContactSensor, ContactSensorCfg

            # Body-level only (Isaac Lab limitation) and the four feet are geoms
            # on "hopper", so this is one aggregate vehicle contact force.
            self._contact_sensor = ContactSensor(
                ContactSensorCfg(prim_path="/World/envs/env_.*/Rocket/.*hopper.*")
            )
            self.scene.sensors["rocket_contact"] = self._contact_sensor
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
        # RCS thruster duties: [0, 1] only (not [-1, 1]) -- a<=0 means "off,"
        # matching a real RCS jet resting idle at zero command and reusing
        # SAC's zero-centered action init as a natural "no attitude
        # authority commanded yet" start state (was the TVC gimbal command,
        # centered at 0 = neutral gimbal angle; RCS has no such neutral
        # angle, only on/off duty per fixed-direction jet).
        rcs_duty = torch.clamp(self._actions[:, 1:], 0.0, 1.0)
        self._actions[:, 1:] = rcs_duty

        quat = _xyzw_to_wxyz(self._rocket.data.root_quat_w)

        # Fixed main engine: straight down the hopper's own -Z axis, no
        # gimbal -- the force line passes through the body's Z-axis so it
        # contributes zero torque about the CoM (replaces the old
        # TVC-deflected thrust + lever-arm torque calc).
        engine_dir_w = _quat_rotate(quat, self._engine_thrust_dir_b.unsqueeze(0).expand(self.num_envs, -1))
        force_w = engine_dir_w * (self._actions[:, 0:1] * self._max_thrust_n.unsqueeze(-1))
        torque_w = torch.zeros_like(force_w)

        # RCS authority curriculum: same floor-to-ceiling-by-difficulty shape
        # the TVC gimbal-angle ramp used, now scaling max per-thruster force
        # instead (see curriculum_full_rcs_difficulty's cfg comment for why
        # the shape -- not the TVC geometry -- is what's being reused).
        # Stage 2 of the staged curriculum (see cfg.stage1_nav_only):
        # freezes this at policy_max_rcs_thrust_n once a checkpoint has
        # already learned RCS control at full authority, so resuming
        # doesn't reset an already-calibrated action mapping back to the
        # floor.
        if self.cfg.curriculum_rcs_always_max:
            rcs_frac = torch.ones_like(self._difficulty)
        else:
            rcs_frac = torch.clamp(
                self._difficulty / max(self.cfg.curriculum_full_rcs_difficulty, 1e-6), 0.0, 1.0
            )
        rcs_authority_n = (
            self.cfg.policy_min_rcs_thrust_n
            + rcs_frac * (self.cfg.policy_max_rcs_thrust_n - self.cfg.policy_min_rcs_thrust_n)
        ).unsqueeze(-1)

        # 8 independent, fixed-direction jets (see rcs_thruster_layout):
        # each contributes both a force (RCS does add a little net
        # translation, not just torque -- not special-cased away) and a
        # torque about the CoM from its fixed lever arm.
        for i in range(self._rcs_pos_b.shape[0]):
            dir_w_i = _quat_rotate(quat, self._rcs_dir_b[i].unsqueeze(0).expand(self.num_envs, -1))
            pos_w_i = _quat_rotate(quat, self._rcs_pos_b[i].unsqueeze(0).expand(self.num_envs, -1))
            force_w_i = dir_w_i * (rcs_duty[:, i : i + 1] * rcs_authority_n)
            force_w = force_w + force_w_i
            torque_w = torque_w + torch.cross(pos_w_i, force_w_i, dim=-1)

        self._forces[:, 0, :] = force_w
        self._torques[:, 0, :] = torque_w

    def _apply_action(self) -> None:
        self._rocket.permanent_wrench_composer.set_forces_and_torques(
            forces=self._forces,
            torques=self._torques,
            body_ids=self._body_ids,
            is_global=True,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        pos = self._rocket.data.root_pos_w
        quat = _xyzw_to_wxyz(self._rocket.data.root_quat_w)
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
        quat = _xyzw_to_wxyz(self._rocket.data.root_quat_w)
        lin_vel = self._rocket.data.root_lin_vel_w
        ang_vel = self._rocket.data.root_ang_vel_w

        delta_xy = self._target_pos_w[:, :2] - pos[:, :2]
        distance_xy = torch.linalg.norm(delta_xy, dim=-1)
        # Kept up to date for _reset_idx's bookkeeping even though the
        # whiteboard reward is absolute-state based, not progress based.
        self._previous_distance_xy = distance_xy.detach()

        foot_clearances = self._foot_clearances(pos, quat)
        altitude = torch.clamp(torch.amin(foot_clearances, dim=-1), min=0.0)
        self._previous_altitude = altitude.detach()
        near_ground = torch.exp(-altitude / 5.0)
        horizontal_speed = torch.linalg.norm(lin_vel[:, :2], dim=-1)
        vertical_speed = torch.abs(lin_vel[:, 2])
        angular_speed = torch.linalg.norm(ang_vel, dim=-1)
        # theta on the board: residual attitude error against the local terrain
        # normal near the ground, world vertical while still in flight.
        _, tilt_angle = self._tilt_against_terrain(quat, near_ground)
        dt = float(self.step_dt)
        cfg = self.cfg

        # --- konum: bounded proximity reward, the board's -alpha*(1/konum)^beta.
        # 1/(1+d) is 1 at the target and decays to 0 far out, so there is no
        # singularity at d=0 (see the sign note on rew_wb_proximity_alpha).
        proximity = torch.pow(1.0 / (1.0 + distance_xy), cfg.rew_wb_proximity_beta)

        # --- Bounded, always-positive attitude reward (was the board's literal
        # -T0*e^(IMU theta)*altitude^zeta penalty). tilt_angle is the residual
        # attitude error against the local terrain normal, so matching a slope
        # costs nothing; e^(-theta) is 1 (max) when perfectly aligned and
        # decays toward 0 as the residual grows, never negative -- an
        # unbounded exponential penalty here gave SAC no ceiling to relax
        # toward once already stable.
        altitude_factor = torch.pow(
            1.0 + altitude / cfg.max_altitude_m, cfg.rew_wb_tilt_altitude_zeta
        )
        tilt_reward = torch.exp(-tilt_angle) * altitude_factor

        # --- -h * (F_ayak / F_ayak_max)^c.  Real per-foot normal force from the
        # contact sensor where collision geometry exists; see _foot_normal_forces.
        foot_forces = self._foot_normal_forces()
        peak_foot_force = torch.amax(foot_forces, dim=-1)
        foot_load_ratio = peak_foot_force / max(cfg.rew_wb_foot_force_max_n, 1e-6)
        foot_load_cost = torch.pow(foot_load_ratio, cfg.rew_wb_foot_load_c)

        # --- axis-separated position error, distinct from the radial term above.
        rel_x = torch.abs(delta_xy[:, 0])
        rel_y = torch.abs(delta_xy[:, 1])

        throttle_cmd = self._actions[:, 0]
        # RCS duty per thruster is already clamped to [0, 1] in
        # _pre_physics_step; sum across all 8 for a fuel/control-effort
        # measure (was gimbal_cmd = norm(yaw, pitch) under TVC).
        rcs_duty = self._actions[:, 1:]

        # --- Touchdown "readiness": a dense, near_ground-gated bonus that is
        # only large when ALL SIX soft-landing axes are simultaneously close
        # to their gate at once (a product of per-axis closeness scores, each
        # in (0, 1], -> 1 only if every factor is near 1). Added because
        # per-gate diagnostics across many full runs showed each individual
        # gate (distance/horizontal/vertical/angular/tilt/foot) reaching
        # 75-94% pass rate on its own, yet joint success_rate plateaued at
        # ~22-29% -- the axes were fighting each other during the final
        # braking maneuver (killing vertical speed fast tends to induce
        # horizontal drift or tilt correction, etc.), so a policy that is
        # "usually good on every axis separately" was still rarely good on
        # all of them AT THE SAME INSTANT of touchdown. Rewarding terms
        # individually can't fix that; only rewarding their conjunction can.
        termination = self._last_termination_metrics
        foot_spread = termination["foot_clearance_spread"]
        # Dedicated, much tighter altitude gate than the shared `near_ground`
        # (exp(-altitude/5.0), used for terrain-normal blending elsewhere) --
        # rew_wb_readiness_weight was cut 8.0 -> 2.5 previously because at
        # 8.0 a calm hover at 2-5m altitude (where near_ground/5.0 is still
        # ~0.6-1.0) could bank most of this term's max reward without ever
        # actually landing, nearly canceling the loiter penalty and keeping
        # timeout_rate at 70-89%. This run's TensorBoard data confirms the
        # opposite failure at weight=2.5 instead: the term is now so small
        # (0.4-0.8/episode vs loiter's -145 to -180) that it does nothing to
        # counteract the 6-gate joint-simultaneity problem it was added to
        # solve -- success_rate erodes in lockstep with EVERY gate together
        # whenever curriculum difficulty ticks up, exactly the pattern this
        # term can't currently prevent. Narrowing the altitude scale from
        # 5.0m to 1.2m means near_ground_tight is only non-negligible in the
        # last ~1-2m of descent -- a calm hover at 2m+ (the previous exploit)
        # now scores close to zero here, so the weight can go back up without
        # reopening it.
        near_ground_tight = torch.exp(-altitude / cfg.rew_wb_readiness_altitude_scale_m)
        readiness = (
            near_ground_tight
            * torch.exp(-torch.pow(distance_xy / cfg.target_radius_m, 2))
            * torch.exp(-torch.pow(horizontal_speed / cfg.soft_horizontal_speed_mps, 2))
            * torch.exp(-torch.pow(vertical_speed / cfg.soft_vertical_speed_mps, 2))
            * torch.exp(-torch.pow(angular_speed / cfg.soft_angular_speed_rps, 2))
            * torch.exp(-torch.pow(tilt_angle / math.radians(cfg.soft_tilt_deg), 2))
            * torch.exp(-torch.pow(foot_spread / cfg.landing_max_foot_clearance_m, 2))
        )

        terms = {
            # + alpha * (1/(1+konum))^beta
            "proximity": dt * cfg.rew_wb_proximity_alpha * proximity,
            # + T0 * e^(-IMU theta) * altitude^zeta
            "tilt": dt * cfg.rew_wb_tilt_t0 * tilt_reward,
            # - h * (F_ayak/F_ayak_max)^c
            "foot_load": -dt * cfg.rew_wb_foot_load_h * foot_load_cost,
            # - s0 * throttle^2
            "throttle": -dt * cfg.rew_wb_throttle_s0 * throttle_cmd * throttle_cmd,
            # - k_z * |z_hedef - z|  (altitude above the target's own surface)
            "altitude": -dt * cfg.rew_wb_altitude_k * altitude,
            # - |rcs|^d: control-effort penalty summed over all 8 thrusters.
            "rcs": -dt * cfg.rew_wb_rcs_weight * torch.sum(torch.pow(rcs_duty, cfg.rew_wb_rcs_d), dim=-1),
            # Bounded, always-positive per-axis closeness reward (mirrors the
            # proximity term's 1/(1+d) shape): weight_* at rel=0, ->0 far
            # away. Replaces the board's literal -x1*(rel_x)^x2 penalty,
            # which gave SAC no ceiling to relax toward once already close
            # and converged to "hover nearby forever" across three full runs.
            "rel_xy": dt
            * (
                cfg.rew_wb_rel_x_weight * torch.pow(1.0 / (1.0 + rel_x), cfg.rew_wb_rel_x_power)
                + cfg.rew_wb_rel_y_weight * torch.pow(1.0 / (1.0 + rel_y), cfg.rew_wb_rel_y_power)
            ),
            # Bounded, always-positive "being slow" reward (was the board's
            # literal -x5*(V_xy)^x6 penalty plus unbounded vertical/angular
            # braking terms): each speed's 1/(1+v) is 1 (max) at a dead stop
            # and ->0 as speed grows, still driving the board's "Dikey hiz <
            # 1" / "Acisal hiz < 0.5" constraints without an unbounded penalty
            # SAC has no ceiling to relax toward once already slow.
            "velocity": dt
            * (
                cfg.rew_wb_vxy_weight * torch.pow(1.0 / (1.0 + horizontal_speed), cfg.rew_wb_vxy_power)
                + cfg.rew_wb_vz_weight * torch.pow(1.0 / (1.0 + vertical_speed), cfg.rew_wb_vz_power)
                + cfg.rew_wb_angular_weight / (1.0 + angular_speed)
            ),
            # Zeroed in stage1_nav_only mode: readiness rewards the 6
            # SOFT-LANDING axes together, which is meaningless before the
            # policy is even attempting to land -- see cfg.stage1_nav_only's
            # comment for the staged-curriculum rationale.
            "readiness": (
                0.0 if cfg.stage1_nav_only else dt * cfg.rew_wb_readiness_weight * readiness
            ),
        }

        # Flat per-second cost for not having landed yet -- dense so SAC gets
        # gradient every step instead of only at the terminal timeout penalty
        # 20s later. Zeroed on the step touchdown happens so it never eats
        # into the terminal soft/harsh payout.
        #
        # TESTED AND REJECTED: a rew_wb_loiter_grace_s=9.0s grace period
        # (zero loiter cost before t=9s, theorized to let the policy spend
        # the ~9s of lateral-accel budget needed to null a 1 m/s horizontal
        # error at the ~4 deg gimbal floor before loiter pressures a rushed
        # attempt). Result on a fresh run: whole-run success_rate 0.0%
        # across all 484 log windows, gate_distance/horizontal/tilt/foot all
        # collapsed to 3-7% (from 70-85% at the flat-loiter baseline), and
        # mean_episode_length_s rose to 15.9-18.5s (near the 20s cap) --
        # confirming the policy DID get more patient, but that patience
        # never converted into better landings, it just delayed the same
        # poor outcome. This is the third consecutive "give the policy more
        # time/authority" intervention (after two gimbal-floor increases,
        # see the archived TVC gimbal-authority history in
        # LunarLanderEnvCfg) to collapse nearly every
        # gate at once rather than improving the targeted one -- a pattern
        # suggesting these changes disrupt some other dependency in the
        # existing reward/training dynamics rather than validating or
        # refuting the "not enough authority/time" theory itself. Reverted
        # to the flat per-step cost.
        # P2 fix (2026-09-18): the flat cost above was tested with a FIXED
        # 9.0s grace period and collapsed training outright (see the
        # TESTED-AND-REJECTED note) -- but per-difficulty TensorBoard
        # analysis on the 07-46-02 run shows a different, altitude-SCALED
        # mechanism: a safe descent at the soft_vertical_speed_mps gate
        # takes altitude/soft_vertical_speed_mps seconds, and that minimum
        # grows linearly with spawn altitude while loiter's cost over that
        # same span grows just as fast -- at diff=0.2 (alt=5.4m) the
        # required ~9s of patient descent already costs ~-90 loiter, which
        # is where success_rate collapsed (95%->59% terminal-reward flip).
        # The flat 9.0s grace failed because it was the SAME for every
        # spawn altitude, including the ~0.05 difficulty episodes where 9s
        # is far more slack than needed (loitering pays for exactly nothing
        # useful there); this one is sized to THIS episode's own spawn
        # altitude, so easy/low-altitude episodes get little or no grace
        # (nothing to fix there) while only genuinely-high spawns get the
        # extra patience budget their own physics requires.
        loiter_grace_s = (
            self._spawn_altitude_m
            / max(cfg.soft_vertical_speed_mps, 1e-6)
            * cfg.rew_wb_loiter_altitude_grace_frac
        )
        elapsed_s = self.episode_length_buf.float() * dt
        loiter_active = (elapsed_s > loiter_grace_s).float()
        # Zeroed in stage1_nav_only mode: this penalizes not having LANDED
        # yet, which has no meaning when landing isn't part of the task.
        terms["loiter"] = (
            0.0
            if cfg.stage1_nav_only
            else -dt
            * cfg.rew_wb_loiter_penalty_per_s
            * (~termination["landed"]).float()
            * loiter_active
        )
        # slammed (landed too fast) previously shared "failed"'s flat
        # rew_crash with no quality credit at all -- identical to flying off
        # out of bounds. That gave zero signal distinguishing a well-placed,
        # upright touchdown that was merely a bit too fast from a total miss,
        # which likely pushed the policy toward excess caution on final
        # descent. landing_quality's own vertical-speed term already decays
        # sharply past crash_vertical_speed_mps, so reusing harsh's
        # quality bonus here can't let a slam farm a positive net reward --
        # worst case (bad position, fast) stays at rew_crash; best case (good
        # position, just a bit too fast) lands around rew_crash + ~0.7 *
        # rew_landing_quality, still clearly negative.
        left_bounds = termination["left_bounds"].float()
        slammed = termination["slammed"].float()
        if cfg.stage1_nav_only:
            # Stage 1 of the staged curriculum (see cfg.stage1_nav_only):
            # keep the escape/out-of-bounds AND hard-impact penalties --
            # flying away and slamming into the ground are both real
            # navigation/control failures worth penalizing at the terminal
            # level, not just through the dense foot_load estimate -- but
            # drop the landing-QUALITY payouts (soft/harsh bonus, timeout),
            # which reward or punish something this stage never attempts.
            # FIXED 2026-09-18: an earlier version of this branch omitted
            # `slammed` entirely, so a hard ground impact only cost the
            # single-step foot_load estimate while escaping cost a full
            # rew_crash -- on the very first live run this let
            # failed_slammed_fraction/failed_left_bounds_fraction swing from
            # 0.74/0.26 to 0.24/0.76 within 700k steps, consistent with the
            # policy finding escaping cheaper than a hard landing under that
            # lopsided accounting.
            terminal = (left_bounds + slammed) * self.cfg.rew_crash
        else:
            terminal = (
                termination["soft"].float() * self.cfg.rew_soft_landing
                + termination["harsh"].float()
                * (
                    self.cfg.rew_harsh_landing
                    + self.cfg.rew_landing_quality * termination["landing_quality"]
                )
                + slammed
                * (
                    self.cfg.rew_crash
                    + self.cfg.rew_landing_quality * termination["landing_quality"]
                )
                + left_bounds * self.cfg.rew_crash
                + termination["timeout_only"].float() * self.cfg.rew_timeout
            )
        terms["terminal"] = terminal
        reward = sum(terms.values())
        terms["total"] = reward
        for name, value in terms.items():
            self._episode_reward_sums[name] += value
        self._telemetry_log_step(pos, quat, lin_vel, ang_vel, distance_xy, altitude)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self._rocket.data.root_pos_w
        quat = _xyzw_to_wxyz(self._rocket.data.root_quat_w)
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
        gate_distance = distance_xy <= self.cfg.target_radius_m
        gate_horizontal = horizontal_speed <= self.cfg.soft_horizontal_speed_mps
        gate_vertical = vertical_speed <= self.cfg.soft_vertical_speed_mps
        gate_angular = angular_speed <= self.cfg.soft_angular_speed_rps
        # tilt_penalty is judged against the local terrain normal near the
        # ground (see _tilt_against_terrain), so this gates the residual
        # attitude error after banking to match the slope -- not raw
        # world-vertical tilt, which real terrain often can't satisfy.
        gate_tilt = tilt_penalty <= tilt_limit
        # On a natural slope the first foot touches before the highest foot.
        # Judge whether the footprint fits the terrain by the spread, not by
        # absolute clearance (which also includes touchdown margin). A rocket
        # correctly banked to the local slope keeps this small regardless of
        # the slope's magnitude; see tests/test_terrain_landability.py for the
        # curvature-residual proof.
        gate_foot = foot_clearance_spread <= self.cfg.landing_max_foot_clearance_m
        soft = (
            landed
            & gate_distance
            & gate_horizontal
            & gate_vertical
            & gate_angular
            & gate_tilt
            & gate_foot
        )
        landing_quality = (
            0.45 * torch.exp(-distance_xy / 2.0)
            + 0.35 * torch.exp(-vertical_speed / 0.75)
            + 0.10 * torch.exp(-horizontal_speed / 0.75)
            + 0.10 * torch.exp(-tilt_angle / math.radians(8.0))
        )
        # Whiteboard "Kisitlar": dikey hiz < 1. A touchdown faster than that is
        # a structural failure, not merely a low-quality landing.
        slammed = landed & (vertical_speed > self.cfg.crash_vertical_speed_mps)
        left_bounds = (out_of_bounds | escaped) & ~landed
        failed = left_bounds | slammed
        harsh = landed & ~soft & ~slammed
        terminated = landed | failed
        self._last_termination_metrics = {
            "soft": soft.detach(),
            "harsh": harsh.detach(),
            "failed": failed.detach(),
            "landed": landed.detach(),
            "out_of_bounds": out_of_bounds.detach(),
            # Failure-type breakdown -- "failed" conflates two very different
            # things (flying off / out of bounds vs. a too-fast touchdown);
            # see _reset_idx's Metrics/failed_slammed_fraction /
            # failed_left_bounds_fraction logging.
            "slammed": slammed.detach(),
            "left_bounds": left_bounds.detach(),
            "timeout_only": (timeout & ~terminated).detach(),
            "distance_xy": distance_xy.detach(),
            "horizontal_speed": horizontal_speed.detach(),
            "vertical_speed": vertical_speed.detach(),
            "angular_speed": angular_speed.detach(),
            "tilt_penalty": tilt_penalty.detach(),
            "landing_quality": landing_quality.detach(),
            "foot_clearance_spread": foot_clearance_spread.detach(),
            # Per-gate pass/fail, to diagnose WHICH soft-landing constraint is
            # the actual bottleneck among episodes that do touch down (landed)
            # but miss "soft" -- see _reset_idx's Metrics/gate_* logging.
            "gate_distance": gate_distance.detach(),
            "gate_horizontal": gate_horizontal.detach(),
            "gate_vertical": gate_vertical.detach(),
            "gate_angular": gate_angular.detach(),
            "gate_tilt": gate_tilt.detach(),
            "gate_foot": gate_foot.detach(),
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
            # Breakdown of WHICH kind of failure: flying off / out of bounds
            # vs. a too-fast (slammed) touchdown. As a fraction of ALL failed
            # episodes (not all episodes), so the two always sum to 1.0.
            failed_ids = completed_env_ids[metrics["failed"][completed_env_ids]]
            if len(failed_ids) > 0:
                log["Metrics/failed_slammed_fraction"] = (
                    metrics["slammed"][failed_ids].float().mean().item()
                )
                log["Metrics/failed_left_bounds_fraction"] = (
                    metrics["left_bounds"][failed_ids].float().mean().item()
                )
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
            # Among episodes that actually touched down (landed, whether soft
            # or harsh), what fraction pass EACH individual soft-landing gate
            # -- pinpoints which constraint is the real bottleneck instead of
            # guessing from the combined soft/harsh rates alone.
            landed_ids = completed_env_ids[metrics["landed"][completed_env_ids]]
            if len(landed_ids) > 0:
                for gate in (
                    "gate_distance",
                    "gate_horizontal",
                    "gate_vertical",
                    "gate_angular",
                    "gate_tilt",
                    "gate_foot",
                ):
                    log[f"Metrics/{gate}_pass_rate"] = metrics[gate][landed_ids].float().mean().item()
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
        # Geometric (spawn/target reach) ramp deliberately slower than the
        # RCS-authority ramp in _pre_physics_step, which stays linear in
        # `difficulty` -- see curriculum_distance_ramp_power's cfg comment
        # for the TensorBoard evidence (success_rate 0.336 -> 0.14-0.18
        # plateau, never recovering, immediately after curriculum_level
        # crossed 0.2 under the archived TVC gimbal-authority ramp) that a
        # linear geometric ramp outpaces a still-mostly-floor attitude
        # authority early in the curriculum.
        difficulty_geo = torch.pow(difficulty, self.cfg.curriculum_distance_ramp_power)
        target_xy_range = self.cfg.target_xy_min_range_m + difficulty_geo * (
            self.cfg.target_xy_range_m - self.cfg.target_xy_min_range_m
        )
        target_offset = (
            torch.empty(count, 2, device=self.device).uniform_(-1.0, 1.0)
            * target_xy_range.unsqueeze(-1)
        )
        target_xy = origins[:, :2] + target_offset
        spawn_xy_range = self.cfg.spawn_xy_min_range_m + difficulty_geo * (
            self.cfg.spawn_xy_range_m - self.cfg.spawn_xy_min_range_m
        )
        spawn_offset = (
            torch.empty(count, 2, device=self.device).uniform_(-1.0, 1.0)
            * spawn_xy_range.unsqueeze(-1)
        )
        # Feasibility clip: spawn_xy_range and the curriculum's commandable
        # RCS authority (_pre_physics_step's rcs_frac) both ramp with
        # `difficulty`, but nothing previously checked they ramp at MATCHED
        # rates -- a curriculum level can hand out a spawn offset the
        # currently-available lateral acceleration cannot null out in the
        # episode's time budget, which is a physically un-landable draw, not
        # a policy failure. That silently inflates timeout_rate. Bound the
        # sampled horizontal offset to what a bang-bang (accelerate then
        # brake) maneuver can cover at THIS env's own attitude authority
        # within curriculum_feasible_time_frac of the episode: for max
        # lateral accel a = g*tan(max_effective_tilt_deg) and travel time t,
        # reachable distance is a*(t/2)^2. Direction is preserved, only
        # magnitude is capped. max_effective_tilt_deg is NOT physically
        # derived from RCS authority the way the old gimbal angle was -- see
        # its cfg comment -- so this clip is an untuned placeholder pending
        # real RCS telemetry.
        max_lateral_accel = abs(float(self.cfg.sim.gravity[2])) * math.tan(
            math.radians(self.cfg.max_effective_tilt_deg)
        )
        travel_time_s = self.cfg.episode_length_s * self.cfg.curriculum_feasible_time_frac
        max_reach_m = max_lateral_accel * (travel_time_s / 2.0) ** 2
        spawn_dist = torch.linalg.norm(spawn_offset, dim=-1)
        reach_scale = torch.clamp(max_reach_m / torch.clamp(spawn_dist, min=1e-6), max=1.0)
        spawn_offset = spawn_offset * reach_scale.unsqueeze(-1)
        # P1 fix (2026-09-18): was linear in `difficulty` while spawn/target
        # XY range switched to difficulty_geo (difficulty**power) earlier --
        # that made curriculum_level a near-pure ALTITUDE ramp (at diff=0.2,
        # spawn_off_max was only 0.48m but altitude was already 5.4m).
        # Per-difficulty TensorBoard breakdown on the 07-46-02 run showed the
        # collapse starting exactly there: mean_episode_length_s jumped
        # 7.4s->15.6s and Episode_Reward/terminal flipped +95->-78 the
        # moment altitude crossed ~5m, while every soft-landing gate was
        # still >70% -- horizontal precision was never the bottleneck at
        # that difficulty, vertical descent time-vs-loiter was. Switching
        # altitude to the same difficulty_geo keeps it in step with the
        # (already fixed) geometric ramp instead of outracing it.
        spawn_altitudes = self.cfg.spawn_altitude_min_m + difficulty_geo * (
            self.cfg.spawn_altitude_m - self.cfg.spawn_altitude_min_m
        )
        self._randomize_terrain(env_ids, origins[:, :2])
        log = self.extras.setdefault("log", {})
        log["Curriculum/spawn_reach_clip_fraction"] = (reach_scale < 1.0).float().mean().item()
        root_state[:, :2] = target_xy + spawn_offset
        root_state[:, 2] = self._upright_root_height(root_state[:, :2]) + spawn_altitudes
        # write_root_pose_to_sim expects (x, y, z, w); this is identity (no rotation).
        # A (w,x,y,z)-style (1,0,0,0) here is silently read back as x=1,w=0 --
        # a 180 deg roll about world X, i.e. the rocket spawns upside down.
        root_state[:, 3:7] = torch.tensor((0.0, 0.0, 0.0, 1.0), device=self.device)
        root_state[:, 7:] = 0.0
        # "Orbital mechanics" spawn cue: instead of spawning stationary, give
        # the rocket a randomized horizontal drift velocity to null out
        # during descent -- like an approach/deorbit trajectory rather than
        # a free hover-drop. Ramped by difficulty_geo like spawn/target
        # range (0 at difficulty=0); vertical/angular velocity stay at 0.
        spawn_speed = self.cfg.spawn_lateral_speed_min_mps + difficulty_geo * (
            self.cfg.spawn_lateral_speed_max_mps - self.cfg.spawn_lateral_speed_min_mps
        )
        spawn_heading = torch.empty(count, device=self.device).uniform_(0.0, 2.0 * torch.pi)
        root_state[:, 7] = spawn_speed * torch.cos(spawn_heading)
        root_state[:, 8] = spawn_speed * torch.sin(spawn_heading)
        self._rocket.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._rocket.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

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
        self._spawn_altitude_m[env_ids] = spawn_altitudes

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

    def _foot_normal_forces(self) -> torch.Tensor:
        """Per-foot normal force in Newtons, shape (num_envs, num_feet).

        Two sources, in order of preference:

        1. The real contact sensor (``self._contact_sensor``), when the scene
           actually has collision geometry under the vehicle. Isaac Lab's
           ContactSensor only supports *body-level* sensing, and this rocket's
           four feet are geoms on the single ``hopper`` body (see
           assets/rocket/hopper_lunar.xml), so the sensor reports one aggregate
           contact force for the whole vehicle. It is split across the feet by
           how close each foot is to the surface, which is a model, not a
           measurement -- true per-foot load cells need each foot promoted to
           its own rigid body in the asset.
        2. A momentum estimate, used when there is no contact sensor or no
           collision geometry to touch. The default GPU terrain-pool path is
           analytic: terrain height is a math function, the only real collision
           plane sits at z=-100, and touchdown is decided by comparing computed
           clearance against a threshold -- nothing ever physically collides,
           so a contact sensor there reads exactly zero. The estimate is the
           peak force a touchdown at the current descent rate *would* produce,
           F = m*|vz| / (t_contact * n_feet), which is what the foot-load term
           is meant to discourage anyway.
        """
        num_feet = self._foot_offsets_b.shape[0]
        sensor = getattr(self, "_contact_sensor", None)
        if sensor is not None:
            # (num_envs, num_bodies, 3) -> total contact force magnitude per env
            net_forces = sensor.data.net_forces_w
            total_force = torch.linalg.norm(net_forces, dim=-1).sum(dim=-1)
            if bool(torch.any(total_force > 0.0)):
                pos = self._rocket.data.root_pos_w
                quat = _xyzw_to_wxyz(self._rocket.data.root_quat_w)
                clearances = torch.clamp(self._foot_clearances(pos, quat), min=0.0)
                # Lowest foot carries the most load; weights sum to 1 per env.
                weights = 1.0 / (clearances + 1e-3)
                weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-6)
                return total_force.unsqueeze(-1) * weights
        vertical_speed = torch.abs(self._rocket.data.root_lin_vel_w[:, 2])
        contact_time = max(float(self.cfg.rew_wb_foot_contact_time_s), 1e-4)
        estimated = self._rocket_mass_kg * vertical_speed / (contact_time * num_feet)
        return estimated.unsqueeze(-1).expand(-1, num_feet)

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


def _xyzw_to_wxyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    # Isaac Lab's ArticulationData.root_quat_w / write_root_pose_to_sim_index
    # use (x, y, z, w) (see base_rigid_object_data.py's QUAT_XYZW_ELEMENT_NAMES
    # and base_rigid_object.py's write_root_pose_to_sim_index docstring).
    # Every helper below (_quat_rotate etc.) expects (w, x, y, z) -- convert
    # once at this read boundary. Getting this wrong silently spawns the
    # rocket rotated 180 deg about the world X axis (upside down): an
    # (x,y,z,w)=(0,0,0,1) identity misread as (w,x,y,z) becomes w=0, x=0,
    # y=0, z=1 -- a 180 deg yaw, not upside-down -- but the reverse mistake
    # (writing an intended (w,x,y,z) identity (1,0,0,0) into an (x,y,z,w)
    # slot) is read back as x=1, w=0: a 180 deg roll about X.
    return quat_xyzw[..., (3, 0, 1, 2)]


def _wxyz_to_xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    return quat_wxyz[..., (1, 2, 3, 0)]


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
