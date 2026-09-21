from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_LUNAR_LANDER_DIR = ROOT / "source/lunar_rocket_lab/lunar_rocket_lab/tasks/direct/lunar_lander"
ENV_SOURCE = _LUNAR_LANDER_DIR / "lunar_lander_env.py"
# LunarLanderEnvCfg lives in its own module, separate from the runtime env
# class (Isaac Lab convention -- see lunar_lander_env.py's import comment):
# gym.register's env_cfg_entry_point must resolve without ever importing
# DirectRLEnv/pxr. Most assertions below don't care which of the two files a
# literal lives in, so tests read the concatenation of both.
CFG_SOURCE = _LUNAR_LANDER_DIR / "lunar_lander_env_cfg.py"
TASK_INIT = _LUNAR_LANDER_DIR / "__init__.py"
TRAIN_SAC_SOURCE = ROOT / "source/lunar_rocket_lab/scripts/train_sac.py"


def _combined_env_source() -> str:
    return ENV_SOURCE.read_text() + "\n" + CFG_SOURCE.read_text()


class IsaacLabStaticTests(unittest.TestCase):
    def test_isaaclab_env_uses_terrain_relative_observations_and_dones(self) -> None:
        source = _combined_env_source()

        self.assertIn("observation_space = 137", source)
        self.assertIn("foot_clearances = self._foot_clearances(pos, quat)", source)
        self.assertIn("altitude = torch.amin(foot_clearances, dim=-1)", source)
        self.assertIn("terrain_metrics = self._terrain_metrics", source)
        self.assertIn("lidar = self._lidar_ranges", source)
        self.assertIn("terrain_scan = self._terrain_scan", source)
        self.assertIn("sensor_features = self._sensor_features", source)
        self.assertIn("lidar_ray_count = 64", source)
        self.assertIn("rocket_ground_clearance_m = 0.38", source)
        self.assertIn("rocket_foot_radius_m = 0.025", source)
        self.assertIn("def _upright_root_height(", source)
        self.assertIn("foot_clearance_spread <= self.cfg.landing_max_foot_clearance_m", source)

    def test_isaaclab_thrust_is_body_frame_and_rcs_generates_torque(self) -> None:
        source = _combined_env_source()

        # Fixed main engine: no gimbal, force line passes through the body
        # Z-axis so it contributes zero torque (TVC's lever-arm torque calc
        # is gone).
        self.assertIn("engine_thrust_dir_b", source)
        self.assertIn("engine_dir_w = _quat_rotate(quat, self._engine_thrust_dir_b", source)
        self.assertIn("torque_w = torch.zeros_like(force_w)", source)
        # RCS ring: 8 independent fixed-direction jets, each contributing a
        # torque via its own fixed lever arm.
        self.assertIn("rcs_thruster_layout", source)
        self.assertIn("rcs_duty = torch.clamp(self._actions[:, 1:], 0.0, 1.0)", source)
        self.assertIn("torque_w = torque_w + torch.cross(pos_w_i, force_w_i, dim=-1)", source)
        self.assertIn("is_global=True", source)
        self.assertIn("hover_throttle * (raw_throttle + 1.0)", source)
        self.assertIn("max_commanded_throttle = 0.35", source)

    def test_reward_follows_the_whiteboard_model_and_logs_completed_episodes(self) -> None:
        source = _combined_env_source()

        self.assertNotIn("rew_target =", source)
        # Every symbol of the hand-derived formula has a cfg field and a term.
        for field in (
            "rew_wb_proximity_alpha",
            "rew_wb_proximity_beta",
            "rew_wb_tilt_t0",
            "rew_wb_tilt_altitude_zeta",
            "rew_wb_foot_load_h",
            "rew_wb_foot_load_c",
            "rew_wb_foot_force_max_n",
            "rew_wb_throttle_s0",
            "rew_wb_altitude_k",
            "rew_wb_rcs_d",
            "rew_wb_rel_x_power",
            "rew_wb_rel_y_power",
            "rew_wb_vxy_power",
        ):
            self.assertIn(field, source, f"missing whiteboard reward field {field}")
        for term in ("proximity", "tilt", "foot_load", "throttle", "altitude", "rcs", "rel_xy", "velocity"):
            self.assertIn(f'"{term}"', source, f"missing whiteboard reward term {term}")
        # Dense terms stay per-second so an early crash never pays off by
        # cutting off the penalty stream.
        self.assertIn("dt = float(self.step_dt)", source)
        # theta is measured against the terrain normal, not world vertical.
        self.assertIn("_, tilt_angle = self._tilt_against_terrain(quat, near_ground)", source)
        # Foot load comes from the contact sensor where it can, momentum
        # otherwise -- never silently from nothing.
        self.assertIn("def _foot_normal_forces(", source)
        self.assertIn("contact_sensor_enabled = True", source)
        self.assertIn("curriculum_success_threshold", source)
        self.assertIn("policy_max_rcs_thrust_n", source)
        self.assertIn('log["Curriculum/difficulty"]', source)
        self.assertIn('log["Metrics/success_rate"]', source)
        self.assertIn('terms["terminal"] = terminal', source)

    def test_whiteboard_landing_gates_match_the_board_constraints(self) -> None:
        source = _combined_env_source()

        # "Kisitlar" block: theta +/-15 deg, dikey hiz < 1 (crash above),
        # yatay ve acisal hiz < 0.5, bacaklar arasi 0.16 m. soft_horizontal
        # and soft_vertical are deliberate, user-approved deviations from the
        # board's literal soft-landing numbers (see the cfg comments) --
        # gate diagnostics across several full runs showed gate_horizontal
        # and then gate_vertical as binding constraints keeping success_rate
        # plateaued well below the goal with the other gates intact.
        # soft_vertical_speed_mps was moved toward, not past, the board's own
        # hard crash_vertical_speed_mps = 1.0 limit.
        self.assertIn("soft_tilt_deg = 15.0", source)
        self.assertIn("soft_vertical_speed_mps = 0.6", source)
        self.assertIn("soft_horizontal_speed_mps = 0.7", source)
        self.assertIn("soft_angular_speed_rps = 0.5", source)
        self.assertIn("landing_max_foot_clearance_m = 0.16", source)
        self.assertIn("crash_vertical_speed_mps = 1.0", source)
        # Slamming in above the vertical-speed constraint is a failure, not a
        # merely-harsh landing that could still collect the quality bonus.
        self.assertIn("slammed = landed & (vertical_speed > self.cfg.crash_vertical_speed_mps)", source)
        self.assertIn("harsh = landed & ~soft & ~slammed", source)

    def test_isaaclab_uses_exact_xml_derived_articulation(self) -> None:
        source = _combined_env_source()

        self.assertIn('assets/rocket/hopper_lunar.usd', source)
        self.assertIn("scale=(1.0, 1.0, 1.0)", source)
        self.assertIn('rocket_cfg.spawn.spawn_path = "/World/envs/env_0/Rocket"', source)
        self.assertIn("self._rocket = Articulation(rocket_cfg)", source)
        # No actuators: the RCS ring and fixed main engine are wrench-composer
        # forces, not PhysX-driven joints (see rcs_thruster_layout).
        self.assertIn("actuators={}", source)
        self.assertIn('find_bodies("hopper")', source)
        self.assertNotIn("PreviewLandingPad", source)
        self.assertNotIn("PreviewLandingTarget", source)
        self.assertNotIn("ViewportBody", source)

    def test_episode_terrain_randomization_preserves_natural_target_surface(self) -> None:
        source = _combined_env_source()

        self.assertIn("self._randomize_terrain(env_ids, origins[:, :2])", source)
        self.assertIn("self._maybe_swap_terrain_pool()", source)
        self.assertIn("ThreadPoolExecutor", source)
        self.assertIn('self._terrain_pool["crater_xy"][slots]', source)
        self.assertIn('self._terrain_pool["base_height"][slots]', source)
        self.assertIn("NASA_DEM=", source)
        self.assertIn("target_xy = origins[:, :2] + target_offset", source)
        self.assertIn("return raw_height + self._local_detail_height", source)
        self.assertNotIn("center_height + blend * (raw_height - center_height)", source)
        self.assertIn("def _local_detail_height(", source)
        self.assertIn("ISAACLAB_USE_DEM_TERRAIN", source)
        self.assertIn("repeat_interleave(samples_per_env)", source)
        self.assertNotIn("self._procedural_height_raw(origin_xy, env_ids)", source)

    def test_physics_domain_randomization_covers_mass_and_thrust_per_env(self) -> None:
        source = _combined_env_source()

        self.assertIn("physics_randomization_enabled = True", source)
        self.assertIn("mass_randomization_range = (0.85, 1.15)", source)
        self.assertIn("thrust_randomization_range = (0.85, 1.15)", source)
        self.assertIn(
            'self._rocket_mass_kg = torch.full((self.num_envs,), float(cfg.rocket_mass_kg), device=self.device)',
            source,
        )
        self.assertIn(
            'self._max_thrust_n = torch.full((self.num_envs,), float(cfg.max_thrust_n), device=self.device)',
            source,
        )
        # Every physics consumer must read the per-env tensor, never the cfg
        # scalar directly -- otherwise the randomization is sampled but unused.
        self.assertNotIn("/ self.cfg.rocket_mass_kg", source)
        self.assertNotIn("* self.cfg.max_thrust_n", source)
        self.assertIn("hover_throttle = self._rocket_mass_kg * abs(float(self.cfg.sim.gravity[2])) / self._max_thrust_n", source)
        self.assertIn("force_w = engine_dir_w * (self._actions[:, 0:1] * self._max_thrust_n.unsqueeze(-1))", source)
        self.assertIn("estimated = self._rocket_mass_kg * vertical_speed / (contact_time * num_feet)", source)

    def test_residual_action_is_a_disabled_by_default_scaffold(self) -> None:
        source = _combined_env_source()

        self.assertIn("use_residual_action = False", source)
        self.assertIn("residual_gain = 0.35", source)
        # The guard must be the first thing _pre_physics_step does, so the
        # existing end-to-end action path is untouched (bit-identical) while
        # the flag stays False -- it is the A/B baseline for B1.
        pre_physics_step = source.split("def _pre_physics_step(self, actions: torch.Tensor) -> None:")[1]
        guard, _, rest = pre_physics_step.partition("self._prev_actions.copy_(self._actions)")
        self.assertIn("if self.cfg.use_residual_action:", guard)
        self.assertIn("raise NotImplementedError", guard)

    def test_lidar_ablation_zeroes_signal_without_changing_observation_space(self) -> None:
        source = _combined_env_source()

        self.assertIn('os.environ.get("ISAACLAB_ABLATE_LIDAR", "0")', source)
        self.assertIn("if self._ablate_lidar:", source)
        self.assertIn("lidar = torch.zeros_like(lidar)", source)
        # The ablation must not shrink the observation tensor -- only zero its
        # content -- so a B2 comparison run (R2) is architecture-identical to
        # the baseline (R0) and isolates the information-content variable.
        self.assertIn("observation_space = 137", source)

    def test_td3_and_ddpg_algo_arms_are_registered_and_selectable(self) -> None:
        task_init = TASK_INIT.read_text()
        train_sac_source = TRAIN_SAC_SOURCE.read_text()

        self.assertIn('"sb3_td3_cfg_entry_point": f"{agents.__name__}:sb3_td3_cfg.yaml"', task_init)
        self.assertIn('"sb3_ddpg_cfg_entry_point": f"{agents.__name__}:sb3_ddpg_cfg.yaml"', task_init)
        self.assertTrue((ROOT / "source/lunar_rocket_lab/lunar_rocket_lab/tasks/direct/lunar_lander/agents/sb3_td3_cfg.yaml").exists())
        self.assertTrue((ROOT / "source/lunar_rocket_lab/lunar_rocket_lab/tasks/direct/lunar_lander/agents/sb3_ddpg_cfg.yaml").exists())
        # train_sac.py must select the SB3 algorithm class from the agent
        # config's "algo" key (SAC config has none -> defaults to SAC), not
        # hardcode SAC, or the new yaml arms silently still train SAC.
        self.assertIn('from stable_baselines3 import DDPG, SAC, TD3', train_sac_source)
        self.assertIn('agent_cfg.pop("algo", "SAC")', train_sac_source)
        self.assertIn('algo_classes = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG}', train_sac_source)
        self.assertIn("algo_cls = algo_classes[algo_name]", train_sac_source)
        self.assertNotIn("agent = SAC(policy_arch", train_sac_source)

    def test_default_training_configuration_uses_the_gpu_terrain_pool(self) -> None:
        # The real USD/DEM mesh only ever spawns up to legacy_terrain_usd_max_envs
        # (16) envs; at higher env counts it silently falls back to a ground
        # plane with zero terrain variety. The GPU terrain-pool path is what
        # reward/termination/observations actually use, so it must be the
        # default, at a high env count for SAC's update-to-data ratio.
        train_script = (ROOT / "scripts/train_isaaclab_docker.sh").read_text()

        self.assertIn('NUM_ENVS="${ISAACLAB_NUM_ENVS:-512}"', train_script)
        self.assertIn('TERRAIN_LOCAL_DETAIL="${ISAACLAB_TERRAIN_LOCAL_DETAIL:-0}"', train_script)
        self.assertIn('USE_DEM_TERRAIN="${ISAACLAB_USE_DEM_TERRAIN:-0}"', train_script)
        self.assertIn("-e ISAACLAB_USE_DEM_TERRAIN=", train_script)
