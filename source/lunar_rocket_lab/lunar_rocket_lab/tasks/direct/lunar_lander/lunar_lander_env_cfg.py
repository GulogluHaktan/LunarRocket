from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
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
    observation_space = 125
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
        # The real collision mesh is 80x80 m. Keep cloned surfaces disjoint so
        # one environment can never contact a neighbour's translated terrain.
        env_spacing=90.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    rocket: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Rocket",
        spawn=sim_utils.UsdFileCfg(
            # This USD is the checked-in Isaac conversion of
            # assets/rocket/hopper_lunar.xml. Keep its authored 1:1 scale.
            usd_path=str(_project_root() / "assets/rocket/hopper_lunar.usd"),
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # Required for the foot-load ContactSensor: without it PhysX never
            # applies the contact reporter API to the body and sensor init
            # fails outright ("could not find any bodies with contact reporter API").
            activate_contact_sensors=True,
            # fix_root_link moved off UsdFileCfg into the (solver-common)
            # articulation root properties as of Isaac Lab >=3.0 -- the
            # rocket must stay a floating base (it flies), not fixed to world.
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 25.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "tvc": ImplicitActuatorCfg(
                joint_names_expr=["tvc_.*_joint"],
                effort_limit_sim=2.5,
                stiffness=10.0,
                damping=0.1,
                armature=0.002,
            ),
        },
    )

    # 1.63 kg hopper + 0.01/0.02 kg authored TVC links.
    rocket_mass_kg = 1.66
    max_thrust_n = 30.0
    max_commanded_throttle = 0.35
    # Per-env physical domain randomization (A1): mass and max thrust are
    # resampled per environment on every reset within these multiplicative
    # ranges, so the policy cannot rely on flying an exact, fixed vehicle.
    # Disable to reproduce the pre-randomization baseline for A/B comparison.
    physics_randomization_enabled = True
    mass_randomization_range = (0.85, 1.15)
    thrust_randomization_range = (0.85, 1.15)
    # Residual action space (B1, scaffold only -- see README roadmap notes).
    # The intent is a_cmd = a_analytic_guidance(pos, vel, t_go) + residual_gain
    # * pi_theta(obs), so the policy learns a correction on top of a guidance
    # law instead of the raw gimbal/throttle command from scratch. Left False
    # by default so the existing end-to-end action path is the A/B baseline;
    # turning it on currently raises NotImplementedError (see
    # _pre_physics_step) because mapping the reward's world-frame guidance
    # target into a body-frame gimbal/throttle command depends on the current
    # attitude and has not been validated against real physics yet -- do not
    # implement the analytic term without an Isaac Lab GPU smoke test.
    use_residual_action = False
    residual_gain = 0.35
    max_gimbal_deg = 25.0
    # The XML joint retains its physical +/-25 degree hard stop. The policy
    # command envelope ramps from a conservative low-authority range up to the
    # full trained range as the curriculum difficulty increases, so an early,
    # untrained policy cannot saturate the TVC into large destabilizing torques.
    policy_min_gimbal_deg = 4.0
    policy_max_gimbal_deg = 8.0
    # Three separate runs (varying entropy-floor settings, ruling out entropy
    # collapse as the cause) all collapsed the moment curriculum_level reached
    # this value: success 46-76% -> single digits, timeout_rate -> 55-90%,
    # sustained for millions of steps with no recovery. Per-episode difficulty
    # is torch.rand()*curriculum_level (see _reset_idx), so at
    # curriculum_full_gimbal_difficulty == curriculum_level, the hardest
    # episodes in the batch hit hard 8 deg gimbal saturation for the first
    # time -- while target/spawn distance ranges keep scaling smoothly past
    # that point with no comparable break. Spreading the ramp across the
    # entire curriculum range (instead of saturating in the first 30%)
    # removes that artificial saturation point.
    curriculum_full_gimbal_difficulty = 1.0
    # XML: tvc_yaw z=-0.150, thrust_site z=-0.100 in the nested pitch body.
    thrust_lever_arm_m = 0.25
    # XML: foot centers z=-0.355, sphere radius=0.025.
    rocket_ground_clearance_m = 0.38
    rocket_foot_radius_m = 0.025
    rocket_foot_offsets_m = (
        (0.4, 0.0, -0.355),
        (-0.4, 0.0, -0.355),
        (0.0, 0.4, -0.355),
        (0.0, -0.4, -0.355),
    )
    spawn_altitude_m = 25.0
    spawn_altitude_min_m = 0.5
    spawn_xy_range_m = 6.0
    spawn_xy_min_range_m = 0.25
    target_xy_range_m = 14.0
    target_xy_min_range_m = 1.0
    curriculum_enabled = True
    curriculum_initial_difficulty = 0.05
    curriculum_success_threshold = 0.35
    # A per-run minimum; the effective window also scales with num_envs (see
    # __init__) so it isn't satisfied by a handful of reset calls at high env
    # counts. At 512 envs and short early-difficulty episodes, a raw window of
    # 64 filled in well under a second of sim time -- letting curriculum
    # difficulty race to ~1.0 in a few million steps while success at that
    # difficulty was still single digits, far outpacing what the policy had
    # actually learned. A run that hit this raced ~4M steps to 95%+ difficulty
    # and then measurably regressed (25% success -> 4-7.5%) instead of improving.
    curriculum_window_episodes = 64
    curriculum_increment = 0.05
    target_radius_m = 1.0
    max_xy_m = 28.0
    max_altitude_m = 60.0
    max_lidar_distance_m = 30.0
    lidar_ray_count = 64
    terrain_scan_count = 24
    landing_altitude_m = 0.04
    # Whiteboard "Kısıtlar" block: bacaklar arası açıklık 0.16 m, dikey hız < 1,
    # yatay ve açısal hız < 0.5, theta = +/-15 deg aralıkta inebilir.
    landing_max_foot_clearance_m = 0.16
    soft_vertical_speed_mps = 0.5
    soft_horizontal_speed_mps = 0.5
    soft_tilt_deg = 15.0
    soft_angular_speed_rps = 0.5
    # "Dikey hiz < 1": above this a touchdown is not a hard landing, it is a
    # structural failure (crash), not just a low-quality one.
    crash_vertical_speed_mps = 1.0
    terrain_height_scale_m = 0.45
    terrain_slope_scale = 0.035
    terrain_crater_count = 3
    terrain_rock_count = 5
    terrain_detail_radius_m = 8.0
    terrain_detail_amplitude_range_m = (0.006, 0.020)
    # Fine-detail fade shape: "smoothstep" (default, original behavior) |
    # "gaussian" | "hybrid". See experiments/terrain_transition/ for the
    # CPU/NumPy study this was validated against before wiring it in here.
    terrain_blend_method = "smoothstep"
    # exp(-k*(r/R)^2), k = ln(1000) so the Gaussian's practical support
    # matches terrain_detail_radius_m at the 0.1% level (see
    # experiments/terrain_transition/blend_functions.py GAUSSIAN_K_CALIBRATED).
    terrain_gaussian_k = 6.907755278982137
    # hybrid: Gaussian for r <= contact radius (landing-gear footprint +
    # touchdown-drift margin), smoothstep beyond it, joined by a
    # smoothstep-weighted switch over terrain_hybrid_switch_width_m.
    terrain_hybrid_contact_radius_m = 1.5
    terrain_hybrid_switch_width_m = 0.3
    terrain_pool_size = 15
    terrain_pool_refresh_assignments = 60
    terrain_pool_dem_quality = "low_debug"
    legacy_terrain_config_path = "configs/terrain_config.yaml"
    legacy_rocket_config_path = "configs/rocket_config.yaml"
    legacy_terrain_usd_max_envs = 16
    # Real contact sensor on the vehicle body (the board's "Ayak basinci"
    # sensor). Isaac Lab's ContactSensor is body-level only and the four feet
    # are geoms on the single "hopper" body, so this reports one aggregate
    # contact force -- see _foot_normal_forces for how it is split per foot and
    # why the analytic terrain path falls back to a momentum estimate.
    contact_sensor_enabled = True
    imu_accel_noise_std = 0.03
    imu_gyro_noise_std = 0.01
    altimeter_noise_std = 0.01
    lidar_noise_std = 0.015

    # =================================================================
    # Whiteboard reward model.  Each rew_wb_* field is one symbol from the
    # hand-derived formula:
    #
    #   rew = -alpha*(1/konum)^beta  -  T0 * e^(IMU theta) * altitude^zeta
    #         -  h * (F_ayak / F_ayak_max)^c
    #         -  s0 * throttle^2  -  k_z * |z_hedef - z|  -  |tvc|^d
    #         -  x1 * (relative_x)^x2  -  x4 * (relative_y)^x3
    #         -  x5 * (V_xy)^x6
    #         +  Landing_reward  -  Crash
    #
    # Dense terms are per-second costs (scaled by dt in _get_rewards) so a
    # early crash is never attractive merely because it stops the penalty
    # stream.  The terminal block below is a one-time per-episode payout.
    # =================================================================

    # -alpha*(1/konum)^beta.  NOTE ON SIGN: written literally, -a*(1/d)^b is
    # most negative AT the target and ~0 far away, which rewards running away.
    # Implemented as the bounded proximity *reward* +alpha*(1/(1+d))^beta
    # (1 at the target, ->0 far out, no singularity) which is what "throttle
    # ~ 1/konum" on the same board implies. To switch to a plain distance
    # penalty instead, use -alpha * d^beta in _get_rewards.
    rew_wb_proximity_alpha = 6.0
    rew_wb_proximity_beta = 1.0
    # -T0 * e^(IMU theta) * altitude^zeta.  theta is the attitude error against
    # the LOCAL TERRAIN NORMAL (see _tilt_against_terrain), not world vertical,
    # so banking to match a slope is not punished -- only the residual error is.
    # zeta = 0 disables the altitude scaling (the board's "Yukseklik^zeta").
    rew_wb_tilt_t0 = 2.0
    rew_wb_tilt_altitude_zeta = 0.0
    # -h * (F_ayak/F_ayak_max)^c.  See _get_rewards: F_ayak is ESTIMATED from
    # momentum, not measured -- there is no contact sensor on this vehicle.
    rew_wb_foot_load_h = 3.0
    rew_wb_foot_load_c = 2.0
    # Per-foot force budget, the board's "F_Bacak = [a, b]" constraint.
    # Reference point: hovering puts m*g/4 = 1.66*1.62/4 ~= 0.67 N on each foot;
    # a 0.5 m/s touchdown arrested over ~50 ms puts ~4 N on each foot.
    rew_wb_foot_force_max_n = 8.0
    rew_wb_foot_contact_time_s = 0.05
    # -s0 * throttle^2
    rew_wb_throttle_s0 = 0.05
    # -k_z * |z_hedef - z|: altitude above the target's own surface, i.e. the
    # board's "25 m -> 0 m" descent driver.
    rew_wb_altitude_k = 0.2
    # -|tvc|^d
    rew_wb_tvc_d = 2.0
    rew_wb_tvc_weight = 0.05
    # -x1*(relative_x)^x2 - x4*(relative_y)^x3: axis-separated position error,
    # kept distinct from the radial proximity term above on purpose.
    rew_wb_rel_x_weight = 0.5
    rew_wb_rel_x_power = 2.0
    rew_wb_rel_y_weight = 0.5
    rew_wb_rel_y_power = 2.0
    # -x5*(V_xy)^x6
    rew_wb_vxy_weight = 1.0
    rew_wb_vxy_power = 2.0
    # Vertical speed is not an explicit term on the board, but "Dikey hiz < 1"
    # is a hard constraint and nothing else in the formula brakes the descent.
    rew_wb_vz_weight = 4.0
    rew_wb_vz_power = 2.0
    rew_wb_angular_weight = 0.15

    rew_soft_landing = 300.0
    # A harsh touchdown must stay negative even at maximum quality; otherwise
    # SAC learns to farm "almost good" crashes instead of crossing the soft gate.
    rew_harsh_landing = -220.0
    rew_landing_quality = 160.0
    rew_crash = -220.0
    # Must stay worse than even the best-case harsh landing
    # (rew_harsh_landing + rew_landing_quality = -60) or SAC learns that never
    # attempting a landing (hover to timeout) beats a bad but real attempt.
    rew_timeout = -150.0
