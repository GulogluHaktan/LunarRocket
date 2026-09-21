from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs.direct_rl_env_cfg import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass


def _project_root() -> Path:
    return Path(os.environ.get("LUNAR_ROCKET_ROOT", Path(__file__).resolve().parents[5])).resolve()


@configclass
class LunarLanderEnvCfg(DirectRLEnvCfg):
    # Raised from 20.0: TensorBoard on the 2026-09-17_18-32-01 run showed
    # Metrics/timeout_rate (episode ends without ever reaching landing_altitude_m)
    # at ~49% last-third-average -- by far the single largest loss bucket,
    # bigger than harsh_landing_rate (~17.5%) and failure_rate (~16.4%)
    # combined. Every reward-weight/curriculum-amplitude single-parameter
    # change tried so far (loiter, gimbal floor/ceiling, readiness weight,
    # a loiter grace period -- see policy_min_gimbal_deg/rew_wb_loiter_penalty
    # _per_s/rew_wb_readiness_weight's comments below) collapsed nearly every
    # gate at once, evidence the current reward/curriculum balance is a
    # narrow, fragile optimum. This axis is deliberately orthogonal to that:
    # it changes none of the reward magnitudes those weights were calibrated
    # against, it only gives marginal attempts (still slow gimbal authority,
    # 4-8 deg) more wall-clock time to close the distance and brake before
    # the episode is cut off. The per-second loiter cost's marginal price is
    # unchanged; only true full-length timeouts (already the worst outcome)
    # accrue a larger total penalty. Untested hypothesis -- validate against
    # Metrics/success_rate and Metrics/timeout_rate on a fresh run before
    # iterating further.
    episode_length_s = 28.0
    decimation = 2
    # [throttle, rcs_0..rcs_7] -- was 3 ([throttle, yaw_gimbal, pitch_gimbal])
    # before the TVC->RCS conversion; see rcs_thruster_layout.
    action_space = 9
    # +12 over the old 125: self._actions/self._prev_actions each grew 3->9
    # in the observation concat (2 * (9 - 3) = 12), everything else unchanged.
    observation_space = 137
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
        # No actuators: the TVC gimbal servos are gone (fixed main engine, no
        # joints), and the RCS ring is 8 fixed-direction force emitters, not
        # PhysX-driven joints -- both are computed as body wrenches directly
        # in LunarLanderEnv._pre_physics_step from rcs_thruster_layout below,
        # the same way the old TVC-deflected thrust force always was.
        actuators={},
    )

    # 1.63 kg hopper. The old +0.01/0.02 kg TVC yaw/pitch link mass is gone
    # along with the gimbal bodies -- the fixed engine and RCS sites add no
    # separate mass (RCS sites are massless MJCF reference points).
    rocket_mass_kg = 1.63
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
    # ARCHIVED (pre-RCS, kept for context): every TVC gimbal-authority number
    # this project ever tried (4/8deg baseline, 6/10 and 10/15deg raises, all
    # tested and REJECTED -- raising the floor at all above 4deg collapsed
    # success_rate from ~22-27% to <1.5% by destabilizing SAC's exploration
    # noise into self-induced oscillation) is now moot: TVC coupled lateral
    # authority to g*tan(gimbal_angle) at hover throttle, capping it at
    # ~0.11 m/s^2 and explaining the gate_horizontal/gate_tilt/gate_foot
    # plateau at 60-85% across every reward configuration tried. RCS
    # decouples attitude torque from throttle entirely (see
    # rcs_thruster_layout / LunarLanderEnv._pre_physics_step), removing that
    # specific structural bound -- but the OTHER lesson from this history,
    # that handing an untrained SAC policy full actuator authority from step
    # one causes self-induced-oscillation collapse regardless of actuator
    # type, is assumed to still apply and is why an authority curriculum is
    # kept below rather than commanding full RCS thrust immediately.
    #
    # RCS authority curriculum (fresh design, replaces the gimbal-angle ramp
    # above 1:1 in SHAPE only -- same floor-to-ceiling-by-difficulty pattern,
    # now scaling max per-thruster RCS force instead of gimbal angle).
    # UNTRAINED PLACEHOLDER VALUES: nothing below this comment has been
    # validated against a real GPU training run -- they are starting points
    # only, expect to retune all of them (same iterative process documented
    # throughout this file for the reward weights) once RCS telemetry from an
    # actual run exists.
    max_rcs_thrust_n = 1.2
    policy_min_rcs_thrust_n = 0.3
    policy_max_rcs_thrust_n = 1.2
    curriculum_full_rcs_difficulty = 1.0
    # Not physically derived (see the plan this was implemented from): RCS
    # thrusters produce torque, not direct lateral thrust, so lateral motion
    # is still tilt-then-thrust as it was under TVC, just via a torque
    # impulse settling to a hold-tilt rather than a directly commanded
    # gimbal angle. This stands in for "the tilt angle the RCS ring can
    # actually hold at policy_max_rcs_thrust_n" in the feasibility-clip math
    # below (_reset_idx) until replaced with a value measured from real RCS
    # telemetry (e.g. via preview_scene.py).
    max_effective_tilt_deg = 8.0
    # Fraction of episode_length_s treated as the horizontal travel-and-brake
    # budget when clipping spawn_offset to what THIS env's own RCS-driven
    # tilt authority (max_effective_tilt_deg) can physically reach (see
    # _reset_idx's feasibility clip, right after spawn_offset is sampled).
    # The remaining (1 - this) fraction is left uncounted for final
    # approach/braking margin and vertical descent, which is
    # throttle-limited, not attitude-limited, and not part of this clip.
    curriculum_feasible_time_frac = 0.6
    # spawn_xy_range/target_xy_range interpolate on
    # difficulty**curriculum_distance_ramp_power rather than difficulty
    # directly (see _reset_idx) so early-curriculum reach requirements grow
    # more slowly than early-curriculum RCS authority, avoiding the
    # geometric-vs-authority ramp mismatch documented in the archived TVC
    # history above (there, linear/linear ramps still outpaced gimbal
    # authority and collapsed training at curriculum_level=0.2).
    # spawn_altitude_m is deliberately NOT affected -- vertical descent is
    # throttle-limited, not attitude-limited, so it was never part of this
    # reach mismatch.
    curriculum_distance_ramp_power = 2.0
    # Randomized horizontal spawn velocity ("orbital mechanics" cue): the
    # rocket no longer spawns stationary, it spawns mid-descent with a
    # lateral drift velocity to null out, like an approach/deorbit
    # trajectory rather than a free hover-drop. Ramped by difficulty_geo the
    # same way spawn/target range is (0 at difficulty=0). UNTRAINED
    # PLACEHOLDER: conservative starting magnitude, not validated against a
    # real run.
    spawn_lateral_speed_min_mps = 0.0
    spawn_lateral_speed_max_mps = 2.0
    # RCS thruster ring: 8 fixed-direction jets near the top of main_body
    # (see assets/rocket/hopper_lunar.xml's rcs_0..rcs_7 sites), firing
    # tangentially, alternating CCW/CW so the ring has bidirectional torque
    # authority about all three body axes without needing a hand-designed
    # jet-select mixing logic -- the policy commands each of the 8 jets
    # independently (see LunarLanderEnv._pre_physics_step) and learns
    # whatever combination nulls pitch/yaw/roll itself. Each entry is
    # (pos_xyz_m, dir_xyz) in the hopper body frame; dir is a unit tangent
    # vector at that position, sign alternating per the XML site comments.
    rcs_thruster_layout = (
        ((0.070, 0.000, 0.40), (0.0, 1.0, 0.0)),
        ((0.0495, 0.0495, 0.40), (0.7071, -0.7071, 0.0)),
        ((0.000, 0.070, 0.40), (-1.0, 0.0, 0.0)),
        ((-0.0495, 0.0495, 0.40), (0.7071, 0.7071, 0.0)),
        ((-0.070, 0.000, 0.40), (0.0, -1.0, 0.0)),
        ((-0.0495, -0.0495, 0.40), (-0.7071, 0.7071, 0.0)),
        ((0.000, -0.070, 0.40), (1.0, 0.0, 0.0)),
        ((0.0495, -0.0495, 0.40), (-0.7071, -0.7071, 0.0)),
    )
    # Fixed main engine: straight down the hopper's own -Z axis (no gimbal),
    # so its force line passes through the body's Z-axis and contributes
    # zero torque about the CoM -- all attitude authority comes from
    # rcs_thruster_layout above. XML: thrust_site z=-0.250.
    engine_thrust_dir_b = (0.0, 0.0, -1.0)
    engine_offset_m = 0.25
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
    # Stage 1 of a staged curriculum: 2026-09-18 evidence (see
    # rew_wb_readiness_weight/curriculum_distance_ramp_power comments) is
    # that every geometric-difficulty fix so far still runs into the same
    # wall -- success_rate erodes because the policy must learn accurate
    # attitude-controlled navigation AND all-six-gates-at-once landing
    # precision SIMULTANEOUSLY from scratch. When True, _get_rewards drops
    # every landing-outcome term (terminal soft/harsh/slammed/timeout
    # payouts, loiter, readiness) and keeps only the dense navigation/hover
    # terms (proximity, rel_xy, velocity, tilt, throttle, rcs, altitude) plus
    # the escape/out-of-bounds penalty -- the intent is to pretrain pure
    # hover-and-approach competence under RCS attitude control (checkpoint
    # it), THEN resume with this False and the full landing reward restored,
    # so the policy only has to learn the final-approach/touchdown piece on
    # top of already-solid navigation, not both at once. Off by default --
    # normal training is unaffected; opt in via ISAACLAB_STAGE1_NAV_ONLY=1.
    stage1_nav_only = False
    # Stage 2 of the staged curriculum: keeps the normal geometric
    # curriculum (spawn/target distance still ramps with success-gated
    # difficulty, see _reset_idx) but freezes RCS authority at
    # policy_max_rcs_thrust_n regardless of difficulty -- see
    # _pre_physics_step's comment. Meant to be turned on when resuming from a
    # Stage 1 (stage1_nav_only=True, curriculum_initial_difficulty=1.0)
    # checkpoint, whose actor is already calibrated to full RCS authority;
    # leaving the normal ramp on would reset that authority back to the
    # policy_min_rcs_thrust_n floor and destabilize the already-learned
    # action mapping. Off by default; opt in via
    # ISAACLAB_CURRICULUM_RCS_ALWAYS_MAX=1.
    curriculum_rcs_always_max = False
    curriculum_enabled = True
    # Lowered from 0.05 to the floor (0.0): at 0.05, spawn_altitude_m ~=
    # 1.7m and target_xy_range ~= 1.7m -- still a full navigate-then-land
    # task from the first episode. curriculum_level has never sustained
    # above this floor in this project's history (see the archived TVC
    # gimbal-authority history above for the pre-RCS consequence of that),
    # so whatever value is set here has effectively
    # been the ENTIRE training regime across every run to date, not just an
    # early phase. At 0.0, spawn_altitude_min_m=0.5m and
    # target_xy_min_range_m=1.0m put the rocket almost directly above the
    # target at low altitude -- closer to "learn to touch down cleanly
    # first" than "learn to navigate and touch down at the same time".
    # Testing whether isolating the touchdown sub-skill this way converts
    # into better simultaneous 6-gate precision than the current mix of
    # navigation + touchdown difficulty at 0.05.
    curriculum_initial_difficulty = 0.0
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
    # Raised from 1.0: per-gate diagnostics (Metrics/gate_*_pass_rate) on a
    # trained checkpoint showed gate_distance passing in ~0-25% of landed
    # episodes while gate_vertical/angular/tilt passed 30-100% -- attitude and
    # speed control were already solid, the touchdown just wasn't precise
    # enough to clear a 1 m radius even at the EASIEST curriculum difficulty
    # (target itself only spawns ~1-1.65 m from origin there). 1.0 m was
    # gating on horizontal precision the policy hadn't learned yet rather than
    # on landing quality.
    target_radius_m = 2.0
    max_xy_m = 28.0
    max_altitude_m = 60.0
    max_lidar_distance_m = 30.0
    lidar_ray_count = 64
    terrain_scan_count = 24
    landing_altitude_m = 0.04
    # Whiteboard "Kısıtlar" block: bacaklar arası açıklık 0.16 m, dikey hız < 1,
    # yatay ve açısal hız < 0.5, theta = +/-15 deg aralıkta inebilir.
    landing_max_foot_clearance_m = 0.16
    # Loosened from the board's literal 0.5: the board's own hard limit for a
    # touchdown at all (vs. a structural failure) is crash_vertical_speed_mps
    # = 1.0 -- 0.5 as the *soft*-landing bar was already a stricter-than-board
    # choice. gate_vertical was one of the two weakest gates (~0.67-0.72
    # pass rate among landed episodes, alongside gate_foot) after the
    # rew_timeout fix forced more attempts from marginal states; moving the
    # soft bar toward (not past) the board's actual 1.0 hard limit targets
    # that gate directly without exceeding what the board already allows.
    # Pulled back 0.75 -> 0.6: at 0.75 the buffer to the hard crash line
    # shrank to 0.25 m/s (from 0.5 m/s at the original 0.5), and
    # failed_slammed_fraction (structural failures, too-fast touchdown) jumped
    # from ~30-40% to 72-82% of all failures across the next two runs --
    # consistent with a policy that aims to just clear the soft gate and, with
    # normal control noise/entropy, frequently overshoots into the crash zone
    # when that gate sits too close to it. 0.6 keeps some of the loosening
    # (vs. the board's 0.5) while restoring a wider margin (0.4 m/s) to crash.
    soft_vertical_speed_mps = 0.6
    # Deliberately loosened from the board's literal 0.5: per-gate diagnostics
    # across several full runs showed gate_horizontal as the single weakest
    # soft-landing constraint (~8-30% pass rate among landed episodes) even
    # after attitude/vertical-speed/position control converged to 70-90%+ --
    # requiring ALL six gates simultaneously multiplies their individual pass
    # rates down, and success_rate plateaued at ~22-29% across three
    # consecutive full runs of pure reward-weight tuning. Loosening this one
    # threshold (user's explicit choice, made knowingly as a deviation from
    # the board's exact spec) targets that specific bottleneck directly.
    soft_horizontal_speed_mps = 0.7
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
    #         -  s0 * throttle^2  -  k_z * |z_hedef - z|  -  |rcs|^d
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
    # Bounded, always-positive attitude reward (was the board's literal
    # -T0*e^(IMU theta)*altitude^zeta penalty -- unbounded and gave SAC
    # nothing to relax toward once already stable). theta is the attitude
    # error against the LOCAL TERRAIN NORMAL (see _tilt_against_terrain), not
    # world vertical, so banking to match a slope is not punished -- only the
    # residual error is. zeta = 0 disables the altitude scaling (the board's
    # "Yukseklik^zeta").
    # Halved from 2.0: a live run showed this near its max (+31 of a possible
    # +40/episode) once the policy learned to hold a calm attitude, combining
    # with the other positive terms to out-earn the loiter penalty and keep
    # the hover-to-timeout habit alive -- see rew_wb_loiter_penalty_per_s.
    rew_wb_tilt_t0 = 1.0
    rew_wb_tilt_altitude_zeta = 0.0
    # -h * (F_ayak/F_ayak_max)^c.  See _get_rewards: F_ayak is ESTIMATED from
    # momentum, not measured -- there is no contact sensor on this vehicle.
    rew_wb_foot_load_h = 3.0
    rew_wb_foot_load_c = 2.0
    # Per-foot force budget, the board's "F_Bacak = [a, b]" constraint.
    # Reference point: hovering puts m*g/4 = 1.63*1.62/4 ~= 0.66 N on each foot;
    # a 0.5 m/s touchdown arrested over ~50 ms puts ~4 N on each foot.
    rew_wb_foot_force_max_n = 8.0
    rew_wb_foot_contact_time_s = 0.05
    # -s0 * throttle^2
    rew_wb_throttle_s0 = 0.05
    # -k_z * |z_hedef - z|: altitude above the target's own surface, i.e. the
    # board's "25 m -> 0 m" descent driver.
    rew_wb_altitude_k = 0.2
    # -|rcs|^d: control-effort/fuel penalty on the 8 RCS thruster duties,
    # same role the board's TVC deflection penalty had before the TVC->RCS
    # conversion (was rew_wb_tvc_d/rew_wb_tvc_weight, keyed off a single
    # 2-axis gimbal command; now summed over 8 independent thruster duties
    # in [0, 1] -- see _get_rewards). UNTRAINED PLACEHOLDER: not retuned for
    # the new 8-thruster action space yet.
    rew_wb_rcs_d = 2.0
    rew_wb_rcs_weight = 0.05
    # Axis-separated position term, kept distinct from the radial proximity
    # term above on purpose. Originally the board's literal -x1*(rel_x)^x2 -
    # x4*(rel_y)^x3 penalty; halving the weight (0.5 -> 0.25) fixed a wild
    # spike (-20 to over -1000 between logging windows at the SAME curriculum
    # difficulty) but three full runs (14.7M steps combined) still converged
    # to "hover near target at low speed, never touch down" -- an unbounded
    # per-axis penalty gives SAC no ceiling to relax toward once it's already
    # close, so it has nothing pulling it the rest of the way down. Rewritten
    # as a bounded, always-positive reward for closeness (mirrors the
    # proximity term's 1/(1+d) shape, just per-axis): max reward weight_*
    # exactly at rel=0, decaying toward 0 far away, never negative.
    # Halved from 1.0 alongside tilt/velocity below, same reason: too much
    # combined positive shaping for a stationary-but-close policy to out-earn
    # the loiter penalty with.
    rew_wb_rel_x_weight = 0.5
    rew_wb_rel_x_power = 2.0
    rew_wb_rel_y_weight = 0.5
    rew_wb_rel_y_power = 2.0
    # Bounded, always-positive "being slow" reward (was the board's literal
    # -x5*(V_xy)^x6 plus unbounded vertical/angular braking terms). Vertical
    # speed is not an explicit term on the board, but "Dikey hiz < 1" is a
    # hard constraint and nothing else in the formula brakes the descent.
    # Halved from 1.0/4.0/0.15: at the old weights this term alone maxed out
    # around +100/episode for a calm hover, which (combined with tilt/rel_xy)
    # comfortably out-earned rew_wb_loiter_penalty_per_s and kept the policy
    # hovering to timeout instead of landing.
    # vxy raised back to 1.0 (vz/angular left at their halved values): per-gate
    # diagnostics across two full runs showed gate_horizontal (soft_horizontal
    #_speed_mps) as the single weakest constraint on average (~8.5% pass rate
    # among landed episodes, worse than distance/foot/tilt/angular/vertical)
    # -- sharpening just the horizontal-speed term's weight targets that
    # specific bottleneck without touching the board's actual speed/tilt/
    # clearance thresholds.
    rew_wb_vxy_weight = 1.0
    rew_wb_vxy_power = 2.0
    # Raised 2.0 -> 3.0: after the rew_timeout fix (see rew_timeout) forced
    # more attempts from marginal states, gate_vertical_pass_rate became the
    # weakest gate (~0.67-0.72, alongside gate_tilt/gate_foot) among landed
    # episodes -- sharpen the vertical-speed shaping specifically since it
    # also drives failed_slammed_fraction (too-fast touchdown).
    rew_wb_vz_weight = 3.0
    rew_wb_vz_power = 2.0
    rew_wb_angular_weight = 0.08

    # Not on the board: a dense, near_ground-gated bonus for having ALL SIX
    # soft-landing axes close to their gate SIMULTANEOUSLY (see _get_rewards'
    # `readiness` -- a product of per-axis closeness scores). Added because
    # each gate individually reached 75-94% pass rate among landed episodes
    # across several full runs, yet joint success_rate plateaued at ~22-29%:
    # the axes trade off against each other during the final braking
    # maneuver, so rewarding them separately (rel_xy/velocity/tilt above)
    # cannot teach the policy to hold them all at once. This term's gradient
    # only rewards the conjunction. Lowered from 8.0: on a FRESH run (no prior
    # loiter-trained habit of committing to touchdown) this term was earnable
    # continuously while merely hovering near the ground with good-but-not-
    # landed stats -- near_ground stays close to 1 out to ~5m, so a calm,
    # centered hover could bank close to its max (~160/episode at weight 8.0)
    # and nearly cancel out rew_wb_loiter_penalty_per_s's -200/episode max
    # cost, reopening the "hover forever" local optimum loiter was added to
    # close (timeout_rate stayed ~70-89% the whole run instead of dropping
    # the way it did once loiter was tuned in the non-readiness lineage).
    # 2026-09-18 fix: raised back from 2.5 alongside narrowing this term's
    # OWN altitude gate (rew_wb_readiness_altitude_scale_m below), rather
    # than raising the old weight against the wide (5.0m) shared
    # near_ground gate that caused the original hover exploit. Evidence for
    # the raise: TensorBoard on the 2026-09-18_07-46-02 run showed this term
    # contributing only 0.4-0.8 reward/episode (vs loiter's -145 to -180 and
    # terminal's -73 to -132) -- statistically inert -- while success_rate
    # eroded in lockstep across ALL SIX gates together on every curriculum
    # difficulty step (0.15->0.2->0.25), the exact joint-simultaneity failure
    # this term exists to counteract. At the previous weight it wasn't doing
    # its job. See rew_wb_readiness_altitude_scale_m's comment for why this
    # is safe against the original 8.0 exploit this time.
    rew_wb_readiness_weight = 6.0
    # Altitude decay scale (meters) for readiness's OWN near-ground gate,
    # dedicated and much tighter than the shared near_ground (exp(-altitude
    # /5.0), used for terrain-normal blending elsewhere -- see
    # _get_rewards's near_ground_tight). At 5.0m, a calm hover at 2-5m
    # scored close to this term's max, which is what forced the weight down
    # to 2.5 previously (see rew_wb_readiness_weight's comment). At 1.2m,
    # exp(-altitude/1.2) is ~0.19 at 2m and ~0.02 at 5m -- a parked hover
    # above ~2m earns almost nothing here regardless of weight, so the
    # weight raise above only pays off in the last ~1-2m of an actual
    # descent, not a stationary hover.
    rew_wb_readiness_altitude_scale_m = 1.2

    # Not on the board, but SAC learns from dense per-step signal far better
    # than from one big number 20s later: a small flat cost for every second
    # spent not yet landed, on top of (not instead of) rew_timeout. Kept small
    # relative to rew_crash/rew_harsh_landing (-220) so it nudges the policy
    # toward attempting a landing instead of hovering to timeout, without
    # making a fast/unsafe touchdown look cheap by comparison -- a rocket that
    # rushes still risks the much larger crash penalty either way. Raised
    # again, 2.5 -> 6.0: at 1.5 a full run only partially broke hover-to-
    # timeout; at 2.5, converting tilt/velocity/rel_xy to positive bounded
    # rewards (see those fields) let a calm, close hover earn ~+95/episode of
    # combined shaping, comfortably out-earning a -50 max loiter cost. Halving
    # those weights AND raising this together should keep the max hover
    # payout (now ~+50-60/episode) below the max loiter cost (6.0*20s=-120).
    # Raised again, 6.0 -> 10.0: after several rounds of continued training,
    # per-episode attitude/speed/position control (gate_* pass rates) reached
    # 74-89% among LANDED episodes, but timeout_rate was still ~43% -- nearly
    # half of episodes never attempted touchdown at all. success_rate is the
    # product of (fraction that land) x (fraction of those that land softly),
    # and the first factor, not the second, was now the binding constraint.
    # TESTED AND REJECTED: lowered 10.0 -> 5.0, reasoning from the
    # Episode_Reward/* breakdown that loiter dwarfing the rew_crash/
    # rew_timeout gap creates a rush-to-fail incentive (crashing early costs
    # less total reward than crashing late or timing out). The theory may
    # still be correct, but the fresh run at 5.0 collapsed the same way the
    # gimbal-floor and loiter-grace-period experiments did: whole-run
    # success_rate 0.4%, gate_distance/tilt/foot fell to 6-14% (from
    # 70-85% at 10.0). This is now the FOURTH consecutive single-parameter
    # change away from the validated {loiter=10.0, gimbal=4.0/8.0,
    # rew_timeout=-250} point to collapse training in a near-identical way
    # (every gate crashing together, not just the targeted one) -- strong
    # evidence the current reward/curriculum balance is a narrow, fragile
    # optimum rather than a smoothly-improvable one. Stopped perturbing this
    # parameter; reverted to 10.0. Any future attempt at this specific
    # rush-to-fail theory should probably test a much smaller nudge (e.g.
    # 10.0 -> 8.5) rather than halving it, or address it via the crash/
    # timeout terminal values themselves instead of loiter's magnitude.
    rew_wb_loiter_penalty_per_s = 10.0
    # Fraction of THIS episode's own physically-implied minimum safe-descent
    # time (spawn_altitude_m / soft_vertical_speed_mps) exempted from the
    # loiter cost above -- see _get_rewards's loiter_grace_s. 0.8 leaves
    # some time pressure (not a full free ride) while removing the part of
    # the cost that was purely a tax on unavoidable descent time at higher
    # spawn altitudes. Distinct from the earlier, REJECTED flat
    # rew_wb_loiter_grace_s=9.0 (same grace for every altitude, including
    # ones that didn't need it) -- this one scales per-episode.
    rew_wb_loiter_altitude_grace_frac = 0.8
    # A rew_wb_loiter_grace_s (zero-cost grace period before this kicks in)
    # was tried and rejected -- see the loiter term's comment in
    # _get_rewards for the result. Removed rather than left unused.

    rew_soft_landing = 300.0
    # A harsh touchdown must stay negative even at maximum quality; otherwise
    # SAC learns to farm "almost good" crashes instead of crossing the soft gate.
    rew_harsh_landing = -220.0
    rew_landing_quality = 160.0
    rew_crash = -220.0
    # Must stay worse than even the WORST-case attempt outcome (bare
    # rew_crash/rew_harsh_landing, -220), not just the best-case harsh
    # landing (rew_harsh_landing + rew_landing_quality = -60). At -150 this
    # was worse than the best case but *better* than a bare crash -- and
    # since rew_wb_loiter_penalty_per_s accrues almost identically whether
    # an episode times out or attempts-then-crashes late, the terminal
    # delta (-150 vs -220) was the real comparison an uncertain policy
    # faced. That let it rationally hedge: attempt only from states it was
    # already confident about (hence gate_*_pass_rate among LANDED episodes
    # staying high, 86-90%), and hover to timeout otherwise -- observed as
    # timeout_rate *rising* over a run (0.52 -> 0.65) while failure_rate
    # among attempts fell (0.19 -> 0.09): the policy was getting better at
    # landing but attempting less. Moved below rew_crash so "never try" is
    # dominated in every case, not just the lucky ones.
    rew_timeout = -250.0
