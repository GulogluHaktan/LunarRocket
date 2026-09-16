from __future__ import annotations

import argparse
import contextlib
import random
import sys
import time

import lunar_rocket_lab.tasks  # noqa: F401


def _raw_throttle_from_fraction(
    throttle: float,
    hover_throttle: float,
    max_commanded_throttle: float,
) -> float:
    throttle = max(0.0, min(1.0, float(throttle)))
    if throttle <= hover_throttle:
        return throttle / max(hover_throttle, 1e-6) - 1.0
    return (throttle - hover_throttle) / max(max_commanded_throttle - hover_throttle, 1e-6)


def main() -> None:
    import gymnasium as gym
    import torch

    from isaaclab.envs import DirectMARLEnvCfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

    parser = argparse.ArgumentParser(description="Preview the LunarRocket Isaac Lab scene before training.")
    parser.add_argument("--task", type=str, default="LunarRocket-Lander-Direct-v0")
    parser.add_argument("--agent", type=str, default="sb3_sac_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--spawn_altitude", type=float, default=2.5)
    parser.add_argument("--throttle", type=float, default=0.09)
    parser.add_argument("--controller", choices=("constant", "vertical_pd"), default="constant")
    parser.add_argument("--real-time", action="store_true", default=True)
    add_launcher_args(parser)
    if "--headless" not in parser._option_string_actions:
        parser.add_argument("--headless", action="store_true", default=False)
    args_cli, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + hydra_args

    env_cfg, _ = resolve_task_config(args_cli.task, args_cli.agent)
    with launch_simulation(env_cfg, args_cli):
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)

        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.spawn_altitude_m = float(args_cli.spawn_altitude)
        env_cfg.spawn_altitude_min_m = env_cfg.spawn_altitude_m
        env_cfg.spawn_xy_range_m = 0.0
        env_cfg.spawn_xy_min_range_m = 0.0
        env_cfg.target_xy_range_m = 0.0
        env_cfg.target_xy_min_range_m = 0.0
        env_cfg.curriculum_enabled = False
        env_cfg.legacy_terrain_usd_max_envs = max(env_cfg.legacy_terrain_usd_max_envs, args_cli.num_envs)

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        unwrapped = env.unwrapped
        env.reset()
        _set_camera_on_first_rocket(unwrapped)
        preview_action = torch.zeros((unwrapped.num_envs, 3), device=unwrapped.device)
        hover_throttle = (
            unwrapped.cfg.rocket_mass_kg
            * abs(float(unwrapped.cfg.sim.gravity[2]))
            / unwrapped.cfg.max_thrust_n
        )
        preview_action[:, 0] = _raw_throttle_from_fraction(
            args_cli.throttle,
            hover_throttle,
            unwrapped.cfg.max_commanded_throttle,
        )

        print(
            "[LunarRocket] preview scene is live: generated terrain + rocket are visible. "
            "Press Ctrl+C here to close.",
            flush=True,
        )

        dt = unwrapped.step_dt
        steps = max(1, int(float(args_cli.duration) / max(dt, 1e-6)))
        timing_start = time.perf_counter()
        with contextlib.suppress(KeyboardInterrupt):
            for step in range(steps):
                start_time = time.time()
                if args_cli.controller == "vertical_pd":
                    root_pos = unwrapped._rocket.data.root_pos_w
                    root_quat = unwrapped._rocket.data.root_quat_w
                    root_vz = unwrapped._rocket.data.root_lin_vel_w[:, 2]
                    altitude = torch.clamp(
                        torch.amin(unwrapped._foot_clearances(root_pos, root_quat), dim=-1),
                        min=0.0,
                    )
                    desired_vz = -torch.clamp(0.35 + 0.11 * altitude, max=3.0)
                    desired_net_acc_z = torch.clamp(desired_vz - root_vz, min=-3.0, max=3.0)
                    throttle = torch.clamp(
                        (
                            desired_net_acc_z
                            - float(unwrapped.cfg.sim.gravity[2])
                        )
                        * unwrapped.cfg.rocket_mass_kg
                        / unwrapped.cfg.max_thrust_n,
                        min=0.0,
                        max=unwrapped.cfg.max_commanded_throttle,
                    )
                    preview_action[:, 0] = torch.where(
                        throttle <= hover_throttle,
                        throttle / hover_throttle - 1.0,
                        (throttle - hover_throttle)
                        / (unwrapped.cfg.max_commanded_throttle - hover_throttle),
                    )
                _, _, terminated, truncated, _ = env.step(preview_action)
                if bool(terminated[0] or truncated[0]):
                    metrics = unwrapped._last_termination_metrics
                    print(
                        "[LunarRocket] touchdown/reset: "
                        f"soft={bool(metrics['soft'][0])}, "
                        f"harsh={bool(metrics['harsh'][0])}, "
                        f"timeout={bool(metrics['timeout_only'][0])}, "
                        f"distance_xy={float(metrics['distance_xy'][0]):.3f}, "
                        f"vxy={float(metrics['horizontal_speed'][0]):.3f}, "
                        f"vz={float(metrics['vertical_speed'][0]):.3f}, "
                        f"angular={float(metrics['angular_speed'][0]):.3f}, "
                        f"foot_spread={float(metrics['foot_clearance_spread'][0]):.3f}",
                        flush=True,
                    )
                if step % 30 == 0:
                    _set_camera_on_first_rocket(unwrapped)
                    elapsed = max(time.perf_counter() - timing_start, 1e-6)
                    pos = unwrapped._rocket.data.root_pos_w[0]
                    quat = unwrapped._rocket.data.root_quat_w[0]  # (w, x, y, z)
                    vel_z = float(unwrapped._rocket.data.root_lin_vel_w[0, 2])
                    # local +Z axis expressed in world frame: >0 means right-side up
                    # (legs/engine pointing toward world -Z, nose toward +Z), <0 upside down.
                    w, x, y, z = (float(v) for v in quat)
                    world_up_z = 1.0 - 2.0 * (x * x + y * y)
                    orientation = "RIGHT-SIDE-UP" if world_up_z > 0 else "UPSIDE-DOWN"
                    throttle_pct = 100.0 * float(preview_action[0, 0].clamp(-1.0, 1.0) * 0.5 + 0.5)
                    yaw_gimbal_deg = float(unwrapped._rocket.data.joint_pos[0, unwrapped._yaw_joint_ids[0]]) * 57.29578
                    pitch_gimbal_deg = float(unwrapped._rocket.data.joint_pos[0, unwrapped._pitch_joint_ids[0]]) * 57.29578
                    print(
                        f"[LunarRocket] step {step}/{steps} FPS={max(step, 1) / elapsed:.1f} "
                        f"pos=({float(pos[0]):.2f},{float(pos[1]):.2f},{float(pos[2]):.2f}) vz={vel_z:.3f} "
                        f"quat(w,x,y,z)=({w:.3f},{x:.3f},{y:.3f},{z:.3f}) [{orientation}, world_up_z={world_up_z:.3f}] "
                        f"raw_throttle_cmd={float(preview_action[0, 0]):.3f} (~{throttle_pct:.0f}% of commandable range) "
                        f"gimbal_yaw={yaw_gimbal_deg:.1f}deg gimbal_pitch={pitch_gimbal_deg:.1f}deg",
                        flush=True,
                    )
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

        env.close()


def _set_camera_on_first_rocket(env) -> None:
    with contextlib.suppress(Exception):
        pos = env._rocket.data.root_pos_w[0].detach().cpu()
        target = [float(pos[0]), float(pos[1]), max(0.8, float(pos[2]) * 0.55)]
        eye = [target[0] + 3.5, target[1] - 4.5, target[2] + 2.8]
        env.sim.set_camera_view(eye=eye, target=target)


if __name__ == "__main__":
    main()
