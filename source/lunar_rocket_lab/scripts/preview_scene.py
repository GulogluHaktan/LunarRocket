from __future__ import annotations

import argparse
import contextlib
import random
import sys
import time

import lunar_rocket_lab.tasks  # noqa: F401


def _raw_throttle_from_fraction(throttle: float) -> float:
    return 2.0 * max(0.0, min(1.0, float(throttle))) - 1.0


def main() -> None:
    import gymnasium as gym
    import torch

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.envs import DirectMARLEnvCfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

    parser = argparse.ArgumentParser(description="Preview the LunarRocket Isaac Lab scene before training.")
    parser.add_argument("--task", type=str, default="LunarRocket-Lander-Direct-v0")
    parser.add_argument("--agent", type=str, default="sb3_sac_cfg_entry_point")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--spawn_altitude", type=float, default=2.5)
    parser.add_argument("--throttle", type=float, default=0.09)
    parser.add_argument("--real-time", action="store_true", default=True)
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + hydra_args

    env_cfg, _ = resolve_task_config(args_cli.task, args_cli.agent)
    with launch_simulation(env_cfg, args_cli):
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)

        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.scene.clone_in_fabric = False
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.sim.use_fabric = False
        env_cfg.spawn_altitude_m = float(args_cli.spawn_altitude)
        env_cfg.spawn_xy_range_m = 0.0
        env_cfg.legacy_terrain_usd_max_envs = max(env_cfg.legacy_terrain_usd_max_envs, args_cli.num_envs)

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        unwrapped = env.unwrapped
        env.reset()
        _set_camera_on_first_rocket(unwrapped)
        preview_action = torch.zeros((unwrapped.num_envs, 3), device=unwrapped.device)
        preview_action[:, 0] = _raw_throttle_from_fraction(args_cli.throttle)

        print(
            "[LunarRocket] preview scene is live: generated terrain + rocket are visible. "
            "Press Ctrl+C here to close.",
            flush=True,
        )

        dt = unwrapped.step_dt
        steps = max(1, int(float(args_cli.duration) / max(dt, 1e-6)))
        with contextlib.suppress(KeyboardInterrupt):
            for step in range(steps):
                start_time = time.time()
                env.step(preview_action)
                if step % 300 == 0:
                    _set_camera_on_first_rocket(unwrapped)
                    print(f"[LunarRocket] preview step {step}/{steps}", flush=True)
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

        env.close()


def _set_camera_on_first_rocket(env) -> None:
    with contextlib.suppress(Exception):
        pos = env._rocket.data.root_pos_w[0].detach().cpu()
        target = [float(pos[0]), float(pos[1]), max(0.8, float(pos[2]) * 0.55)]
        eye = [target[0] + 7.0, target[1] - 9.0, target[2] + 5.0]
        env.sim.set_camera_view(eye=eye, target=target)


if __name__ == "__main__":
    main()
