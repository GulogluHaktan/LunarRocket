from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
import time
from pathlib import Path

import lunar_rocket_lab.tasks  # noqa: F401


def main() -> None:
    import gymnasium as gym
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecNormalize

    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.envs import DirectMARLEnvCfg

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

    parser = argparse.ArgumentParser(description="Play a LunarRocket Stable-Baselines3 SAC checkpoint.")
    parser.add_argument("--video", action="store_true", default=False, help="Record a play video.")
    parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video in steps.")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default="LunarRocket-Lander-Direct-v0")
    parser.add_argument("--agent", type=str, default="sb3_sac_cfg_entry_point")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a SAC model checkpoint.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--real-time", action="store_true", default=True, help="Run in real-time when possible.")
    parser.add_argument("--keep_all_info", action="store_true", default=True)
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + hydra_args

    if args_cli.video:
        args_cli.enable_cameras = True

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    with launch_simulation(env_cfg, args_cli):
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)

        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
        env_cfg.seed = agent_cfg["seed"]
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.num_envs is not None and args_cli.num_envs <= 4:
            env_cfg.spawn_altitude_m = float(os.environ.get("ISAACLAB_WATCH_SPAWN_ALTITUDE", "8.0"))

        task_name = args_cli.task.split(":")[-1]
        log_root_path = os.path.abspath(os.path.join("logs", "sb3_sac", task_name))
        if args_cli.checkpoint is None:
            checkpoint_path = get_checkpoint_path(
                log_root_path,
                ".*",
                r"model_.*\.zip",
                sort_alpha=False,
                preferred_checkpoint=r"model\.zip",
            )
        else:
            checkpoint_path = args_cli.checkpoint
        log_dir = os.path.dirname(checkpoint_path)
        env_cfg.log_dir = log_dir

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        with contextlib.suppress(Exception):
            env.unwrapped.sim.set_camera_view(eye=[14.0, -16.0, 10.0], target=[0.0, 0.0, 3.0])
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        if args_cli.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=os.path.join(log_dir, "videos", "play"),
                step_trigger=lambda step: step == 0,
                video_length=args_cli.video_length,
                disable_logger=True,
            )

        agent_cfg = process_sb3_cfg(agent_cfg, env.unwrapped.num_envs)
        env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

        vec_norm_path = Path(checkpoint_path.replace("/model", "/model_vecnormalize").replace(".zip", ".pkl"))
        if vec_norm_path.exists():
            print(f"Loading saved normalization: {vec_norm_path}")
            env = VecNormalize.load(vec_norm_path, env)
            env.training = False
            env.norm_reward = False
        elif agent_cfg.pop("normalize_input", False):
            env = VecNormalize(
                env,
                training=False,
                norm_obs=True,
                norm_reward=False,
                clip_obs=agent_cfg.pop("clip_obs", 100.0),
            )

        print(f"Loading SAC checkpoint from: {checkpoint_path}")
        agent = SAC.load(checkpoint_path, env, print_system_info=True)
        dt = env.unwrapped.step_dt
        obs = env.reset()
        with contextlib.suppress(Exception):
            env.unwrapped.sim.set_camera_view(eye=[14.0, -16.0, 10.0], target=[0.0, 0.0, 3.0])
        timestep = 0
        print("[INFO] Play loop started. Press Ctrl+C in this terminal to stop.")

        with contextlib.suppress(KeyboardInterrupt):
            while True:
                start_time = time.time()
                with torch.inference_mode():
                    actions, _ = agent.predict(obs, deterministic=True)
                    obs, _, _, _ = env.step(actions)

                timestep += 1
                if timestep % 300 == 0:
                    print(f"[INFO] Play step: {timestep}", flush=True)

                if args_cli.video:
                    if timestep >= args_cli.video_length:
                        break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

        env.close()


if __name__ == "__main__":
    main()
