from __future__ import annotations

import argparse
import contextlib
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import lunar_rocket_lab.tasks  # noqa: F401


def _cleanup_pbar(*args):
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


def main() -> None:
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

    signal.signal(signal.SIGINT, _cleanup_pbar)

    parser = argparse.ArgumentParser(description="Train LunarRocket with Stable-Baselines3 SAC.")
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video_length", type=int, default=200)
    parser.add_argument("--video_interval", type=int, default=2000)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--agent", type=str, default="sb3_sac_cfg_entry_point")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=100_000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--keep_all_info", action="store_true", default=True)
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + hydra_args

    if args_cli.video:
        args_cli.enable_cameras = True

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    with launch_simulation(env_cfg, args_cli):
        import gymnasium as gym
        import numpy as np
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
        from stable_baselines3.common.vec_env import VecNormalize
        from isaaclab.envs import DirectMARLEnvCfg, ManagerBasedRLEnvCfg
        from isaaclab.utils.dict import print_dict
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
        import isaaclab_tasks  # noqa: F401

        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)

        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
        if args_cli.max_iterations is not None:
            steps_per_iteration = int(agent_cfg.pop("steps_per_iteration", 32))
            agent_cfg["n_timesteps"] = args_cli.max_iterations * steps_per_iteration * env_cfg.scene.num_envs
        else:
            agent_cfg.pop("steps_per_iteration", None)

        env_cfg.seed = agent_cfg["seed"]
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_root_path = os.path.abspath(os.path.join("logs", "sb3_sac", args_cli.task))
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        print(f"Exact experiment name requested from command line: {run_info}")
        log_dir = os.path.join(log_root_path, run_info)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        (Path(log_dir) / "command.txt").write_text(" ".join(sys.orig_argv))

        agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
        policy_arch = agent_cfg.pop("policy")
        n_timesteps = int(agent_cfg.pop("n_timesteps"))

        if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
            print("[WARN] IO descriptors are only supported for manager based RL environments.")
        env_cfg.log_dir = log_dir

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        start_time = time.time()
        env = Sb3VecEnvWrapper(env, fast_variant=not args_cli.keep_all_info)

        norm_args = {key: agent_cfg.pop(key) for key in ("normalize_input", "normalize_value", "clip_obs") if key in agent_cfg}
        if norm_args and norm_args.get("normalize_input"):
            env = VecNormalize(
                env,
                training=True,
                norm_obs=norm_args["normalize_input"],
                norm_reward=norm_args.get("normalize_value", False),
                clip_obs=norm_args.get("clip_obs", 100.0),
                gamma=agent_cfg["gamma"],
                clip_reward=np.inf,
            )

        agent = SAC(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)
        if args_cli.checkpoint is not None:
            agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

        callbacks = [
            CheckpointCallback(save_freq=1000, save_path=log_dir, name_prefix="model", verbose=2),
            LogEveryNTimesteps(n_steps=args_cli.log_interval),
        ]
        with contextlib.suppress(KeyboardInterrupt):
            agent.learn(total_timesteps=n_timesteps, callback=callbacks, progress_bar=True, log_interval=None)

        agent.save(os.path.join(log_dir, "model"))
        print("Saving to:")
        print(os.path.join(log_dir, "model.zip"))
        if isinstance(env, VecNormalize):
            env.save(os.path.join(log_dir, "model_vecnormalize.pkl"))
        print(f"Training time: {round(time.time() - start_time, 2)} seconds")
        env.close()


if __name__ == "__main__":
    main()
