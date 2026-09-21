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
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import VecNormalize

    from isaaclab.envs import DirectMARLEnvCfg

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import (
        add_launcher_args,
        get_checkpoint_path,
        launch_simulation,
        resolve_task_config,
        setup_preset_cli,
    )

    parser = argparse.ArgumentParser(description="Play a LunarRocket Stable-Baselines3 SAC checkpoint.")
    parser.add_argument("--video", action="store_true", default=False, help="Record a play video.")
    parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video in steps.")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default="LunarRocket-Lander-Direct-v0")
    parser.add_argument("--agent", type=str, default="sb3_sac_cfg_entry_point")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a SAC model checkpoint.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--evaluation_steps", type=int, default=0, help="Stop after this many steps; zero runs forever.")
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time when possible.")
    parser.add_argument("--keep_all_info", action="store_true", default=True)
    add_launcher_args(parser)
    if "--headless" not in parser._option_string_actions:
        parser.add_argument("--headless", action="store_true", default=False)
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
        env_cfg.curriculum_enabled = False
        if os.environ.get("ISAACLAB_WATCH_POLICY_RCS_THRUST_N"):
            env_cfg.policy_max_rcs_thrust_n = float(
                os.environ["ISAACLAB_WATCH_POLICY_RCS_THRUST_N"]
            )
        if args_cli.num_envs is not None and args_cli.num_envs <= 4:
            env_cfg.spawn_altitude_m = float(os.environ.get("ISAACLAB_WATCH_SPAWN_ALTITUDE", "8.0"))
            env_cfg.spawn_altitude_min_m = env_cfg.spawn_altitude_m
            env_cfg.spawn_xy_range_m = float(os.environ.get("ISAACLAB_WATCH_SPAWN_XY_RANGE", "0.5"))
            env_cfg.spawn_xy_min_range_m = env_cfg.spawn_xy_range_m
            env_cfg.target_xy_range_m = float(os.environ.get("ISAACLAB_WATCH_TARGET_XY_RANGE", "2.0"))
            env_cfg.target_xy_min_range_m = env_cfg.target_xy_range_m
        env_cfg.viewer.eye = (6.0, -7.0, 5.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 1.5)
        if args_cli.video and env_cfg.video_recorder is not None:
            capture_stride = max(1, int(os.environ.get("ISAACLAB_VIDEO_CAPTURE_STRIDE", "8")))
            os.environ["ISAACLAB_ENABLE_VIDEO_CAMERA"] = "1"
            env_cfg.sim.render_interval = env_cfg.decimation * capture_stride
            env_cfg.video_recorder.window_width = int(os.environ.get("ISAACLAB_VIDEO_WIDTH", "854"))
            env_cfg.video_recorder.window_height = int(os.environ.get("ISAACLAB_VIDEO_HEIGHT", "480"))
            print(f"[INFO] Isaac camera capture stride: {capture_stride}", flush=True)

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
        base_env = env.unwrapped
        if args_cli.video and getattr(base_env, "_video_camera", None) is not None:
            # Prime the RTX sensor after its final look-at pose is set, before
            # Gym's recorder asks for its first frame.
            for _ in range(4):
                base_env.sim.render()
                base_env.scene.update(dt=base_env.physics_dt)
        with contextlib.suppress(Exception):
            _eye = [float(v) for v in (os.environ.get("ISAACLAB_WATCH_CAMERA_EYE") or "6.0,-7.0,5.0").split(",")]
            _target = [float(v) for v in (os.environ.get("ISAACLAB_WATCH_CAMERA_TARGET") or "0.0,0.0,1.5").split(",")]
            env.unwrapped.sim.set_camera_view(eye=_eye, target=_target)
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
            _eye = [float(v) for v in (os.environ.get("ISAACLAB_WATCH_CAMERA_EYE") or "6.0,-7.0,5.0").split(",")]
            _target = [float(v) for v in (os.environ.get("ISAACLAB_WATCH_CAMERA_TARGET") or "0.0,0.0,1.5").split(",")]
            env.unwrapped.sim.set_camera_view(eye=_eye, target=_target)
        timestep = 0
        episode_count = 0
        success_count = 0
        harsh_count = 0
        failure_count = 0
        timeout_count = 0
        completed_returns: list[float] = []
        completed_lengths: list[float] = []
        print("[INFO] Play loop started. Press Ctrl+C in this terminal to stop.")
        stochastic_policy = os.environ.get(
            "ISAACLAB_STOCHASTIC_POLICY",
            "0",
        ).lower() in {"1", "true", "yes"}
        print(
            f"[INFO] Policy sampling: {'stochastic' if stochastic_policy else 'deterministic'}",
            flush=True,
        )

        with contextlib.suppress(KeyboardInterrupt):
            while True:
                start_time = time.time()
                with torch.inference_mode():
                    actions, _ = agent.predict(obs, deterministic=not stochastic_policy)
                    obs, _, dones, infos = env.step(actions)

                done_ids = np.flatnonzero(dones)
                if done_ids.size:
                    metrics = base_env._last_termination_metrics
                    done_tensor = torch.as_tensor(done_ids, dtype=torch.long, device=base_env.device)
                    episode_count += int(done_ids.size)
                    success_count += int(metrics["soft"][done_tensor].sum().item())
                    harsh_count += int(metrics["harsh"][done_tensor].sum().item())
                    failure_count += int(metrics["failed"][done_tensor].sum().item())
                    timeout_count += int(metrics["timeout_only"][done_tensor].sum().item())
                    for index in done_ids:
                        episode = infos[index].get("episode")
                        if episode:
                            completed_returns.append(float(episode["r"]))
                            completed_lengths.append(float(episode["l"]))

                timestep += 1
                if timestep % 300 == 0:
                    print(f"[INFO] Play step: {timestep}", flush=True)

                if os.environ.get("ISAACLAB_DEBUG_ORIENTATION"):
                    quat = base_env._rocket.data.root_quat_w[0:1]
                    world_up = torch.zeros((1, 3), device=quat.device)
                    world_up[:, 2] = 1.0
                    q_vec = quat[:, 1:4]
                    q_w = quat[:, 0:1]
                    t = 2.0 * torch.cross(q_vec, world_up, dim=-1)
                    body_up_w = world_up + q_w * t + torch.cross(q_vec, t, dim=-1)
                    up_z = float(body_up_w[0, 2].item())
                    pos_z = float(base_env._rocket.data.root_pos_w[0, 2].item())
                    print(f"[ORIENT] step={timestep} up_z={up_z:+.3f} (>0 upright,<0 upside-down) pos_z={pos_z:.3f}", flush=True)

                if args_cli.video:
                    if timestep >= args_cli.video_length:
                        break
                elif args_cli.evaluation_steps > 0 and timestep >= args_cli.evaluation_steps:
                    break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

        if episode_count:
            mean_return = float(np.mean(completed_returns)) if completed_returns else float("nan")
            mean_length = float(np.mean(completed_lengths)) if completed_lengths else float("nan")
            print(
                "[RESULT] "
                f"episodes={episode_count}, success_rate={success_count / episode_count:.3f}, "
                f"harsh_rate={harsh_count / episode_count:.3f}, "
                f"failure_rate={failure_count / episode_count:.3f}, "
                f"timeout_rate={timeout_count / episode_count:.3f}, "
                f"mean_return={mean_return:.2f}, mean_length={mean_length:.1f}",
                flush=True,
            )
        else:
            print("[RESULT] no episode completed inside the requested play horizon", flush=True)
        env.close()


if __name__ == "__main__":
    main()
