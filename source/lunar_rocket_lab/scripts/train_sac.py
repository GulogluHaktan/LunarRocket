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
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

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
    # Isaac Lab's launcher still consumes ``headless`` but recent develop
    # revisions no longer expose the legacy ``--headless`` CLI option.  Keep
    # the project entrypoint compatible with both the released and develop
    # launchers used by our Docker scripts.
    if "--headless" not in parser._option_string_actions:
        parser.add_argument("--headless", action="store_true", default=False)
    args_cli, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + hydra_args

    if args_cli.video:
        args_cli.enable_cameras = True

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    # A2 algorithm-comparison arms: sb3_td3_cfg.yaml / sb3_ddpg_cfg.yaml set
    # algo: "TD3" / "DDPG"; sb3_sac_cfg.yaml has no such key, so this defaults
    # to SAC and existing SAC-only runs are unaffected.
    algo_name = str(agent_cfg.pop("algo", "SAC")).upper()
    if algo_name not in {"SAC", "TD3", "DDPG"}:
        raise ValueError(f"Unsupported algo {algo_name!r} in agent config (expected SAC, TD3, or DDPG)")
    with launch_simulation(env_cfg, args_cli):
        import math

        import gymnasium as gym
        import numpy as np
        import torch
        from stable_baselines3 import DDPG, SAC, TD3
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, LogEveryNTimesteps
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.vec_env import VecNormalize
        from isaaclab.envs import DirectMARLEnvCfg, ManagerBasedRLEnvCfg
        from isaaclab.utils.dict import print_dict
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

        algo_classes = {"SAC": SAC, "TD3": TD3, "DDPG": DDPG}
        import isaaclab_tasks  # noqa: F401

        class EpisodeMetricsCallback(BaseCallback):
            """Forward Isaac Lab's completed-episode metrics to SB3/TensorBoard."""

            def _on_step(self) -> bool:
                for info in self.locals.get("infos", ()):
                    episode = info.get("episode")
                    if not episode:
                        continue
                    for key, value in episode.items():
                        if key in {"r", "l"} or not np.isscalar(value):
                            continue
                        self.logger.record_mean(key, float(value))
                return True

        class EntCoefFloorCallback(BaseCallback):
            """Keep SAC's auto-tuned entropy coefficient from collapsing to ~0.

            Observed repeatedly on this task: ent_coef decays from its ~0.85
            starting point down to ~0.0006-0.001 within a few million steps
            regardless of target_entropy (-3 default, -1.5 tuned). At
            difficulty 0.25 that near-zero value still gave 46.7% success, so
            the collapse itself is not fatal -- what breaks the policy is a
            *sudden large* jump in exploration noise (a first attempt with
            floor=0.05, ~70x the converged 0.0007, immediately drove
            mean_distance_xy from ~0.3m to ~12m and success to 0% with no
            recovery over 4M+ steps: the actor's precision was tuned for
            near-zero noise and a 70x shock destroyed it outright). This
            floor is set close enough to the natural collapse point (~7x, not
            ~70x) to stay just above literal zero without disrupting learned
            precision, and ramps in linearly over the first million steps of
            THIS run so a checkpoint that already converged to near-zero
            entropy is not hit with a step-function noise increase.
            """

            def __init__(self, floor: float = 0.005, ramp_steps: int = 1_000_000):
                super().__init__()
                self._log_floor = math.log(floor)
                self._ramp_steps = max(ramp_steps, 1)

            def _on_step(self) -> bool:
                log_ent_coef = getattr(self.model, "log_ent_coef", None)
                if log_ent_coef is not None:
                    ramp_frac = min(self.num_timesteps / self._ramp_steps, 1.0)
                    with torch.no_grad():
                        current_min = log_ent_coef.data.new_tensor(
                            -20.0 + ramp_frac * (self._log_floor - -20.0)
                        )
                        log_ent_coef.data.clamp_(min=current_min)
                return True

        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)

        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        # Keep SAC's update-to-transition ratio invariant. The YAML value of 32
        # is calibrated for 512 envs; real DEM collision training intentionally
        # uses 16 envs, which therefore needs one gradient step per vector step.
        base_gradient_steps = int(agent_cfg.get("gradient_steps", 32))
        agent_cfg["gradient_steps"] = max(
            1,
            round(base_gradient_steps * env_cfg.scene.num_envs / 512),
        )
        print(
            f"[INFO] SAC gradient_steps={agent_cfg['gradient_steps']} "
            f"for {env_cfg.scene.num_envs} envs",
            flush=True,
        )
        if os.environ.get("ISAACLAB_EPISODE_LENGTH_S"):
            env_cfg.episode_length_s = float(os.environ["ISAACLAB_EPISODE_LENGTH_S"])
        if os.environ.get("ISAACLAB_CURRICULUM_INITIAL_DIFFICULTY"):
            # The curriculum level lives on the env instance, not in the SB3
            # checkpoint, so resuming from a mid-curriculum checkpoint would
            # otherwise silently restart difficulty ramp-up from scratch.
            env_cfg.curriculum_initial_difficulty = float(
                os.environ["ISAACLAB_CURRICULUM_INITIAL_DIFFICULTY"]
            )
        agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
        if args_cli.max_iterations is not None:
            steps_per_iteration = int(agent_cfg.pop("steps_per_iteration", 32))
            agent_cfg["n_timesteps"] = args_cli.max_iterations * steps_per_iteration * env_cfg.scene.num_envs
        else:
            agent_cfg.pop("steps_per_iteration", None)

        env_cfg.seed = agent_cfg["seed"]
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Keep the historical "sb3_sac" directory for SAC (existing tooling --
        # train_terrain_blend_comparison.sh, TensorBoard bookmarks -- points
        # there); TD3/DDPG comparison runs (A2) get their own algo-named
        # subdirectory so they never mix with or overwrite SAC logs.
        log_subdir = "sb3_sac" if algo_name == "SAC" else f"sb3_{algo_name.lower()}"
        log_root_path = os.path.abspath(os.path.join("logs", log_subdir, args_cli.task))
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        print(f"Exact experiment name requested from command line: {run_info}")
        log_dir = os.path.join(log_root_path, run_info)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        (Path(log_dir) / "command.txt").write_text(" ".join(sys.orig_argv))

        agent_cfg = process_sb3_cfg(agent_cfg, env_cfg.scene.num_envs)
        policy_arch = agent_cfg.pop("policy")
        n_timesteps = int(agent_cfg.pop("n_timesteps"))
        # TD3/DDPG have no entropy-driven exploration like SAC, so their
        # configs carry an explicit action_noise_std (see sb3_td3_cfg.yaml /
        # sb3_ddpg_cfg.yaml); SAC's config has no such key.
        action_noise_std = agent_cfg.pop("action_noise_std", None)

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
            vecnormalize_path = None
            if args_cli.checkpoint is not None:
                checkpoint_path = Path(args_cli.checkpoint)
                suffix = checkpoint_path.stem.removeprefix("model")
                candidate = checkpoint_path.with_name(f"model_vecnormalize{suffix}.pkl")
                if candidate.exists():
                    vecnormalize_path = candidate
            if vecnormalize_path is not None:
                print(f"[INFO] Loading VecNormalize statistics: {vecnormalize_path}", flush=True)
                env = VecNormalize.load(str(vecnormalize_path), env)
                env.training = True
                env.norm_reward = norm_args.get("normalize_value", False)
                env.clip_obs = norm_args.get("clip_obs", 100.0)
            else:
                env = VecNormalize(
                    env,
                    training=True,
                    norm_obs=norm_args["normalize_input"],
                    norm_reward=norm_args.get("normalize_value", False),
                    clip_obs=norm_args.get("clip_obs", 100.0),
                    gamma=agent_cfg["gamma"],
                    clip_reward=np.inf,
                )

        algo_kwargs = dict(agent_cfg)
        if action_noise_std is not None:
            n_actions = int(env.action_space.shape[-1])
            algo_kwargs["action_noise"] = NormalActionNoise(
                mean=np.zeros(n_actions), sigma=float(action_noise_std) * np.ones(n_actions)
            )
        algo_cls = algo_classes[algo_name]
        agent = algo_cls(policy_arch, env, verbose=1, tensorboard_log=log_dir, **algo_kwargs)
        if args_cli.checkpoint is not None:
            # SAC.load() restores tensorboard_log from the checkpoint's pickled
            # hyperparameters (the *old* run's log_dir) unless overridden here,
            # which silently redirects this run's tensorboard events into the
            # previous run's directory instead of this one's.
            agent = agent.load(args_cli.checkpoint, env, tensorboard_log=log_dir, print_system_info=True)

        callbacks = [
            EpisodeMetricsCallback(),
            EntCoefFloorCallback(),
            CheckpointCallback(
                save_freq=max(1, 500_000 // env_cfg.scene.num_envs),
                save_path=log_dir,
                name_prefix="model",
                save_vecnormalize=True,
                verbose=2,
            ),
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
