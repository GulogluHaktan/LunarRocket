# KNOWN ISSUE (post TVC->RCS/RigidObject conversion): as of this commit,
# running this script against LunarLanderEnv crashes inside Isaac Lab's own
# env init -- `sim.reset()` -> RigidObject._initialize_impl -> WrenchComposer
# init -> `data.body_com_pos_w` -> `get_transforms()` raises "Failed to get
# rigid body transforms from backend". This happens before any of this
# script's own code runs, with `--enable_cameras`/render_mode="rgb_array"
# active; the identical env construction path (same RigidObject, same
# rcs_thruster_layout wrenches) runs fine without cameras -- both the
# ISAACLAB_MAX_ITERATIONS=1 training smoke test and
# scripts/watch_isaaclab_docker.sh's interactive preview_scene.py path
# (6800+ real steps, correct free-fall/reset physics, zero spurious torque)
# complete cleanly. Root cause looks like an Isaac Lab-internal physics-view
# timing interaction between RigidObject and camera rendering, not a bug in
# this project's env/asset code -- not chased further here. If you hit this,
# try preview_scene.py (via watch_isaaclab_docker.sh) or play_sac.py --video
# instead until upstream Isaac Lab addresses it.
from __future__ import annotations

import argparse
import contextlib
import os
import sys

import lunar_rocket_lab.tasks  # noqa: F401


def main() -> None:
    import gymnasium as gym
    import numpy as np
    import torch

    from isaaclab.envs import DirectMARLEnvCfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

    parser = argparse.ArgumentParser(description="Capture a single still frame of the LunarRocket scene.")
    parser.add_argument("--task", type=str, default="LunarRocket-Lander-Direct-v0")
    parser.add_argument("--agent", type=str, default="sb3_sac_cfg_entry_point")
    parser.add_argument("--spawn_altitude", type=float, default=3.0)
    parser.add_argument("--settle_steps", type=int, default=90)
    parser.add_argument("--out", type=str, default="logs/screenshots/rocket_top.png")
    add_launcher_args(parser)
    if "--headless" not in parser._option_string_actions:
        parser.add_argument("--headless", action="store_true", default=True)
    args_cli, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + hydra_args

    args_cli.enable_cameras = True
    os.environ["ISAACLAB_ENABLE_VIDEO_CAMERA"] = "1"

    env_cfg, _ = resolve_task_config(args_cli.task, args_cli.agent)
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = 1
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.curriculum_enabled = False
        env_cfg.spawn_altitude_m = float(args_cli.spawn_altitude)
        env_cfg.spawn_altitude_min_m = env_cfg.spawn_altitude_m
        env_cfg.spawn_xy_range_m = 0.0
        env_cfg.spawn_xy_min_range_m = 0.0
        env_cfg.target_xy_range_m = 0.0
        env_cfg.target_xy_min_range_m = 0.0

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        unwrapped = env.unwrapped
        env.reset()

        with contextlib.suppress(Exception):
            _eye = [float(v) for v in (os.environ.get("ISAACLAB_WATCH_CAMERA_EYE") or "6.0,-7.0,5.0").split(",")]
            _target = [float(v) for v in (os.environ.get("ISAACLAB_WATCH_CAMERA_TARGET") or "0.0,0.0,1.5").split(",")]
            unwrapped.sim.set_camera_view(eye=_eye, target=_target)

        # Prime the RTX camera after its pose is set, same as play_sac.py's
        # --video path -- the first render() after set_world_poses_from_view
        # can return a stale/blank frame otherwise.
        for _ in range(4):
            unwrapped.sim.render()
            unwrapped.scene.update(dt=unwrapped.physics_dt)

        # raw_throttle==0 already maps to hover thrust in _pre_physics_step
        # (raw_throttle<=0 branch: hover_throttle*(raw_throttle+1)). RCS
        # duties (indices 1:) default to 0 -> clamped to off, matching the
        # action-space width [throttle, rcs_0..rcs_7] set in env_cfg.py.
        action = torch.zeros(
            (unwrapped.num_envs, 1 + len(unwrapped.cfg.rcs_thruster_layout)), device=unwrapped.device
        )

        best_frame = None
        best_contrast = -1.0
        for _ in range(max(1, int(args_cli.settle_steps))):
            env.step(action)
            frame = unwrapped.render()
            if frame is not None:
                contrast = float(frame.max()) - float(frame.min())
                if contrast > best_contrast:
                    best_contrast = contrast
                    best_frame = frame
        # The RTX camera needs several converged samples before a frame is
        # anything but a flat placeholder -- keep polling render() with no
        # further physics stepping until one shows real contrast, same as
        # lunar_lander_env.py's own render() cache is meant to settle into.
        for _ in range(60):
            frame = unwrapped.render()
            if frame is None:
                continue
            contrast = float(frame.max()) - float(frame.min())
            if contrast > best_contrast:
                best_contrast = contrast
                best_frame = frame
            if contrast > 8.0:
                break
        frame = best_frame
        if frame is None:
            raise RuntimeError("render() returned no frame -- is ISAACLAB_ENABLE_VIDEO_CAMERA/--enable_cameras active?")
        print(f"[LunarRocket] best frame contrast: {best_contrast:.1f}", flush=True)

        from PIL import Image

        out_path = args_cli.out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        Image.fromarray(np.asarray(frame)).save(out_path)
        print(f"[LunarRocket] screenshot saved: {out_path}", flush=True)

        env.close()


if __name__ == "__main__":
    main()
