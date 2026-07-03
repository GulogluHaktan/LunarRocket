from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_configs


def _require_training_dependencies() -> None:
    missing: list[str] = []
    if importlib.util.find_spec("gymnasium") is None:
        missing.append("gymnasium")
    if importlib.util.find_spec("stable_baselines3") is None:
        missing.append("stable-baselines3")
    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            "Missing RL training dependencies: "
            f"{packages}. Run `./scripts/train_sac_docker.sh`; the Docker entrypoint installs "
            "the RL dependencies into the persistent Isaac cache."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on the LunarRocket Isaac environment")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--model-out", default="outputs/models/sac_lunar_landing")
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"), help="Torch device for SB3 policy")
    parser.add_argument("--buffer-size", type=int, default=None, help="Override SAC replay buffer size")
    parser.add_argument(
        "--skip-close",
        action="store_true",
        help="Exit without SimulationApp.close(); useful in Docker when Isaac aborts during shutdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_training_dependencies()
    configs = load_configs(args.config_dir)
    rl_config = configs.get("rl", {})
    configs["sim"]["headless"] = bool(args.headless)
    configs["sim"]["render_on_step"] = not bool(args.headless)
    configs["sim"].setdefault("camera", {})["enabled"] = False

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": bool(args.headless),
            "renderer": configs["sim"].get("renderer", "RayTracedLighting"),
            "disable_viewport_updates": bool(args.headless),
        }
    )

    exit_code = 0
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.monitor import Monitor

        from app.env_lunar_landing import LunarLandingIsaacEnv
        from app.world_adapter import IsaacWorldAdapter
        from app.world_builder import LunarLandingWorld

        world = LunarLandingWorld(configs)
        world.build()
        adapter = IsaacWorldAdapter(
            world,
            {
                "control_dt": float(configs["sim"].get("physics_dt", 1.0 / 120.0)),
                "max_lidar_distance_m": float(rl_config.get("observation", {}).get("max_lidar_distance_m", 30.0)),
            },
        )
        env = LunarLandingIsaacEnv(
            adapter,
            {
                **rl_config,
                "max_episode_steps": int(rl_config.get("sac_plan", {}).get("max_episode_steps", 1000)),
                "control_dt": float(configs["sim"].get("physics_dt", 1.0 / 120.0)),
            },
        )
        env = Monitor(env)

        sac_plan = rl_config.get("sac_plan", {})
        buffer_size = int(args.buffer_size if args.buffer_size is not None else sac_plan.get("buffer_size", 500000))
        observation = rl_config.get("observation", {})
        observation_mode = observation.get("early_sac_mode", observation.get("mode", "state_lidar"))
        print(
            "[LunarRocket] RL observation "
            f"mode={observation_mode} keys={list(env.unwrapped.observation_space.spaces.keys())} "
            f"buffer_size={buffer_size} device={args.device}",
            flush=True,
        )
        print(
            "[LunarRocket] creating SAC model",
            flush=True,
        )
        model = SAC(
            "MultiInputPolicy",
            env,
            learning_rate=float(sac_plan.get("learning_rate", 3e-4)),
            buffer_size=buffer_size,
            batch_size=int(sac_plan.get("batch_size", 256)),
            gamma=float(sac_plan.get("gamma", 0.99)),
            tau=float(sac_plan.get("tau", 0.005)),
            train_freq=int(sac_plan.get("train_freq", 1)),
            gradient_steps=int(sac_plan.get("gradient_steps", 1)),
            ent_coef=sac_plan.get("ent_coef", "auto"),
            learning_starts=int(sac_plan.get("learning_starts", 10000)),
            target_update_interval=int(sac_plan.get("target_update_interval", 1)),
            device=args.device,
            verbose=1,
        )
        print("[LunarRocket] SAC model created", flush=True)
        print(
            "[LunarRocket] SAC training start "
            f"timesteps={int(args.timesteps)} gravity={configs['sim'].get('gravity_mps2')} "
            f"rocket_mass={configs['rocket'].get('mass_kg')} action=[thrust,gimbal_x,gimbal_y]",
            flush=True,
        )
        model.learn(total_timesteps=int(args.timesteps))
        output = Path(args.model_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output))
        print(f"[LunarRocket] SAC model saved: {output}", flush=True)
    except BaseException:
        exit_code = 1
        traceback.print_exc()
    finally:
        if args.skip_close:
            print("[LunarRocket] skipping SimulationApp.close for Docker shutdown stability", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        simulation_app.close()
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
