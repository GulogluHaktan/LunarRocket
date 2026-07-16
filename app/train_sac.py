from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
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
    parser.add_argument("--progress-freq", type=int, default=1000, help="Print training progress every N env steps")
    parser.add_argument("--checkpoint-freq", type=int, default=10000, help="Save a checkpoint every N env steps")
    parser.add_argument("--checkpoint-dir", default="outputs/models/checkpoints")
    parser.add_argument(
        "--fast-terrain",
        action="store_true",
        help="Use a lightweight training terrain: low-res mesh, no dense landing patch, no rocks.",
    )
    parser.add_argument(
        "--visual-detail-fast-physics",
        action="store_true",
        help="Keep dense landing detail and rocks visible/observable, but use coarse collision for speed.",
    )
    parser.add_argument("--terrain-quality", default=None, help="Override terrain quality preset")
    parser.add_argument("--local-detail-resolution", type=int, default=None, help="Override local detail resolution")
    parser.add_argument("--physics-dt", type=float, default=None, help="Override Isaac physics/control timestep")
    parser.add_argument(
        "--analytic-env",
        action="store_true",
        help="Train against a fast analytic dynamics/terrain adapter instead of launching Isaac.",
    )
    parser.add_argument(
        "--analytic-control-dt",
        type=float,
        default=0.05,
        help="Control timestep for --analytic-env.",
    )
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
    _apply_training_overrides(configs, args)
    rl_config = configs.get("rl", {})
    configs["sim"]["headless"] = bool(args.headless)
    configs["sim"]["render_on_step"] = not bool(args.headless)
    configs["sim"].setdefault("camera", {})["enabled"] = False

    simulation_app = None
    if not args.analytic_env:
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
    else:
        configs["sim"]["physics_dt"] = float(args.analytic_control_dt)
        configs["sim"]["rendering_dt"] = float(args.analytic_control_dt)
        print(
            "[LunarRocket] analytic training env enabled "
            f"control_dt={float(args.analytic_control_dt):.4f} "
            "isaac_stage=false high_detail_terrain=analytic",
            flush=True,
        )

    exit_code = 0
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor

        from app.env_lunar_landing import LunarLandingIsaacEnv

        class TrainingProgressCallback(BaseCallback):
            def __init__(self, total_timesteps: int, progress_freq: int):
                super().__init__(verbose=0)
                self.total_timesteps = max(1, int(total_timesteps))
                self.progress_freq = max(1, int(progress_freq))
                self.started_at = time.monotonic()
                self.last_report_step = 0

            def _on_step(self) -> bool:
                if self.num_timesteps - self.last_report_step < self.progress_freq:
                    return True
                elapsed = max(time.monotonic() - self.started_at, 1e-6)
                steps_per_s = self.num_timesteps / elapsed
                percent = 100.0 * min(self.num_timesteps, self.total_timesteps) / self.total_timesteps
                remaining = max(self.total_timesteps - self.num_timesteps, 0)
                eta_s = remaining / max(steps_per_s, 1e-6)
                print(
                    "[LunarRocket] training progress "
                    f"steps={self.num_timesteps}/{self.total_timesteps} "
                    f"({percent:.2f}%) speed={steps_per_s:.2f} steps/s eta={_format_duration(eta_s)}",
                    flush=True,
                )
                self.last_report_step = self.num_timesteps
                return True

        adapter_config = {
            "control_dt": float(configs["sim"].get("physics_dt", 1.0 / 120.0)),
            "max_lidar_distance_m": float(rl_config.get("observation", {}).get("max_lidar_distance_m", 30.0)),
        }
        if args.analytic_env:
            from app.analytic_world_adapter import AnalyticLunarLandingAdapter

            adapter = AnalyticLunarLandingAdapter(configs, adapter_config)
        else:
            from app.world_adapter import IsaacWorldAdapter
            from app.world_builder import LunarLandingWorld

            world = LunarLandingWorld(configs)
            world.build()
            adapter = IsaacWorldAdapter(world, adapter_config)
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
        callbacks = [
            TrainingProgressCallback(
                total_timesteps=int(args.timesteps),
                progress_freq=max(1, int(args.progress_freq)),
            )
        ]
        if int(args.checkpoint_freq) > 0:
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                CheckpointCallback(
                    save_freq=int(args.checkpoint_freq),
                    save_path=str(checkpoint_dir),
                    name_prefix="sac_lunar_landing",
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                )
            )
            print(
                "[LunarRocket] checkpointing enabled "
                f"dir={checkpoint_dir} freq={int(args.checkpoint_freq)}",
                flush=True,
            )
        model.learn(total_timesteps=int(args.timesteps), callback=CallbackList(callbacks))
        output = Path(args.model_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(output))
        print(f"[LunarRocket] SAC model saved: {output}", flush=True)
    except BaseException:
        exit_code = 1
        traceback.print_exc()
    finally:
        if args.skip_close and simulation_app is not None:
            print("[LunarRocket] skipping SimulationApp.close for Docker shutdown stability", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)
        if simulation_app is not None:
            simulation_app.close()
    if exit_code:
        raise SystemExit(exit_code)


def _format_duration(seconds: float) -> str:
    total = int(max(seconds, 0.0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _apply_training_overrides(configs: dict[str, dict], args: argparse.Namespace) -> None:
    terrain = configs.get("terrain", {})
    sim = configs.get("sim", {})

    if args.fast_terrain:
        terrain["terrain_quality"] = "low_debug"
        terrain.setdefault("local_detail_patch", {})["enabled"] = False
        terrain.setdefault("rocks", {})["enabled"] = False
        terrain["smooth_visual_normals"] = False
        terrain["crater_count_range"] = [2, 6]
        terrain["low_frequency_noise_smoothing_passes"] = min(
            int(terrain.get("low_frequency_noise_smoothing_passes", 10)), 2
        )
        print(
            "[LunarRocket] fast terrain enabled "
            "quality=low_debug local_detail=false rocks=false",
            flush=True,
        )

    if args.visual_detail_fast_physics:
        terrain["terrain_quality"] = "medium_train"
        local_detail = terrain.setdefault("local_detail_patch", {})
        local_detail["enabled"] = True
        local_detail["resolution"] = int(local_detail.get("resolution", 1024))
        local_detail["collision_enabled"] = False
        rocks = terrain.setdefault("rocks", {})
        rocks["enabled"] = True
        rocks["collision_enabled"] = False
        print(
            "[LunarRocket] visual detail fast physics enabled "
            f"quality={terrain['terrain_quality']} local_detail_resolution={local_detail['resolution']} "
            "local_collision=false rocks=true rock_collision=false",
            flush=True,
        )

    if args.terrain_quality:
        terrain["terrain_quality"] = str(args.terrain_quality)
        print(f"[LunarRocket] terrain quality override: {terrain['terrain_quality']}", flush=True)

    if args.local_detail_resolution is not None:
        local_detail = terrain.setdefault("local_detail_patch", {})
        local_detail["enabled"] = int(args.local_detail_resolution) > 0
        local_detail["resolution"] = int(args.local_detail_resolution)
        print(
            "[LunarRocket] local detail override "
            f"enabled={local_detail['enabled']} resolution={local_detail['resolution']}",
            flush=True,
        )

    if args.physics_dt is not None:
        sim["physics_dt"] = float(args.physics_dt)
        sim["rendering_dt"] = max(float(sim.get("rendering_dt", args.physics_dt)), float(args.physics_dt))
        print(f"[LunarRocket] physics dt override: {sim['physics_dt']}", flush=True)


if __name__ == "__main__":
    main()
