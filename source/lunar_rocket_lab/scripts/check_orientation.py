"""Spawn the real env headless, print root orientation right after reset -- no policy, no physics steps, just: is the rocket's local +Z axis pointing toward world +Z (right-side up) or world -Z (upside down) at spawn time."""
import sys

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
import argparse

import lunar_rocket_lab.tasks  # noqa: F401 -- must be imported before resolve_task_config so gym knows the task

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="LunarRocket-Lander-Direct-v0")
parser.add_argument("--agent", type=str, default="sb3_sac_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--log_interval", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
add_launcher_args(parser)
if "--headless" not in parser._option_string_actions:
    parser.add_argument("--headless", action="store_true", default=False)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args
args_cli.headless = True

env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
env_cfg.scene.num_envs = 4

with launch_simulation(env_cfg, args_cli):
    import gymnasium as gym
    import torch

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    # ArticulationData.root_quat_w is (x, y, z, w) in Isaac Lab 3.0
    # (base_rigid_object_data.py: QUAT_XYZW_ELEMENT_NAMES), not (w, x, y, z).
    quat = env._rocket.data.root_quat_w  # (x, y, z, w)
    pos = env._rocket.data.root_pos_w

    def quat_rotate(q, v):
        x, y, z, w = q.unbind(-1)
        qvec = torch.stack([x, y, z], dim=-1)
        uv = torch.cross(qvec, v.expand_as(qvec), dim=-1)
        uuv = torch.cross(qvec, uv, dim=-1)
        return v.expand_as(qvec) + 2 * (w.unsqueeze(-1) * uv + uuv)

    local_up = torch.tensor([0.0, 0.0, 1.0], device=str(quat.device))
    world_up_component = quat_rotate(quat, local_up)

    for i in range(quat.shape[0]):
        q = quat[i].tolist()
        p = pos[i].tolist()
        up = world_up_component[i].tolist()
        print(f"env {i}: pos={p} quat(x,y,z,w)={q} local_+Z_in_world={up}")
        print(f"  -> {'RIGHT-SIDE UP (nose/legs-down-ish)' if up[2] > 0 else 'UPSIDE DOWN'} (world_z_component={up[2]:.4f})")

    env.close()
