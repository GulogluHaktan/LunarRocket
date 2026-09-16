"""Lunar lander direct workflow task."""

import gymnasium as gym

from . import agents

gym.register(
    id="LunarRocket-Lander-Direct-v0",
    entry_point=f"{__name__}.lunar_lander_env:LunarLanderEnv",
    disable_env_checker=True,
    kwargs={
        # Must resolve without importing lunar_lander_env (the runtime env
        # class, which needs SimulationApp/pxr) -- see lunar_lander_env_cfg.py.
        "env_cfg_entry_point": f"{__name__}.lunar_lander_env_cfg:LunarLanderEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
        # A2 algorithm-comparison arms -- same env, tuned SAC hyperparameters
        # ported where applicable (see agents/sb3_td3_cfg.yaml, sb3_ddpg_cfg.yaml).
        "sb3_td3_cfg_entry_point": f"{agents.__name__}:sb3_td3_cfg.yaml",
        "sb3_ddpg_cfg_entry_point": f"{agents.__name__}:sb3_ddpg_cfg.yaml",
    },
)
