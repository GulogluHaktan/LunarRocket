"""Lunar lander direct workflow task."""

import gymnasium as gym

from . import agents

gym.register(
    id="LunarRocket-Lander-Direct-v0",
    entry_point=f"{__name__}.lunar_lander_env:LunarLanderEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lunar_lander_env:LunarLanderEnvCfg",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)
