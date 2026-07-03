from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from app.actuator_model import ActuatorState, process_action
from app.observation_builder import build_observation
from app.reward_function import compute_reward
from app.rl_types import ContactState, EpisodeTimeState, RewardMemory, RocketKinematicState, TerrainSafetyState

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional until training dependencies are installed.
    gym = None
    spaces = None


class LunarWorldAdapter(Protocol):
    def reset(self) -> tuple[RocketKinematicState, dict[str, np.ndarray], TerrainSafetyState, ContactState]:
        ...

    def step(
        self, action: np.ndarray
    ) -> tuple[RocketKinematicState, dict[str, np.ndarray], TerrainSafetyState, ContactState]:
        ...


class LunarLandingIsaacEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(self, world_adapter: LunarWorldAdapter | None, config: dict[str, Any] | None = None):
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required for LunarLandingIsaacEnv")
        self.world_adapter = world_adapter
        self.config = config or {}
        self.rng = np.random.default_rng(int(self.config.get("seed", 0)))
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_config = self._observation_config()
        self.observation_space = self._build_observation_space()
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.actuator_state = ActuatorState()
        self.prev_metrics = RewardMemory(target_distance_xy=0.0)
        self.step_count = 0

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.world_adapter is None:
            raise RuntimeError("LunarLandingIsaacEnv requires a world_adapter before use")
        state, sensors, terrain, contact = self.world_adapter.reset()
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.actuator_state = ActuatorState()
        self.prev_metrics = RewardMemory(target_distance_xy=_target_distance_xy(state))
        self.step_count = 0
        obs = build_observation(state, sensors, terrain, self.prev_action, self.observation_config)
        return obs, {"landing_type": "flying"}

    def step(self, action: np.ndarray):
        if self.world_adapter is None:
            raise RuntimeError("LunarLandingIsaacEnv requires a world_adapter before use")
        actual_action, self.actuator_state = process_action(action, self.actuator_state, self.rng, self.config.get("actuator"))
        state, sensors, terrain, contact = self.world_adapter.step(actual_action)
        self.step_count += 1
        time_state = EpisodeTimeState(
            step_count=self.step_count,
            elapsed_s=self.step_count * float(self.config.get("control_dt", 0.05)),
            max_steps=int(self.config.get("max_episode_steps", 1000)),
            max_episode_s=float(self.config.get("max_episode_s", 50.0)),
        )
        reward, terminated, info = compute_reward(
            state,
            actual_action,
            self.prev_action,
            contact,
            terrain,
            self.prev_metrics,
            time_state,
            self.config.get("reward"),
        )
        self.prev_action = actual_action.copy()
        self.prev_metrics = RewardMemory(target_distance_xy=info["metrics"]["target_distance_xy"])
        obs = build_observation(state, sensors, terrain, self.prev_action, self.observation_config)
        truncated = info["metrics"]["landing_type"] == "timeout"
        return obs, reward, terminated, truncated, info

    def _observation_config(self) -> dict[str, Any]:
        observation = dict(self.config.get("observation", {}))
        mode = str(observation.get("early_sac_mode", observation.get("mode", "state_lidar")))
        if mode == "state_only":
            observation["include_depth"] = False
            observation["include_rgb"] = False
            observation["include_lidar"] = False
        elif mode == "state_lidar":
            observation["include_depth"] = False
            observation["include_rgb"] = False
            observation["include_lidar"] = True
        elif mode == "state_lidar_depth":
            observation["include_depth"] = True
            observation["include_rgb"] = False
            observation["include_lidar"] = True
        else:
            observation["include_depth"] = True
            observation["include_rgb"] = True
            observation["include_lidar"] = True
        return observation

    def _build_observation_space(self):
        fields = {
            "state": spaces.Box(low=-1.0, high=1.0, shape=(27,), dtype=np.float32),
        }
        if bool(self.observation_config.get("include_lidar", True)):
            fields["lidar"] = spaces.Box(low=0.0, high=1.0, shape=(32,), dtype=np.float32)
        if bool(self.observation_config.get("include_depth", False)):
            fields["depth"] = spaces.Box(low=0.0, high=1.0, shape=(1, 64, 64), dtype=np.float32)
        if bool(self.observation_config.get("include_rgb", False)):
            fields["rgb"] = spaces.Box(low=0.0, high=1.0, shape=(3, 84, 84), dtype=np.float32)
        return spaces.Dict(fields)


def _target_distance_xy(state: RocketKinematicState) -> float:
    delta = np.asarray(state.target_position, dtype=np.float32) - np.asarray(state.position, dtype=np.float32)
    return float(np.linalg.norm(delta[:2]))
