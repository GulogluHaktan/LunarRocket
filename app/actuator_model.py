from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


DEFAULT_ACTUATOR_CONFIG: dict[str, Any] = {
    "enabled": True,
    "alpha": 0.7,
    "delay_steps": 1,
    "thrust_noise_std": 0.02,
    "gimbal_noise_std": 0.01,
    "low": [0.0, -1.0, -1.0],
    "high": [1.0, 1.0, 1.0],
}


@dataclass
class ActuatorState:
    previous_actual_action: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    command_queue: list[np.ndarray] = field(default_factory=list)


def process_action(
    commanded_action: np.ndarray,
    state: ActuatorState,
    rng: np.random.Generator,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, ActuatorState]:
    cfg = {**DEFAULT_ACTUATOR_CONFIG, **(config or {})}
    command = clip_action_to_limits(commanded_action, cfg)
    if not cfg["enabled"]:
        state.previous_actual_action = command
        return command, state

    delayed = apply_action_delay(command, state, int(cfg["delay_steps"]))
    lagged = apply_action_lag(delayed, state.previous_actual_action, float(cfg["alpha"]))
    noisy = apply_actuator_noise(lagged, rng, cfg)
    actual = clip_action_to_limits(noisy, cfg)
    state.previous_actual_action = actual
    return actual, state


def apply_action_delay(commanded_action: np.ndarray, state: ActuatorState, delay_steps: int) -> np.ndarray:
    state.command_queue.append(np.asarray(commanded_action, dtype=np.float32))
    if len(state.command_queue) <= max(0, delay_steps):
        return state.previous_actual_action.copy()
    return state.command_queue.pop(0)


def apply_action_lag(commanded_action: np.ndarray, previous_actual_action: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * np.asarray(previous_actual_action, dtype=np.float32) + (1.0 - alpha) * np.asarray(
        commanded_action, dtype=np.float32
    )


def apply_actuator_noise(action: np.ndarray, rng: np.random.Generator, config: dict[str, Any]) -> np.ndarray:
    noisy = np.asarray(action, dtype=np.float32).copy()
    if noisy.size > 0:
        noisy[0] += float(rng.normal(0.0, float(config["thrust_noise_std"])))
    if noisy.size > 1:
        noisy[1:] += rng.normal(0.0, float(config["gimbal_noise_std"]), noisy.size - 1).astype(np.float32)
    return noisy


def clip_action_to_limits(action: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = {**DEFAULT_ACTUATOR_CONFIG, **(config or {})}
    values = np.asarray(action, dtype=np.float32)
    low = np.asarray(cfg["low"], dtype=np.float32)[: values.size]
    high = np.asarray(cfg["high"], dtype=np.float32)[: values.size]
    return np.clip(values, low, high).astype(np.float32)
