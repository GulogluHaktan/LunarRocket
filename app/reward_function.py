from __future__ import annotations

from typing import Any

import numpy as np

from app.landing_classifier import classify_landing
from app.rl_types import (
    ContactState,
    EpisodeTimeState,
    RewardMemory,
    RocketKinematicState,
    TerrainSafetyState,
    angular_speed,
    horizontal_speed,
    target_distance_xy,
    tilt_angle_deg_from_quaternion,
    up_alignment_from_quaternion,
    vertical_speed,
)


DEFAULT_REWARD_CONFIG: dict[str, float | dict[str, float]] = {
    "k_progress": 2.0,
    "k_target": 0.03,
    "k_tilt": 1.5,
    "k_angvel": 0.05,
    "k_vxy": 0.08,
    "k_vz": 0.25,
    "k_fuel": 0.01,
    "k_smooth": 0.03,
    "k_sat": 0.05,
    "k_terrain": 1.0,
    "time_penalty": -0.01,
    "terminal_rewards": {
        "soft_landing": 80.0,
        "acceptable_landing": 30.0,
        "harsh_landing": -20.0,
        "crash": -80.0,
        "timeout": -30.0,
        "flying": 0.0,
    },
}


TERMINAL_TYPES = {"soft_landing", "acceptable_landing", "harsh_landing", "crash", "timeout"}


def compute_reward(
    state: RocketKinematicState,
    action: np.ndarray,
    prev_action: np.ndarray,
    contact: ContactState,
    terrain: TerrainSafetyState,
    prev_metrics: RewardMemory,
    time_state: EpisodeTimeState,
    config: dict[str, Any] | None = None,
) -> tuple[float, bool, dict[str, Any]]:
    cfg = _merge_reward_config(config)
    action = np.asarray(action, dtype=np.float32)
    prev_action = np.asarray(prev_action, dtype=np.float32)

    distance_xy = target_distance_xy(state)
    h_speed = horizontal_speed(state)
    v_speed = vertical_speed(state)
    a_speed = angular_speed(state)

    up_alignment = up_alignment_from_quaternion(state.quaternion_wxyz)
    tilt_penalty = 1.0 - up_alignment
    tilt_angle_deg = tilt_angle_deg_from_quaternion(state.quaternion_wxyz)

    near_ground_weight = float(np.exp(-state.altitude_to_ground / 5.0))
    near_landing_weight = float(np.exp(-state.altitude_to_ground / 10.0))

    r_progress = float(cfg["k_progress"]) * (prev_metrics.target_distance_xy - distance_xy)
    r_target = -float(cfg["k_target"]) * distance_xy
    r_stability = -float(cfg["k_tilt"]) * tilt_penalty - float(cfg["k_angvel"]) * a_speed
    r_velocity = -float(cfg["k_vxy"]) * h_speed - float(cfg["k_vz"]) * v_speed * near_ground_weight

    action_delta = action - prev_action
    saturation = compute_action_saturation(action)
    thrust = float(np.clip(action[0], 0.0, 1.0)) if action.size else 0.0
    r_control = (
        -float(cfg["k_fuel"]) * thrust * thrust
        -float(cfg["k_smooth"]) * float(np.linalg.norm(action_delta))
        -float(cfg["k_sat"]) * saturation
    )

    terrain_hazard = (
        0.4 * float(np.clip(terrain.slope_norm, 0.0, 1.0))
        + 0.4 * float(np.clip(terrain.roughness_norm, 0.0, 1.0))
        + 0.2 * (1.0 - float(np.clip(terrain.safe_zone_score, 0.0, 1.0)))
    )
    r_terrain = -float(cfg["k_terrain"]) * terrain_hazard * near_landing_weight
    r_time = float(cfg["time_penalty"])

    landing_type = classify_landing(state, contact, terrain, time_state, config.get("classification") if config else None)
    terminal_rewards = cfg["terminal_rewards"]
    r_terminal = float(terminal_rewards.get(landing_type, 0.0))
    terminated = landing_type in TERMINAL_TYPES

    reward = float(r_progress + r_target + r_stability + r_velocity + r_control + r_terrain + r_time + r_terminal)

    info = {
        "reward_terms": {
            "progress": float(r_progress),
            "target": float(r_target),
            "stability": float(r_stability),
            "velocity": float(r_velocity),
            "control": float(r_control),
            "terrain": float(r_terrain),
            "time": float(r_time),
            "terminal": float(r_terminal),
            "total": reward,
        },
        "metrics": {
            "landing_type": landing_type,
            "target_distance_xy": float(distance_xy),
            "altitude_to_ground": float(state.altitude_to_ground),
            "horizontal_speed": float(h_speed),
            "vertical_speed": float(v_speed),
            "up_alignment": float(up_alignment),
            "tilt_angle_deg": float(tilt_angle_deg),
            "angular_speed": float(a_speed),
            "safe_zone_score": float(terrain.safe_zone_score),
            "local_slope": float(terrain.local_slope_deg),
            "local_roughness": float(terrain.local_roughness_m),
            "contact_force_max": float(contact.max_force),
            "terrain_hazard": float(terrain_hazard),
            "action_saturation": float(saturation),
        },
    }
    return reward, terminated, info


def compute_action_saturation(action: np.ndarray) -> float:
    if action.size == 0:
        return 0.0
    expected_low = np.array([0.0, -1.0, -1.0], dtype=np.float32)[: action.size]
    expected_high = np.array([1.0, 1.0, 1.0], dtype=np.float32)[: action.size]
    center = (expected_low + expected_high) * 0.5
    half_range = np.maximum((expected_high - expected_low) * 0.5, 1e-6)
    normalized = np.abs((action - center) / half_range)
    return float(np.mean(np.clip(normalized, 0.0, 1.0)))


def _merge_reward_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_REWARD_CONFIG)
    merged["terminal_rewards"] = dict(DEFAULT_REWARD_CONFIG["terminal_rewards"])  # type: ignore[index]
    if not config:
        return merged
    for key, value in config.items():
        if key == "terminal_rewards" and isinstance(value, dict):
            merged["terminal_rewards"].update(value)
        elif key != "classification":
            merged[key] = value
    return merged
