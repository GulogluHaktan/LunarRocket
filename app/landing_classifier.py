from __future__ import annotations

from typing import Any

import numpy as np

from app.rl_types import (
    ContactState,
    EpisodeTimeState,
    RocketKinematicState,
    TerrainSafetyState,
    angular_speed,
    horizontal_speed,
    target_distance_xy,
    tilt_angle_deg_from_quaternion,
    vertical_speed,
)


DEFAULT_CLASSIFICATION_CONFIG: dict[str, float] = {
    "soft_target_distance_m": 1.0,
    "soft_vertical_speed_mps": 0.5,
    "soft_horizontal_speed_mps": 0.3,
    "soft_tilt_deg": 5.0,
    "soft_angular_speed_rps": 0.2,
    "soft_safe_zone_score": 0.7,
    "soft_force_limit_n": 60.0,
    "acceptable_target_distance_m": 2.0,
    "acceptable_vertical_speed_mps": 1.0,
    "acceptable_horizontal_speed_mps": 0.7,
    "acceptable_tilt_deg": 10.0,
    "acceptable_angular_speed_rps": 0.5,
    "acceptable_safe_zone_score": 0.5,
    "acceptable_force_limit_n": 120.0,
    "crash_force_limit_n": 200.0,
    "crash_vertical_speed_mps": 2.0,
    "crash_tilt_deg": 25.0,
    "ground_altitude_m": 0.2,
}


def classify_landing(
    state: RocketKinematicState,
    contact: ContactState,
    terrain: TerrainSafetyState,
    time_state: EpisodeTimeState,
    config: dict[str, Any] | None = None,
) -> str:
    cfg = {**DEFAULT_CLASSIFICATION_CONFIG, **(config or {})}

    if state.invalid or state.out_of_bounds or not _is_state_finite(state):
        return "crash"

    if contact.body_contact or contact.max_force > float(cfg["crash_force_limit_n"]):
        return "crash"

    if time_state.step_count >= time_state.max_steps or time_state.elapsed_s >= time_state.max_episode_s:
        return "timeout"

    if not contact.any_contact and state.altitude_to_ground > float(cfg["ground_altitude_m"]):
        return "flying"

    tilt_deg = tilt_angle_deg_from_quaternion(state.quaternion_wxyz)
    if tilt_deg > float(cfg["crash_tilt_deg"]):
        return "crash"

    if vertical_speed(state) > float(cfg["crash_vertical_speed_mps"]):
        return "crash"

    distance_xy = target_distance_xy(state)
    h_speed = horizontal_speed(state)
    a_speed = angular_speed(state)

    if (
        contact.all_legs_contact
        and not contact.body_contact
        and distance_xy < float(cfg["soft_target_distance_m"])
        and vertical_speed(state) < float(cfg["soft_vertical_speed_mps"])
        and h_speed < float(cfg["soft_horizontal_speed_mps"])
        and tilt_deg < float(cfg["soft_tilt_deg"])
        and a_speed < float(cfg["soft_angular_speed_rps"])
        and terrain.safe_zone_score > float(cfg["soft_safe_zone_score"])
        and contact.max_force < float(cfg["soft_force_limit_n"])
    ):
        return "soft_landing"

    if (
        contact.all_legs_contact
        and not contact.body_contact
        and distance_xy < float(cfg["acceptable_target_distance_m"])
        and vertical_speed(state) < float(cfg["acceptable_vertical_speed_mps"])
        and h_speed < float(cfg["acceptable_horizontal_speed_mps"])
        and tilt_deg < float(cfg["acceptable_tilt_deg"])
        and a_speed < float(cfg["acceptable_angular_speed_rps"])
        and terrain.safe_zone_score > float(cfg["acceptable_safe_zone_score"])
        and contact.max_force < float(cfg["acceptable_force_limit_n"])
    ):
        return "acceptable_landing"

    if contact.any_contact or state.altitude_to_ground <= float(cfg["ground_altitude_m"]):
        return "harsh_landing"

    return "flying"


def _is_state_finite(state: RocketKinematicState) -> bool:
    arrays = (
        state.position,
        state.target_position,
        state.linear_velocity,
        state.quaternion_wxyz,
        state.angular_velocity,
        state.linear_acceleration,
    )
    scalars = (state.altitude_to_ground, state.main_thrust, state.gimbal_x, state.gimbal_y)
    return all(bool(np.all(np.isfinite(array))) for array in arrays) and all(np.isfinite(scalar) for scalar in scalars)
