from __future__ import annotations

from typing import Any

import numpy as np

from app.rl_types import RocketKinematicState, TerrainSafetyState, normalize_quaternion_wxyz, target_delta


DEFAULT_OBSERVATION_CONFIG: dict[str, Any] = {
    "patch_size_m": 80.0,
    "max_velocity_mps": 20.0,
    "max_angular_velocity_rps": 5.0,
    "max_acceleration_mps2": 30.0,
    "max_altitude_m": 60.0,
    "max_lidar_distance_m": 30.0,
    "max_depth_m": 30.0,
    "include_lidar": True,
    "include_depth": True,
    "include_rgb": True,
    "rgb_float": True,
}


def build_observation(
    state: RocketKinematicState,
    sensors: dict[str, np.ndarray] | None,
    terrain: TerrainSafetyState,
    previous_action: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    cfg = {**DEFAULT_OBSERVATION_CONFIG, **(config or {})}
    sensors = sensors or {}

    state_vector = build_state_vector(state, terrain, previous_action, cfg)

    obs: dict[str, np.ndarray] = {
        "state": state_vector.astype(np.float32),
    }

    if bool(cfg.get("include_lidar", True)):
        obs["lidar"] = normalize_lidar(sensors.get("lidar"), cfg).astype(np.float32)
    if bool(cfg.get("include_depth", True)):
        obs["depth"] = normalize_depth(sensors.get("depth"), cfg).astype(np.float32)
    if bool(cfg.get("include_rgb", True)):
        obs["rgb"] = normalize_rgb(sensors.get("rgb"), cfg)

    return obs


def build_state_vector(
    state: RocketKinematicState,
    terrain: TerrainSafetyState,
    previous_action: np.ndarray,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    cfg = {**DEFAULT_OBSERVATION_CONFIG, **(config or {})}
    patch_size = float(cfg["patch_size_m"])
    max_velocity = float(cfg["max_velocity_mps"])
    max_angvel = float(cfg["max_angular_velocity_rps"])
    max_acc = float(cfg["max_acceleration_mps2"])
    max_altitude = float(cfg["max_altitude_m"])

    delta = np.asarray(target_delta(state), dtype=np.float32) / max(patch_size, 1e-6)
    velocity = np.asarray(state.linear_velocity, dtype=np.float32) / max(max_velocity, 1e-6)
    quaternion = normalize_quaternion_wxyz(state.quaternion_wxyz)
    angular_velocity = np.asarray(state.angular_velocity, dtype=np.float32) / max(max_angvel, 1e-6)
    acceleration = np.asarray(state.linear_acceleration, dtype=np.float32) / max(max_acc, 1e-6)
    altitude = np.array([state.altitude_to_ground / max(max_altitude, 1e-6)], dtype=np.float32)

    previous_action = np.asarray(previous_action, dtype=np.float32)
    if previous_action.size < 3:
        previous_action = np.pad(previous_action, (0, 3 - previous_action.size), constant_values=0.0)

    actuator = np.array([state.main_thrust, state.gimbal_x, state.gimbal_y], dtype=np.float32)
    terrain_values = np.array(
        [
            terrain.local_slope_deg / 45.0,
            terrain.local_roughness_m,
            terrain.safe_zone_score,
            terrain.terrain_height_at_target / max_altitude,
        ],
        dtype=np.float32,
    )

    vector = np.concatenate(
        [
            delta,
            velocity,
            quaternion,
            angular_velocity,
            acceleration,
            altitude,
            actuator,
            previous_action[:3],
            terrain_values,
        ]
    )
    return np.clip(vector, -1.0, 1.0).astype(np.float32)


def normalize_lidar(lidar: np.ndarray | None, config: dict[str, Any]) -> np.ndarray:
    if lidar is None:
        return np.ones(32, dtype=np.float32)
    max_distance = float(config.get("max_lidar_distance_m", 30.0))
    values = np.asarray(lidar, dtype=np.float32).reshape(-1)
    return np.clip(values / max(max_distance, 1e-6), 0.0, 1.0)


def normalize_depth(depth: np.ndarray | None, config: dict[str, Any]) -> np.ndarray:
    if depth is None:
        return np.ones((1, 64, 64), dtype=np.float32)
    max_depth = float(config.get("max_depth_m", 30.0))
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim == 2:
        values = values[None, :, :]
    return np.clip(values / max(max_depth, 1e-6), 0.0, 1.0)


def normalize_rgb(rgb: np.ndarray | None, config: dict[str, Any]) -> np.ndarray:
    if rgb is None:
        shape = (3, 84, 84)
        return np.zeros(shape, dtype=np.float32 if bool(config.get("rgb_float", True)) else np.uint8)
    values = np.asarray(rgb)
    if values.ndim == 3 and values.shape[-1] in (1, 3, 4):
        values = np.moveaxis(values[..., :3], -1, 0)
    elif values.ndim == 2:
        values = values[None, :, :]
    if bool(config.get("rgb_float", True)):
        return np.clip(values.astype(np.float32) / 255.0, 0.0, 1.0)
    return np.clip(values, 0, 255).astype(np.uint8)
