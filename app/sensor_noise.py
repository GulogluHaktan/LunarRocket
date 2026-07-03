from __future__ import annotations

from typing import Any

import numpy as np

from app.rl_types import normalize_quaternion_wxyz


DEFAULT_NOISE_CONFIG: dict[str, Any] = {
    "enabled": True,
    "sigma_gyro": 0.01,
    "sigma_acc": 0.05,
    "sigma_orientation_deg": 0.5,
    "sigma_alt": 0.05,
    "altimeter_dropout_prob": 0.01,
    "sigma_vel": 0.04,
    "sigma_lidar": 0.08,
    "lidar_dropout_prob": 0.02,
    "depth_noise_std": 0.02,
    "depth_dropout_pixel_prob": 0.01,
    "rgb_noise_std": 0.02,
    "rgb_brightness_range": [0.9, 1.1],
    "rgb_contrast_range": [0.9, 1.1],
}


def apply_imu_noise(
    angular_velocity: np.ndarray,
    linear_acceleration: np.ndarray,
    quaternion_wxyz: np.ndarray,
    rng: np.random.Generator,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = _cfg(config)
    if not cfg["enabled"]:
        return angular_velocity, linear_acceleration, normalize_quaternion_wxyz(quaternion_wxyz)
    gyro = np.asarray(angular_velocity, dtype=np.float32) + rng.normal(0.0, cfg["sigma_gyro"], 3).astype(np.float32)
    acc = np.asarray(linear_acceleration, dtype=np.float32) + rng.normal(0.0, cfg["sigma_acc"], 3).astype(np.float32)
    quat = perturb_quaternion(quaternion_wxyz, float(cfg["sigma_orientation_deg"]), rng)
    return gyro, acc, quat


def apply_altimeter_noise(
    altitude: float,
    previous_valid: float,
    max_altitude: float,
    rng: np.random.Generator,
    config: dict[str, Any] | None = None,
) -> float:
    cfg = _cfg(config)
    if not cfg["enabled"]:
        return float(altitude)
    if rng.random() < float(cfg["altimeter_dropout_prob"]):
        return float(np.clip(previous_valid, 0.0, max_altitude))
    return float(np.clip(altitude + rng.normal(0.0, float(cfg["sigma_alt"])), 0.0, max_altitude))


def apply_velocity_noise(velocity: np.ndarray, rng: np.random.Generator, config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = _cfg(config)
    if not cfg["enabled"]:
        return np.asarray(velocity, dtype=np.float32)
    return np.asarray(velocity, dtype=np.float32) + rng.normal(0.0, float(cfg["sigma_vel"]), 3).astype(np.float32)


def apply_lidar_noise(
    lidar: np.ndarray,
    max_distance: float,
    rng: np.random.Generator,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    cfg = _cfg(config)
    values = np.asarray(lidar, dtype=np.float32).copy()
    if not cfg["enabled"]:
        return np.clip(values, 0.0, max_distance)
    values += rng.normal(0.0, float(cfg["sigma_lidar"]), values.shape).astype(np.float32)
    dropout = rng.random(values.shape) < float(cfg["lidar_dropout_prob"])
    values[dropout] = max_distance
    return np.clip(values, 0.0, max_distance)


def apply_depth_noise(depth: np.ndarray, rng: np.random.Generator, config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = _cfg(config)
    values = np.asarray(depth, dtype=np.float32).copy()
    if not cfg["enabled"]:
        return np.clip(values, 0.0, 1.0)
    values += rng.normal(0.0, float(cfg["depth_noise_std"]), values.shape).astype(np.float32)
    dropout = rng.random(values.shape) < float(cfg["depth_dropout_pixel_prob"])
    values[dropout] = 1.0
    values = np.round(np.clip(values, 0.0, 1.0) * 255.0) / 255.0
    return values.astype(np.float32)


def apply_rgb_noise(rgb: np.ndarray, rng: np.random.Generator, config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = _cfg(config)
    values = np.asarray(rgb, dtype=np.float32)
    if values.max(initial=0.0) > 1.0:
        values = values / 255.0
    if not cfg["enabled"]:
        return np.clip(values, 0.0, 1.0).astype(np.float32)
    brightness = float(rng.uniform(*cfg["rgb_brightness_range"]))
    contrast = float(rng.uniform(*cfg["rgb_contrast_range"]))
    mean = np.mean(values, axis=(-2, -1), keepdims=True)
    values = (values - mean) * contrast + mean
    values = values * brightness
    values += rng.normal(0.0, float(cfg["rgb_noise_std"]), values.shape).astype(np.float32)
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def perturb_quaternion(quaternion_wxyz: np.ndarray, sigma_deg: float, rng: np.random.Generator) -> np.ndarray:
    q = normalize_quaternion_wxyz(quaternion_wxyz)
    sigma_rad = np.deg2rad(float(sigma_deg))
    axis_angle = rng.normal(0.0, sigma_rad, 3).astype(np.float32)
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-8:
        return q
    axis = axis_angle / angle
    half = angle * 0.5
    dq = np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float32)
    return quaternion_multiply(dq, q)


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = [float(v) for v in normalize_quaternion_wxyz(a)]
    bw, bx, by, bz = [float(v) for v in normalize_quaternion_wxyz(b)]
    return normalize_quaternion_wxyz(
        np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=np.float32,
        )
    )


def _cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    return {**DEFAULT_NOISE_CONFIG, **(config or {})}
