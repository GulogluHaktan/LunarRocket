# LunarRocket RL Reward and Observation Report

## Simulation Constants

| Item | Value |
| --- | --- |
| Lunar gravity | `1.62 m/s^2` |
| Rocket mass | `1.63 kg` |
| Rocket visual scale | `6.0` |
| Max thrust | `30 N` |
| Physics dt | `0.008333 s` |
| Default episode max steps | `1000` |
| Default terrain patch | `80m x 80m` |
| Local landing detail | `512 x 512` mesh by default |

The RL policy target is selected before each episode. The agent does not need to visually discover the target; it must land safely at the provided target coordinate while using sensors to validate terrain and approach safety.

## Action Space

Continuous action vector:

```text
action = [main_thrust, gimbal_x, gimbal_y]
low    = [0.0, -1.0, -1.0]
high   = [1.0,  1.0,  1.0]
```

Meaning:

| Action | Meaning |
| --- | --- |
| `main_thrust` | normalized thrust command |
| `gimbal_x` | normalized pitch/roll gimbal command |
| `gimbal_y` | normalized yaw/second-axis gimbal command |

The actuator model supports clipping, first-order lag, command delay, thrust noise, and gimbal noise for sim-to-real robustness.

## Observation Space

Observation is dictionary-based:

```python
observation = {
    "state": np.ndarray(shape=(27,), dtype=np.float32),
    "lidar": np.ndarray(shape=(32,), dtype=np.float32),
    "depth": np.ndarray(shape=(1, 64, 64), dtype=np.float32),
    "rgb": np.ndarray(shape=(3, 84, 84), dtype=np.float32),
}
```

Recommended early SAC mode:

```text
state + lidar
```

Depth/RGB should be added after the state and LiDAR pipeline is stable.

## State Vector

State vector order:

```text
[
  dx, dy, dz,
  vx, vy, vz,
  qw, qx, qy, qz,
  wx, wy, wz,
  ax, ay, az,
  altitude_to_ground,
  main_thrust, gimbal_x, gimbal_y,
  previous_action_thrust, previous_action_gimbal_x, previous_action_gimbal_y,
  local_slope_at_target,
  local_roughness_at_target,
  safe_zone_score,
  terrain_height_at_target
]
```

Orientation uses quaternion `[qw, qx, qy, qz]`, not Euler angles.

## Sensors

| Sensor | Shape | Role |
| --- | --- | --- |
| LiDAR/raycast | `(32,)` | Sparse terrain distance checks |
| Depth camera | `(1, 64, 64)` | Geometric terrain perception |
| RGB camera | `(3, 84, 84)` | Vision validation, lighting/texture robustness |
| IMU | vector values | angular velocity, acceleration, orientation |
| Altimeter | scalar | altitude to local ground |
| Contact sensors | internal reward/classification | leg contact, body contact, contact force |

Contact sensors are not included in the default policy observation. They are used for reward and terminal classification.

## Reward Structure

Dense reward:

```text
reward =
    r_progress
  + r_target
  + r_stability
  + r_velocity
  + r_control
  + r_terrain
  + r_time
  + r_terminal
```

Reward terms:

| Term | Purpose |
| --- | --- |
| `progress` | reward reduction in target XY distance |
| `target` | small continuous penalty for target distance |
| `stability` | tilt and angular velocity penalty |
| `velocity` | horizontal and near-ground vertical speed penalty |
| `control` | fuel, smoothness, and saturation penalty |
| `terrain` | slope, roughness, unsafe-zone penalty |
| `time` | small hover-prevention penalty |
| `terminal` | landing/crash/timeout outcome |

Default weights:

```yaml
k_progress: 2.0
k_target: 0.03
k_tilt: 1.5
k_angvel: 0.05
k_vxy: 0.08
k_vz: 0.25
k_fuel: 0.01
k_smooth: 0.03
k_sat: 0.05
k_terrain: 1.0
time_penalty: -0.01
```

Terminal rewards:

```yaml
soft_landing: 80.0
acceptable_landing: 30.0
harsh_landing: -20.0
crash: -80.0
timeout: -30.0
```

## Landing Classification

Possible landing types:

```text
flying
soft_landing
acceptable_landing
harsh_landing
crash
timeout
```

Soft landing requires:

```text
all legs contact
no body contact
target distance < 1.0 m
vertical speed < 0.5 m/s
horizontal speed < 0.3 m/s
tilt < 5 deg
angular speed < 0.2 rad/s
safe_zone_score > 0.7
contact force below soft force limit
```

Acceptable landing requires:

```text
all legs contact
no body contact
target distance < 2.0 m
vertical speed < 1.0 m/s
horizontal speed < 0.7 m/s
tilt < 10 deg
angular speed < 0.5 rad/s
safe_zone_score > 0.5
contact force below acceptable force limit
```

Crash conditions include body contact, excessive contact force, vertical speed over `2.0 m/s` at contact, tilt over `25 deg`, invalid state, or leaving terrain bounds.

## Noise and Domain Randomization

Sensor noise wrappers support:

| Signal | Noise |
| --- | --- |
| IMU gyro | Gaussian rad/s noise |
| IMU acceleration | Gaussian m/s² noise |
| Orientation | small quaternion perturbation |
| Altimeter | Gaussian noise + dropout |
| Velocity | Gaussian noise |
| LiDAR | Gaussian noise + dropout |
| Depth | noise, dropout, quantization |
| RGB | brightness, contrast, Gaussian noise |

Actuator wrappers support:

```text
actual_action_t = alpha * previous_actual_action + (1 - alpha) * delayed_command
```

with delay, noise, and clipping.

## SAC Training Plan

Primary algorithm:

```text
SAC
```

Baseline:

```text
PPO
```

Suggested SAC hyperparameters:

```yaml
learning_rate: 3e-4
buffer_size: 500000
batch_size: 256
gamma: 0.99
tau: 0.005
train_freq: 1
gradient_steps: 1
ent_coef: auto
learning_starts: 10000
target_update_interval: 1
max_episode_steps: 1000
```

Training stages:

1. `state` only
2. `state + lidar`
3. `state + lidar + depth`
4. RGB/depth validation

## Implemented Modules

| Module | Purpose |
| --- | --- |
| `app/rl_types.py` | RL dataclasses and quaternion utilities |
| `app/observation_builder.py` | Builds normalized observation dictionaries |
| `app/reward_function.py` | Computes dense reward and debug info |
| `app/landing_classifier.py` | Classifies flying/landing/crash/timeout |
| `app/sensor_noise.py` | Sensor corruption/noise wrappers |
| `app/actuator_model.py` | Delay, lag, noise, clipping for actions |
| `app/env_lunar_landing.py` | Gymnasium-compatible environment wrapper |
| `app/world_adapter.py` | Connects `LunarLandingWorld` to RL types |
| `app/train_sac.py` | SAC training entrypoint |

## Current Limitations

The current `world_adapter` estimates some values until dedicated Isaac sensors are fully wired:

```text
contact sensors: heuristic from altitude
LiDAR: synthetic terrain ring query
depth/RGB: placeholder tensors in the adapter path
angular velocity: placeholder zero until rigid-body state accessor is wired
```

These are isolated in `app/world_adapter.py`, so replacing them with real Isaac sensor/contact APIs is the next integration task.
