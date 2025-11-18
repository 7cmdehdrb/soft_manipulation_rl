# Soft Manipulation Reach Environment

A reinforcement learning environment for training soft actuator manipulation tasks using Isaac Lab and RSL-RL.

## 📋 Overview

This project implements an end-effector pose tracking task for a soft actuator robotic system. The robot learns to reach target 6D poses (position + orientation) using joint position control with implicit actuators.

### Key Features

- **Soft Actuator Control**: Custom soft finger actuator with 10 degrees of freedom (5 segments × 2 DOF each)
- **6D Pose Tracking**: Target position and orientation tracking for the end-effector
- **Curriculum Learning**: Progressive difficulty adjustment during training
- **PPO Algorithm**: Implementation using RSL-RL framework
- **Parallel Simulation**: GPU-accelerated parallel environments (up to 4096 environments)

## 🏗️ Architecture

### Code Structure

```
reach/
├── reach_env_cfg.py              # Base environment configuration
├── mdp/
│   ├── __init__.py
│   └── rewards.py                # Custom reward functions
└── config/
    └── soft_actuator/
        ├── joint_pos_env_cfg.py  # Soft actuator specific config
        └── agents/
            ├── rsl_rl_ppo_cfg.py # PPO hyperparameters
            ├── rl_games_ppo_cfg.yaml
            └── skrl_ppo_cfg.yaml
```

### Configuration Hierarchy

```
ReachEnvCfg (reach_env_cfg.py)
    ↓ inherits
SoftReachEnvCfg (joint_pos_env_cfg.py)
```

- **`ReachEnvCfg`**: Base configuration defining the generic reach task
- **`SoftReachEnvCfg`**: Specialized configuration for soft actuator robot

## 🔧 Components

### 1. Scene Configuration (`ReachSceneCfg`)

Defines the simulation world:
- Ground plane
- Robot articulation (defined in child class)
- Dome lighting

### 2. Commands (`CommandsCfg`)

Goal specification for the agent:
- **`ee_pose`**: Uniform random pose sampling in specified ranges
  - Position: x ∈ [-0.05, 0.05], y ∈ [-0.8, -0.4], z ∈ [0.6, 1.2]
  - Orientation: Full rotation range for roll, pitch, yaw

### 3. Actions (`ActionsCfg`)

Agent control interface:
- **Type**: Joint position control
- **DOF**: 10 joints (joint1:0, joint1:1, ..., joint5:0, joint5:1)
- **Scale**: 1.0 (direct position commands)

### 4. Observations (`ObservationsCfg`)

State information provided to the policy:
- Joint positions (relative, with noise)
- Joint velocities (relative, with noise)
- Target pose command (6D)
- Previous actions

### 5. Rewards (`RewardsCfg`)

Multi-objective reward function:

| Reward Term | Weight | Description |
|-------------|--------|-------------|
| `end_effector_position_tracking` | -0.2 | L2 distance to target position |
| `end_effector_position_tracking_fine_grained` | 3.0 | Tanh-shaped position reward (std=0.1) |
| `joint_position_std_tracking` | -0.5 | Penalize joint position variance |
| `action_rate` | -0.0001* | L2 penalty on action changes |
| `joint_vel` | -0.0001* | L2 penalty on joint velocities |

*Weights increase via curriculum learning

### 6. Events (`EventCfg`)

Environment reset behavior:
- **`reset_robot_joints`**: Randomize joint positions (50%-150% of default)
- **`reset_deformables`**: Reset deformable body properties

### 7. Curriculum (`CurriculumCfg`)

Progressive difficulty:
- `action_rate` penalty: -0.0001 → -0.005 over 4500 steps
- `joint_vel` penalty: -0.0001 → -0.001 over 4500 steps

## 🤖 Robot Configuration

### Soft Actuator Specifications

```python
# 5 segments, each with 2 DOF (rotation around x and y axes)
Joints: joint1:0, joint1:1, joint2:0, joint2:1, ..., joint5:0, joint5:1

# Actuator Properties
Velocity Limit: 3.14 rad/s (all joints)
Effort Limit: 150 N·m (all joints)

# PD Control Gains
Stiffness: 
  - X-axis (joint*:0): 30,000.0
  - Y-axis (joint*:1): 1,000.0
Damping:
  - X-axis (joint*:0): 100.0
  - Y-axis (joint*:1): 10.0
```

### End-Effector

- **Body Name**: `"Cylinder"`
- **Frame**: Target poses specified in robot base frame

## 🚀 Training

### Environment Registration

```python
gym.register(
    id="Isaac-Reach-Soft-Manipulation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": "...joint_pos_env_cfg:SoftReachEnvCfg",
        "rsl_rl_cfg_entry_point": "...rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
    }
)
```

### Training Command

```bash
# Using RSL-RL
python scripts/train.py --task Isaac-Reach-Soft-Manipulation-v0 \
    --num_envs 4096 \
    --headless

# With visualization
python scripts/train.py --task Isaac-Reach-Soft-Manipulation-v0 \
    --num_envs 1024
```

### Hyperparameters (RSL-RL PPO)

```yaml
Algorithm:
  - Learning Rate: 1e-3 (adaptive schedule)
  - Discount Factor (γ): 0.99
  - GAE Lambda (λ): 0.95
  - Clip Param: 0.2
  - Entropy Coefficient: 0.001
  - Value Loss Coefficient: 1.0

Network:
  - Actor Hidden Dims: [64, 64]
  - Critic Hidden Dims: [64, 64]
  - Activation: ELU

Training:
  - Steps per Environment: 24
  - Learning Epochs: 8
  - Mini-batches: 4
  - Max Iterations: 10,000
  - Save Interval: 50
```

## 📊 Simulation Settings

```python
# Timing
Simulation Timestep: 1/60 s (≈16.67 ms)
Decimation: 2
Control Frequency: 30 Hz
Episode Length: 12 seconds (360 control steps)

# Physics
Solver Position Iterations: 8
Solver Velocity Iterations: 0
Self-Collision: Enabled
```

## 🎯 Custom Reward Functions

### `position_command_error`
Computes L2 norm of position error between current and target end-effector position.

### `position_command_error_tanh`
Maps position error through tanh kernel for smooth, bounded reward:
```python
reward = 1 - tanh(distance / std)
```

### `joint_pos_std`
Penalizes variability in joint positions with weighted combination:
```python
combined_std = 0.7 * std(odd_joints) + 0.3 * std(even_joints)
```

## 📝 Key Configuration Parameters

### Modifying Target Ranges

Edit `reach_env_cfg.py`:
```python
commands = CommandsCfg(
    ee_pose = mdp.UniformPoseCommandCfg(
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.05),  # Adjust workspace
            pos_y=(-0.8, -0.4),
            pos_z=(0.6, 1.2),
            # ...
        ),
    )
)
```

### Adjusting Reward Weights

Edit `reach_env_cfg.py`:
```python
rewards = RewardsCfg(
    end_effector_position_tracking=RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,  # Modify weight
    ),
    # ...
)
```

### Changing Robot Model

Edit `joint_pos_env_cfg.py`:
```python
SOFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/path/to/your/robot.usd",  # Change USD path
        # ...
    ),
)
```

## 🔍 Debugging

### Visualization

Enable debug visualization in configuration:
```python
self.actions.action = mdp.JointPositionActionCfg(
    debug_vis=True,  # Show action arrows
)

self.commands.ee_pose = mdp.UniformPoseCommandCfg(
    debug_vis=True,  # Show target pose
)
```

### Common Issues

1. **Deformable Body Reset**: The `reset_deformable_body` function toggles deformable properties to ensure proper reset.
2. **Joint Limits**: Ensure actuator limits match USD file joint limits.
3. **Collision**: Self-collision is enabled; verify collision meshes in USD.

## 📚 Dependencies

- Isaac Lab (Isaac Sim)
- RSL-RL
- PyTorch
- NumPy

## 📄 License

BSD-3-Clause (Isaac Lab Project)

## 🙏 Acknowledgments

Based on the Isaac Lab framework by NVIDIA and the Isaac Lab Project Developers.
