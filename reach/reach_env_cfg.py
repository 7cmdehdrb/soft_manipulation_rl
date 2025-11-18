# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObject,
    DeformableObjectCfg,
)
from isaaclab.devices import DevicesCfg
from isaaclab.devices.gamepad import Se3GamepadCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.soft_manipulation.reach.mdp as mdp

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """
    Configuration for the scene with a robotic arm.
    시뮬레이션 세계를 구성하는 모든 물리적 객체를 정의합니다
    ArticulationCfg의 경우는 --task key arguments를 통하여 호출한 ReachEnvCfg 부모 클래스를 통하여 간접 선언됩니다
    source/isaaclab_tasks/isaaclab_tasks/manager_based/soft_manipulation/reach/config/franka/joint_pos_env_cfg.py/FrankaReachEnvCfg 참조
    """

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """
    Command terms for the MDP.
    에이전트가 달성해야 할 목표를 정의합니다.
    """

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.05),
            pos_y=(-0.8, -0.4),
            pos_z=(0.6, 1.2),
            roll=(-3.14, 3.14),
            pitch=(-3.14, 3.14),
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class ActionsCfg:
    """
    Action specifications for the MDP.
    에이전트가 환경에 적용할 수 있는 행동을 정의합니다.
    ReachEnvCfg 클래스의 필드를 오버라이딩합니다.
    """

    action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """
    Observation specifications for the MDP.
    에이전트가 환경으로부터 받는 정보를 정의합니다.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "ee_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """
    Configuration for events.
    환경 리셋 시 발생하는 이벤트를 정의합니다.
    """

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_deformables = EventTerm(func=mdp.reset_deformable_body, mode="reset")


@configclass
class RewardsCfg:
    """
    Reward terms for the MDP.
    에이전트의 행동에 대한 보상을 정의합니다.
    RewTerm 인스턴스를 선언하는 방법으로 리워드 및 패널티를 정의하며, weight를 통하여 가중치를 곱할 수 있습니다.
    func는 리워드를 정의하는 callable 인스턴스이며, torch 타입을 리턴합니다.
    """

    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Cylinder"),
            "command_name": "ee_pose",
        },
    )  # 0.0~5.0 정도

    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Cylinder"),
            "std": 0.1,
            "command_name": "ee_pose",
        },
    )  # 0.0~0.1 정도

    joint_position_std_tracking = RewTerm(
        func=mdp.joint_pos_std,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )  # 0.0~0.2 정도

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """
    Termination terms for the MDP.
    에피소드가 종료되는 조건을 정의합니다.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """
    Curriculum terms for the MDP.
    학습이 진행됨에 따라 난이도를 조절합니다.
    """

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500},
    )


##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """
    Configuration for the reach end-effector pose tracking environment.
    모든 컴포넌트를 통합하는 최상위 설정 클래스입니다.
    이후에, --task key arguments에 의해 호출된 부모 클래스로 대체됩니다.
    """

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0

        # self.sim.physx.gpu_max_soft_body_contacts = 2**21
        # self.sim.physx.gpu_temp_buffer_capacity = 2**25
        # self.sim.physx.gpu_collision_stack_size = 2**27
