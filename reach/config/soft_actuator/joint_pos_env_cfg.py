import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.schemas import DeformableBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.soft_manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.soft_manipulation.reach.reach_env_cfg import (
    ReachEnvCfg,
)


SOFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/min/7cmdehdrb/isaac_lab/IsaacLab/Collected_ggg/single_finger.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        # deformable_props=DeformableBodyPropertiesCfg(
        #     deformable_enabled=True,
        #     self_collision=True,
        #     collision_simplification_remeshing=True,
        # ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "joint1:0": 0.0,  # x - rotation
            "joint1:1": 0.0,  # y - rotation
            "joint2:0": 0.0,
            "joint2:1": 0.0,
            "joint3:0": 0.0,
            "joint3:1": 0.0,
            "joint4:0": 0.0,
            "joint4:1": 0.0,
            "joint5:0": 0.0,
            "joint5:1": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint1:0",
                "joint1:1",
                "joint2:0",
                "joint2:1",
                "joint3:0",
                "joint3:1",
                "joint4:0",
                "joint4:1",
                "joint5:0",
                "joint5:1",
            ],
            velocity_limit={
                "joint1:0": 3.14,
                "joint1:1": 3.14,
                "joint2:0": 3.14,
                "joint2:1": 3.14,
                "joint3:0": 3.14,
                "joint3:1": 3.14,
                "joint4:0": 3.14,
                "joint4:1": 3.14,
                "joint5:0": 3.14,
                "joint5:1": 3.14,
            },
            effort_limit={
                "joint1:0": 150,
                "joint1:1": 150,
                "joint2:0": 150,
                "joint2:1": 150,
                "joint3:0": 150,
                "joint3:1": 150,
                "joint4:0": 150,
                "joint4:1": 150,
                "joint5:0": 150,
                "joint5:1": 150,
            },
            stiffness={
                "joint1:0": 30000.0,
                "joint1:1": 1000.0,
                "joint2:0": 30000.0,
                "joint2:1": 1000.0,
                "joint3:0": 30000.0,
                "joint3:1": 1000.0,
                "joint4:0": 30000.0,
                "joint4:1": 1000.0,
                "joint5:0": 30000.0,
                "joint5:1": 1000.0,
            },
            damping={
                "joint1:0": 100.0,
                "joint1:1": 10.0,
                "joint2:0": 100.0,
                "joint2:1": 10.0,
                "joint3:0": 100.0,
                "joint3:1": 10.0,
                "joint4:0": 100.0,
                "joint4:1": 10.0,
                "joint5:0": 100.0,
                "joint5:1": 10.0,
            },
        ),
    },
)


@configclass
class SoftReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = SOFT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.action = mdp.JointPositionActionCfg(
            asset_name="robot",
            debug_vis=True,
            joint_names=[
                "joint1:0",
                "joint1:1",
                "joint2:0",
                "joint2:1",
                "joint3:0",
                "joint3:1",
                "joint4:0",
                "joint4:1",
                "joint5:0",
                "joint5:1",
            ],
            scale=1.0,
            use_default_offset=True,
        )
        # override command generator body
        # end-effector is along z-direction

        self.commands.ee_pose.body_name = "Cylinder"
