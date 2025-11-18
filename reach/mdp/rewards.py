# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

from isaaclab.sim import schemas
from isaaclab.sim.schemas import DeformableBodyPropertiesCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


deformable_enabled: bool = False


def position_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]  # desired position in body frame
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )  # desired position in world frame

    # obtain current position
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore

    """
    print(command)

    print(asset.data.root_pos_w)
    print(asset.data.root_quat_w)

    print(asset.data.body_pos_w)

    print(asset_cfg.body_ids)
    print(asset_cfg.body_names)
    
    print("=====")
    
    tensor([[ 0.0376, -0.2706,  0.8663,  0.4880,  0.2251,  0.7332,  0.4167]], # Target 6D Pose
       device='cuda:0')
       
    tensor([[-1.9860e-03, -8.7840e-02,  2.3306e+00]], device='cuda:0') # asset.data.root_pos_w
    tensor([[-1.7114e-08,  1.0000e+00,  1.7216e-10, -3.1799e-16]], device='cuda:0') asset.data.root_quat_w
    
    tensor([[[-1.9860e-03, -8.7840e-02,  2.3306e+00],
         [ 9.5172e-03, -8.7664e-02,  2.0811e+00],
         [ 1.1441e-02, -8.7258e-02,  1.8320e+00],
         [-9.1351e-03, -7.2469e-02,  1.5837e+00],
         [-6.5425e-03,  2.9739e-02,  1.3720e+00],
         [ 3.4676e-05,  2.3426e-01,  1.2401e+00]]], device='cuda:0')
    
    [5] # asset_cfg.body_ids
    ['Cylinder'] # asset_cfg.body_names
    """

    # compute and return position error (L2-norm)
    norm = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return norm


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    result = 1 - torch.tanh(distance / std)
    return result


def joint_pos_std(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize high standard deviation in joint positions.

    The function computes the standard deviation of the joint positions for the specified asset.
    A lower standard deviation indicates more stable joint positions.
    """

    """    
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    """

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos: torch.Tensor = asset.data.joint_pos[
        :, asset_cfg.joint_ids
    ]  # [N_env, N_joints]

    # split into even and odd
    even_grp = joint_pos[:, 0::2]  # shape [N_env, ceil(N_joints/2)]
    odd_grp = joint_pos[:, 1::2]  # shape [N_env, floor(N_joints/2)]

    # compute std for each group; keepdim=False → [N_env]
    std_even = torch.std(even_grp, dim=1)
    std_odd = torch.std(odd_grp, dim=1)

    # weighted combination
    alpha = 0.7
    combined_std = alpha * std_odd + (1.0 - alpha) * std_even

    return combined_std


def reset_deformable_body(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the deformable body to its initial state."""

    global deformable_enabled

    if not deformable_enabled:
        deformable_enabled = True

        print("Reset deformable body called.", deformable_enabled)

        for n in range(int(env.scene.num_envs)):

            prim_path = f"/World/envs/env_{n}/Robot/_FINGER_actuator_02/node_FINGER_actuator/obj1"

            # 1) deformableEnabled = False
            off_cfg = DeformableBodyPropertiesCfg(deformable_enabled=False)
            schemas.modify_deformable_body_properties(prim_path=prim_path, cfg=off_cfg)

            # 2) deformableEnabled = True
            on_cfg = DeformableBodyPropertiesCfg(deformable_enabled=True)
            schemas.modify_deformable_body_properties(prim_path=prim_path, cfg=on_cfg)
