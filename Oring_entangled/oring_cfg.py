# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematicsCfg
from .robots.franka_tip import FRANKA_TIP_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulatorCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg, PhysxCfg, SimCfg, ViewerCfg

@configclass
class GoalMarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.05, 0.05, 0.05]  # x,y,z


@configclass
class FrameMarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.1, 0.1, 0.1]  # x,y,z


##
# MDP settings
##


@configclass
class RandomizationCfg:
    """Randomization of scene at reset."""

    # @configclass
    # class ObjectInitialPoseCfg:
    #     """Randomization of object initial pose."""

    #     # category
    #     position_cat: str = "default"  # randomize position: "default", "uniform"
    #     orientation_cat: str = "default"  # randomize position: "default", "uniform"
    #     # randomize position
    #     position_uniform_min = [0.4, -0.25, 0.075]  # position (x,y,z)
    #     position_uniform_max = [0.6, 0.25, 0.075]  # position (x,y,z)

    # @configclass
    # class ObjectDesiredPoseCfg:
    #     """Randomization of object desired pose."""

    #     # category
    #     position_cat: str = "default"  # randomize position: "default", "uniform"
    #     orientation_cat: str = "default"  # randomize position: "default", "uniform"
    #     # randomize position
    #     position_default = [100.0, 100.0, 100.0]  # position default (x,y,z)
    #     position_uniform_min = [0.4, -0.25, 0.25]  # position (x,y,z)
    #     position_uniform_max = [0.6, 0.25, 0.5]  # position (x,y,z)
    #     # randomize orientation
    #     orientation_default = [1.0, 0.0, 0.0, 0.0]  # orientation default

    # # initialize
    # object_initial_pose: ObjectInitialPoseCfg = ObjectInitialPoseCfg()
    # object_desired_pose: ObjectDesiredPoseCfg = ObjectDesiredPoseCfg()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg:
        """Observations for policy group."""

        # global group settings
        enable_corruption: bool = True
        # observation terms
        
        # - AGENT
        # -- agent state
        tool_positions = {"scale": 1.0}
        tool_orientations_cos = {"scale": 1.0}
        tool_orientations_sin = {"scale": 1.0}
        
        # - OBJECT
        # -- oring state

        pcn_latent = {"scale": 1.0}
        pcn_scale_factor = {"scale": 1.0}


        # - ETC
        # -- phase state
        z_move_states = {"scale": 1.0}

    
    return_dict_obs_in_group = False
    """Whether to return observations as dictionary or flattened vector within groups."""
    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # -- shaped reward
    penalizing_chamfer_dist_default_and_twist = {"weight": -0.01} # scale 100 ~ 1000
    # distance_EE_gole_position = {"weight": 0.1} # need to fix

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    episode_timeout = True  # reset when episode length ended
    # object_falling = False  # reset when object falls off the table
    # is_success = False  # reset when object is lifted


@configclass
class ControlCfg:
    """Processing of MDP actions."""

    # action space
    control_type = "default"  # "default", "inverse_kinematics"
    # decimation: Number of control action updates @ sim dt per policy dt
    decimation = 1

##
# Environment configuration
##


@configclass
class OringEnvCfg(IsaacEnvCfg):
    """Configuration for the Lift environment."""

    # General Settings
    env: EnvCfg = EnvCfg(num_envs=2, env_spacing=10, episode_length_s=5.0)
    viewer: ViewerCfg = ViewerCfg(debug_vis=True, eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0))
    # Physics settings
    sim: SimCfg = SimCfg(
        dt=1./60,
        substeps=4,
        # use_flatcache=False,
        # use_flatcache=False,
        physx=PhysxCfg(
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 64, #4,
            gpu_total_aggregate_pairs_capacity=1024*1024*16, #16 * 1024,
            # friction_correlation_distance=0.00625,
            friction_offset_threshold=0.04, #0.01,
            bounce_threshold_velocity=0.5,#=0.2,
            gpu_max_soft_body_contacts=1024*1024*8,
        ),
    )

    # Scene Settings
    robot_support: SingleArmManipulatorCfg = FRANKA_TIP_CFG

    # -- robot
    robot: SingleArmManipulatorCfg = FRANKA_TIP_CFG
    # robot: RobotCfg = RobotCfg()
    goal_marker: GoalMarkerCfg = GoalMarkerCfg()
    frame_marker: FrameMarkerCfg = FrameMarkerCfg()

    # MDP settings
    randomization: RandomizationCfg = RandomizationCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Controller settings
    control: ControlCfg = ControlCfg()
