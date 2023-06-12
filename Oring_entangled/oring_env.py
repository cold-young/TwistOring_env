# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

####################
#  2023.06.09     #
#  Oring Twist env #
####################
# manual grasp version

import gym.spaces
import math
import numpy as np
import torch
from typing import List

import random
import os
from pxr import UsdGeom, Usd, Gf, PhysicsSchemaTools, Sdf, PhysxSchema, UsdPhysics

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.articulations.articulation_view import ArticulationView
from omni.isaac.core.prims import GeometryPrimView, XFormPrimView
from omni.isaac.core.utils.stage import add_reference_to_stage 
import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.markers import StaticMarker
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit.utils.math import euler_xyz_from_quat

from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs

from .oring_cfg import OringEnvCfg 
from .utils.prim_utils import delete_prim
from .utils.pcd_utils import PointCloudHandle
from .models.meta_modules import PCNNeuralProcessImplicit3DHypernet as Model
from .robots.robot_control_util import RobotUtils

class OringEnv(IsaacEnv):
    """Environment for lifting an object off a table with a single-arm manipulator."""

    def __init__(self, cfg: OringEnvCfg = None, headless: bool = False):
        self.cfg = cfg
        self._path = os.getcwd()
        self._usd_path = self._path + "/source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/soft/Oring_entangled/usd"

        self.robot = SingleArmManipulator(cfg=self.cfg.robot)
        # PCN
        # self.model = Model(16384, 256, 4).to("cuda:0")
        # JH model
        self.model = Model().to("cuda:0")

        self.latent_num = 256
        super().__init__(self.cfg, headless=headless)

        # parse the configuration for information
        self._process_cfg()
        self._initialize_views()

        # prepare the observation manager
        self._observation_manager = ObservationManager(class_to_dict(self.cfg.observations), self, self.device)
        self._reward_manager = RewardManager(
            class_to_dict(self.cfg.rewards), self, self.num_envs, self.dt, self.device
        )
        # print information about MDP
        print("[INFO] Observation Manager:", self._observation_manager)
        print("[INFO] Reward Manager: ", self._reward_manager)

        num_obs = self._observation_manager.group_obs_dim["policy"][0]
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))

        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3]) # x, y, yaw
        print("[INFO]: Completed setting up the environment...")

        # Take an initial step to initialize the scene.
        self.sim.step()

        self.deform_path = self.template_env_ns + '/oring_env/oring_01_05/oring'
    """
    Implementation specifics.
    """

    def _design_scene(self) -> List[str]:
        # ground plane
        self.ground_path = self._usd_path + "/default_ground.usd"
        kit_utils.create_ground_plane("/World/defaultGroundPlane",
                                      z_position=-2.,
                                      usd_path=self.ground_path)
        
        # self.scene_path = self._usd_path + "/oring_task_env_01_05_with_camera.usd"  # texture 
        self.scene_path = self._usd_path + "/oring_task_env_01_05_with_camera_no_render.usd" # no texture
        for i in range(self.num_envs):

            prim_utils.create_prim(prim_path=f"/World/envs/env_{i}/oring_env",
                                usd_path=self.scene_path,
                                translation=[0., 0., 0.],
                                orientation=[1., 0., 0., 0.],
                                scale=[1., 1., 1.])

            self.robot.spawn(f"/World/envs/env_{i}/Robot")
        
        # self.grip_ctl = RobotUtils(num_envs=self.num_envs)
       
        self.pcd = PointCloudHandle(deform_path="/World/envs/.*/oring_env/oring_01_05/oring")
        self.pcd.visualizer_setup()
        self.init_rigid_pos = torch.zeros((self.num_envs, 3))
        self.init_rigid_goal = torch.zeros((self.num_envs, 3))
        self.z_phase = torch.zeros((self.num_envs, 1))

        if self.cfg.viewer.debug_vis and self.enable_render:
            # create point instancer to visualize the goal points
            self._goal_markers = StaticMarker(
                "/Visuals/object_goal",
                self.num_envs,
                usd_path=self.cfg.goal_marker.usd_path,
                scale=self.cfg.goal_marker.scale,
            )
            # create marker for viewing end-effector pose
            self._ee_markers = StaticMarker(
                "/Visuals/ee_current",
                self.num_envs,
                usd_path=self.cfg.frame_marker.usd_path,
                scale=self.cfg.frame_marker.scale,
            )
            # create marker for viewing command (if task-space controller is used)
            if self.cfg.control.control_type == "inverse_kinematics":
                self._cmd_markers = StaticMarker(
                    "/Visuals/ik_command",
                    self.num_envs,
                    usd_path=self.cfg.frame_marker.usd_path,
                    scale=self.cfg.frame_marker.scale,
            )
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: VecEnvIndices):
        self.extras["episode"] = dict()
        self._reward_manager.reset_idx(env_ids, self.extras["episode"])
        self._observation_manager.reset_idx(env_ids)
        self.previous_actions[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0

        # Initialize
        # random.uniform(1.3, 1.6)
        x_angle = [-0.12, 0.12] #random
        rand_x = torch.as_tensor([x_angle[random.randint(0, 1)] for _ in range(self.num_envs)])
        rand_y = ([random.uniform(1.14, 1.2) for _ in range(self.num_envs)])
        for i in range(self.num_envs):
            delete_prim(f"/World/envs/env_{i}/oring_env") 
            delete_prim(f"/World/envs/env_{i}/rigid_pole")
            # delete_prim(f"/World/envs/env_{i}/Robot")

            prim_utils.create_prim(prim_path=f"/World/envs/env_{i}/oring_env",
                                usd_path=self.scene_path,
                                translation=[0., 0., 0.],
                                orientation=[1., 0., 0., 0.],
                                scale=[1., 1., 1.])
            
            self.robot.spawn(f"/World/envs/env_{i}/Robot", [rand_x[i], rand_y[i], -0.2])


        # from IPython import embed; embed(); exit()        
        self.grip_tip = GeometryPrimView(prim_paths_expr="/World/envs/.*/Robot/panda_hand/geometry/albano")
        self.grip_ctl = RobotUtils(self.num_envs, self.grip_tip)

        # Deformable mover
        self.hook = ArticulationView(
            prim_paths_expr="/World/envs/.*/oring_env/pole_env/init_hook",
            name="hook")
        
        self._initialize_views()
        self.hook.initialize()
        self.joint_x = self.hook.get_dof_index("pris_x")
        self.joint_y = self.hook.get_dof_index("pris_y")
        self.joint_z = self.hook.get_dof_index("pris_z")
        self.rev_x = self.hook.get_dof_index("rev_x")

        self.sim.step(render=self.enable_render)
        self.pcd.initialize()

        self.pcd.set_camera_initailize()
        #####
        # print("start save!")
        # norm_pcds, scale_factors, self.raw_pcds = self.pcd.get_deform_point(render=False)
        # self.pcd.save_pcds(norm_pcds)

        # Randomize oring state
        self.get_rigid_pole()
        self.get_reset()

        # init grasp
        self.set_grasp()
        # Initialize pointcloud extractor
        # self.pcd.set_off_render()

    def _step_impl(self, actions: torch.Tensor):
        
        self.actions = actions.clone().to(device=self.device) # x, y, yaw
        self.actions = torch.as_tensor(self.actions, dtype=float)

        # SET ACTION
        self.actions[:, 0] = 0.5 * (self.actions[:, 0] - 1.) # x, y
        self.actions[:, 1] = 0.5 * (self.actions[:, 1] - 1.) # x, y
        self.actions[:, 2] = 3. * (self.actions[:, 2] - 1.) # yaw
        # print(self.actions)

        self.robot_actions[:, :2] = self.actions[:, :2] # x, y 
        self.robot_actions[:, 3:] = 0 # roll, pitch, yaw
        self.robot_actions[:, 5] = self.actions[:, 2] # yaw
        # self.robot_actions[:, 5] = 0

        if self.episode_length_buf[0] < 15:
            # up move
            self.z_phase = torch.ones((self.num_envs, 1))
            self.robot_actions[:, 2] = 1.3 #z

        else:
            # down move
            self.z_phase = torch.zeros((self.num_envs, 1))
            self.robot_actions[:, 2] = -0.5 #z

        # -- state update 
        robot_position = self.robot.data.ee_state_w[:, :2] - self.envs_positions[:, :2]
        target_position = self.init_rigid_pos[:, :2]
        abs_dist = torch.abs(robot_position - target_position)

        # env scale constraint
        self.robot_actions[:, 0] = torch.where(abs_dist[:, 1] > 0.5, 0, self.robot_actions[:, 0])
        self.robot_actions[:, 1] = torch.where(abs_dist[:, 0] > 0.2, 0, self.robot_actions[:, 1])
        
        for _ in range(self.cfg.control.decimation):

            self.robot.apply_action(self.robot_actions)  # robot_action

            # check touch object
            self.sim.step(render=self.enable_render)

            _, _, self.raw_pcds = self.pcd.get_deform_point(self.init_rigid_goal, render=False)

            # # shaped rwd - chamfer distance
            self.chamfer_dists = self.pcd.get_chamfer_distance(self.raw_pcds, self.target_pcds)

            # ## get latent vector
            
            # _, _, self.raw_pcds = self.pcd.get_deform_point(self.init_rigid_pos,render=False)
            # self.chamfer_dists = self.pcd.get_chamfer_distance(self.raw_pcds, self.target_pcds)

            # partial point cloud
            norm_pcds, scale_factors = self.pcd.get_deform_partial_point(rigid_poles=self.init_rigid_goal, render=False, normalize=True)
            self.latent_vectors = torch.as_tensor(self.pcd.get_latent_vetors(norm_pcds), device=self.device)
            self.scale_factors = torch.as_tensor(scale_factors, device=self.device)

            # save .xyz file
            # if self.episode_length_buf[0] % 5 == 0:
            #     self.pcd.save_pcds(norm_pcds)
            #     print("save~")
            # print("latent", self.latent_vectors)
            if self.sim.is_stopped():
                return

        self.extras["time_outs"] = self.episode_length_buf >= self.max_episode_length
        self.robot.update_buffers(self.dt)
   
        self.reward_buf = self._reward_manager.compute() 
        # print("rwd compute!")

        self._check_termination()
        
        self.previous_actions = self.actions.clone()
        
        # print(self.episode_length_buf,"/", self.max_episode_length)

        if self.cfg.viewer.debug_vis and self.enable_render:
            self._debug_vis()

    def _get_observations(self) -> VecEnvObs:
        # compute observations
        return self._observation_manager.compute()

    def _success_checker(self, prev_reward_buf):
        # -- FINAL SUCCESS REWARD
        # 1. CHECK TWIST ORING  (NOT CONTACT CENTER TRIGGER)
        # 2. HANG DEFORM OBJECT (Y>#)
        if self.episode_length_buf[0] == self.max_episode_length:
            # print("final final success rwd")
            # sibals = []
            for i in range(self.num_envs):
                delete_prim(f"/World/envs/env_{i}/Robot")
            
            for _ in range(10):
                self.sim.step()

            _, _, raw_pcds = self.pcd.get_deform_point(rigid_poles=self.init_rigid_goal ,render=False)

            sibal, no_fall = self.grip_ctl.get_oring_final_check(raw_pcds) 

            final_rwds = torch.where(torch.logical_and(sibal, no_fall), 100., 0.)
            # print(final_rwds)
            return final_rwds.reshape(self.num_envs)

        else:
            return prev_reward_buf
    """
    Helper functions - Scene handling.
    """

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # compute constants for environment
        self.dt = self.cfg.control.decimation * self.physics_dt  # control-dt
        # self.max_episode_length = math.ceil(self.cfg.env.episode_length_s / self.dt) # chanyoung
        self.max_episode_length = 50 # 140

    def _initialize_views(self) -> None:
        """Creates views and extract useful quantities from them."""
        self.sim.reset()
        # self.num_actions = 6
        self.robot.initialize(self.env_ns + "/.*/Robot")
        # self.num_actions = self.robot.num_actions - 1
        self.num_actions = 3
        # Articulation initialize
        # history
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        # robot joint actions
        self.robot_actions = torch.zeros((self.num_envs, 6), device=self.device)
        # commands

        # Observation initialize
        self.latent_vectors = torch.zeros((self.num_envs, self.latent_num), device=self.device)
        self.scale_factors = torch.zeros((self.num_envs, 4), device=self.device)

    def _debug_vis(self):
        """Visualize the environment in debug mode."""
        self._ee_markers.set_world_poses(self.robot.data.ee_state_w[:, 0:3], self.robot.data.ee_state_w[:, 3:7])

    """
    Helper functions - MDP.
    """
    def _check_termination(self) -> None:
        self.reset_buf[:] = 0
        # -- episode length
        if self.cfg.terminations.episode_timeout:
            self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)



    """
    Asset fuctions
    """
    def get_rigid_pole(self):
        """
        Randomize rigid pole
        """
        rigid_path = self._usd_path + "/rigid_pole.usd"

        for i in range(self.num_envs):
            add_reference_to_stage(rigid_path, f"/World/envs/env_{i}/rigid_pole")

        init_positions = torch.zeros((self.num_envs, 3))
        for i in range(self.num_envs):
            init_positions[i, 1] = random.uniform(1.3, 1.6)
        
        # move z point more up
        self.init_rigid_pos = init_positions
        self.init_rigid_goal = init_positions.clone().detach().cpu().numpy()
        self.init_rigid_goal[:, 2] += 0.3
        GeometryPrimView(
            prim_paths_expr="/World/envs/.*/rigid_pole",
            translations=init_positions,
            
        )
        visible = torch.ones((self.num_envs, 1), dtype=torch.bool)
        self.rigid_pole = GeometryPrimView(
            prim_paths_expr="/World/envs/.*/rigid_pole/pole",
            visibilities=visible,
        )

    def get_reset(self):
        """
        Reset / oring twist
        """
        # self.rigid_pole.enable_collision()
        random_angle_list = [-6.0, -3.14, 3.14, 6.0]
        # random_angle_list = [0.1, 0.1, 0.1, 0.1] # check no twist
        
        # random_angle_list = [-3.14, -6.0]

        INIT_ANG_LIST = torch.as_tensor([random_angle_list[random.randint(0, 3)] for _ in range(self.num_envs)])
        _state = torch.zeros((self.num_envs))
        _state_2 = torch.where(INIT_ANG_LIST < 0, False, True)
        _break = torch.zeros((self.num_envs))

        self.target_pcds = [[] for _ in range(self.num_envs)]
        while True:        
            self.sim.step()
            hook_joint_pos = self.hook.get_joint_positions()
            
            if all(_break):
                break
            if not all(_state):
                for i in range(self.num_envs):
                    hook_joint_pos[i, self.joint_y] = 0
                    hook_joint_pos[i, self.joint_z] = 0

                    if hook_joint_pos[i, self.joint_x] < float(self.init_rigid_pos[i, 1]) - 0.2:
                        # hook_joint_pos[i, self.joint_x] += 0.01
                        hook_joint_pos[i, self.joint_x] += 0.03

                        # get_chamfer distance (target pcd)
                        if (float(self.init_rigid_pos[i, 1]) - 0.25) > hook_joint_pos[i, self.joint_x] > (float(self.init_rigid_pos[i, 1]) - 0.3):
                            raw_pcd = self.pcd.get_one_deform_point(num_env=i, render=False) # render=True
                            self.target_pcds[i] = raw_pcd

                    else:
                        _state[i] = True
                        # self.sim.step()
                self.hook.set_joint_positions(positions=hook_joint_pos)

            else:
                for i in range(self.num_envs):
                    if _state_2[i] == False:
                        hook_joint_pos[i, self.joint_y] = 0
                        hook_joint_pos[i, self.joint_z] = 0
                        
                        if hook_joint_pos[i, self.rev_x] > INIT_ANG_LIST[i]:
                            hook_joint_pos[i, self.rev_x] -= 0.3
                            if hook_joint_pos[i, self.rev_x] < -6.28: # -2pi
                                hook_joint_pos[i, self.rev_x] = INIT_ANG_LIST[i] - 0.1
                        else:
                            self.rigid_pole.enable_collision(indices=[i])
                            _break[i] = True

                    elif _state_2[i] == True: # 3.14, 6.28
                        hook_joint_pos[i, self.joint_y] = 0
                        hook_joint_pos[i, self.joint_z] = 0
                        
                        if hook_joint_pos[i, self.rev_x] < INIT_ANG_LIST[i]:
                            hook_joint_pos[i, self.rev_x] += 0.3
                            if hook_joint_pos[i, self.rev_x] > 6.28: # 2pi
                                hook_joint_pos[i, self.rev_x] = INIT_ANG_LIST[i]+ 0.1
                        else:
                            self.rigid_pole.enable_collision(indices=[i])
                            _break[i] = True
                    self.hook.set_joint_position_targets(positions=hook_joint_pos)

        for i in range(self.num_envs):
            delete_prim(f"/World/envs/env_{i}/oring_env/pole_env/init_hook")
    
    def set_grasp(self):
        """
        manual grasp action (during reset)
        """
        self.robot_actions *= 0
        for i in range(12):
            # self.grip_ctl.get_collision_check(self.sim)

            # self.robot_actions[:, 2] += -0.025

            self.robot_actions[:, 2] = -1.
            self.robot.apply_action(self.robot_actions)  # robot_action
            self.sim.step()
            if i == 11:
                self.grip_ctl.set_manual_grip()

class ObservationManager(ObservationManager):
    # Shape: [num_envs, #]

    # - AGENT
    # -- agent state
    def tool_positions(self, env: OringEnv):
        """Current end-effector position of the arm. + relate rigid pole"""
        ee_position = env.robot.data.ee_state_w[:, :3] - env.envs_positions 
        return ee_position - env.init_rigid_pos # relative postion 

    def tool_orientations_cos(self, env: OringEnv):
        """Current end-effector orientation of the arm."""
        # make the first element positive
        quat_w = env.robot.data.ee_state_w[:, 3:7]
        quat_w[quat_w[:, 0] < 0] *= -1
        _,_, yaw = euler_xyz_from_quat(quat_w)
        cos_yaw= np.cos(yaw)
        return cos_yaw.reshape(env.num_envs, 1)

    def tool_orientations_sin(self, env: OringEnv):
        """Current end-effector orientation of the arm."""
        # make the first element positive
        quat_w = env.robot.data.ee_state_w[:, 3:7]
        quat_w[quat_w[:, 0] < 0] *= -1
        _,_, yaw = euler_xyz_from_quat(quat_w)
        sin_yaw= np.cos(yaw)
        return sin_yaw.reshape(env.num_envs, 1)

    # - OBJECT
    # --- oring state
    def pcn_latent(self, env: OringEnv):
        """Oring latent vectors"""
        return env.latent_vectors

    def pcn_scale_factor(self, env: OringEnv):
        """Oring pose info. factor"""
        return env.scale_factors
    
    # - ETC
    # -- phase state
    def z_move_states(self, env: OringEnv):
        """ Get z move up[1]/down[0]"""
        return env.z_phase
    
class RewardManager(RewardManager):
    """Reward manager for single-arm object lifting environment."""
    
    # -- SHAPED REWARD
    # 1. CHAMPER DISTANCE BTW DEFAULT ORING AND CURRENT ORING 
    def penalizing_chamfer_dist_default_and_twist(self, env: OringEnv):
        return torch.as_tensor(env.chamfer_dists)