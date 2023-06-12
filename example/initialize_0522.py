
###
# Initialization for ring object RL env
# 23.05.22
###

import argparse
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import os
import random
import torch

# import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World

# from omni.isaac.core.simulation_context import SimulationContext
from pxr import Gf, UsdGeom

# from orbit repo
from pxr import UsdGeom, PhysxSchema
# from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.articulations.articulation_view import ArticulationView
from omni.isaac.core.prims import GeometryPrimView
from omni.isaac.core.utils.stage import open_stage, add_reference_to_stage, get_current_stage

from omni.isaac.core.utils.prims import create_prim, delete_prim
from omni.isaac.core.physics_context.physics_context import PhysicsContext
import math

class Test():
    def __init__(self):
        self._device = "cuda:0"
        self._path = os.getcwd()
        self.num_envs = 1

    def init_simulation(self):
        self._scene = PhysicsContext()
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)
        self._scene.set_friction_offset_threshold(0.01)
        self._scene.set_friction_correlation_distance(0.005)
        self._scene.enable_ccd(flag=False)

    def get_hook(self):

        self.hook = ArticulationView(
            prim_paths_expr="/World/pole_env/init_hook", 
            name="hook")
    
    def get_rigid_pole(self):
        rigid_usd = self._path + "/example/asset" + "/rigid_pole.usd"
        add_reference_to_stage(rigid_usd, "/World/rigid_pole")
        
        init_positions = torch.zeros((self.num_envs, 3))        
        # init_positions[:, 1] = random.uniform(0.8, 1.2)
        init_positions[:, 1] = random.uniform(1.2, 1.8)
        
        # init_positions[:, 0] = random.uniform(-0.3, 0.3)
        
        self.init_rigid_pos = init_positions
        
        GeometryPrimView(
            prim_paths_expr="/World/rigid_pole",
            positions=init_positions,
        )
        
        self.rigid_pole = GeometryPrimView(
            prim_paths_expr="/World/rigid_pole/pole",
            visibilities=[True],
        )
        
    def initalize(self):
        """
        Need to use world.reset()
        """
        self.hook.initialize()
        self.joint_x = self.hook.get_dof_index("pris_x")
        self.joint_y = self.hook.get_dof_index("pris_y")
        self.joint_z = self.hook.get_dof_index("pris_z")
        self.rev_x = self.hook.get_dof_index("rev_x")
        

    def main(self):
        object_usd_path = self._path + "/example/asset" + "/oring_task_env_01_05_default.usd"
        open_stage(usd_path=object_usd_path)
        
        world = World(stage_units_in_meters=1)
        self.world = world
        self.init_simulation()
        
        self.get_hook()
        self.get_rigid_pole()
        world.reset()
        self.initalize()
        
        # random_angle = 3.14
        i = 0
        while simulation_app.is_running():
            if world.is_playing():
                if world.current_time_step_index == 0:
                    world.reset()
            world.step(render=True) 
            i+=1
            
            if i == 10: # randomize reset (max_episode_length)
                self.get_pre_reset()
                self.get_reset()
                i = 0
                

    def get_pre_reset(self):
        delete_prim("/World/rigid_pole")
        delete_prim("/World/pole_env")
        delete_prim("/World/oring_01_05")
        
        object_usd_path = self._path + "/example/asset"  + "/oring_task_env_01_05_default.usd"
        open_stage(usd_path=object_usd_path)
        
        self.get_hook()
        self.get_rigid_pole()
        self.world.reset()
        self.initalize()
        
    
    def get_reset(self):
        # self.rigid_pole.enable_collision()
        random_angle_list = [-6.0, -3.14, 3.14, 6.0]
        INIT_ANG = random.choice(random_angle_list)        
        # print("reset angle :", INIT_ANG)
        
        while True:
            self.world.step(render=True)            
            
            hook_joint_pos = self.hook.get_joint_positions()
            
            # hook move pris_x target            
            if hook_joint_pos[:, self.joint_x] < float(self.init_rigid_pos[:, 1]) - 0.2: 
                hook_joint_pos[:, self.joint_x] += 0.01
                                            
                self.hook.set_joint_positions(positions=hook_joint_pos)
            
            # hook rotation
            else: 
                if INIT_ANG < 0: #-3.14, -6.28
                    hook_joint_pos[:, self.joint_y] = 0
                    hook_joint_pos[:, self.joint_z] = 0
                    
                    if hook_joint_pos[:, self.rev_x] > INIT_ANG:
                        hook_joint_pos[:, self.rev_x] -= 0.5
                        if hook_joint_pos[:, self.rev_x] < -6.28: # -2pi
                            hook_joint_pos[:, self.rev_x] = INIT_ANG - 0.1
                    else:
                        self.rigid_pole.enable_collision()
                        break
                    
                else: # 3.14, 6.28
                    hook_joint_pos[:, self.joint_y] = 0
                    hook_joint_pos[:, self.joint_z] = 0
                    
                    if hook_joint_pos[:, self.rev_x] < INIT_ANG:
                        hook_joint_pos[:, self.rev_x] += 0.5
                        if hook_joint_pos[:, self.rev_x] > 6.28: # 2pi
                            hook_joint_pos[:, self.rev_x] = INIT_ANG + 0.1
                    else:
                        self.rigid_pole.enable_collision()
                        break
                self.hook.set_joint_position_targets(positions=hook_joint_pos)
    
        delete_prim("/World/pole_env/init_hook")
        

                
                
                
            
if __name__ == "__main__":
    try:
        test = Test()
        test.main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
