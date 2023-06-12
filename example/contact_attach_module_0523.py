
###
#  gripper attach module for ring object RL env
# 23.05.23
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
from pxr import UsdGeom, Usd, Gf, PhysicsSchemaTools, Sdf, PhysxSchema

# from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.articulations.articulation_view import ArticulationView
from omni.isaac.core.prims import GeometryPrimView
from omni.isaac.core.utils.stage import open_stage, add_reference_to_stage, get_current_stage
from omni.physx import get_physx_scene_query_interface 

from omni.isaac.core.utils.prims import create_prim, delete_prim, get_prim_at_path
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

           
    def get_gripper(self):
        # gripper_usd = self._path + "/IITP_ws/Deftouchnet/CoRL/asset" + "/franka_gripperX2.usd"
        gripper_usd = self._path +"/example/asset" + "/franka_gripper_fix.usd"
        add_reference_to_stage(gripper_usd, "/World/franka_gripper")
        
        init_positions = torch.zeros((self.num_envs, 3))        
        init_positions[:, 0] = -0.51132
        init_positions[:, 1] = 0.04584
        init_positions[:, 2] = 0.7
        
        self.gripper = ArticulationView(
            prim_paths_expr="/World/franka_gripper",
            translations=init_positions, 
            name="gripper")

    def initalize(self):
        """
        Need to use world.reset()
        """
        self.gripper.initialize()
        

    def main(self):
        object_usd_path = self._path +"/example/asset" + "/oring_task_env_01_05.usd"
        open_stage(usd_path=object_usd_path)
        
        self.deform_path = "/World/oring_01_05/oring"
        world = World(stage_units_in_meters=1)
        self.world = world
        self.init_simulation()
        
        self.get_gripper()
        
        world.reset()
        self.initalize()
    
        self.joint_z = self.gripper.get_dof_index("joint_z")
        self.rev_z = self.gripper.get_dof_index("rev_z")
        
        touch_prim = '/World/franka_gripper/panda_hand/geometry/trigger'
        attach_prim = '/World/franka_gripper/panda_hand/geometry/attach'
        
        # random_angle = 3.14
        TOUCH = False
        touch = False
        while simulation_app.is_running():
            if world.is_playing():
                if world.current_time_step_index == 0:
                    world.reset()
            world.step(render=True) 
            # path_tuple = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(self.deform_path))
            path_tuple = PhysicsSchemaTools.encodeSdfPath(self.deform_path)
            
            self.hit_prim = []
            num_Hit = get_physx_scene_query_interface().overlap_mesh(path_tuple[0], path_tuple[1], self.report_hit, False)
            # print("num_Hits",num_Hit)
            
            if touch_prim in self.hit_prim:
                TOUCH = True
                
            if TOUCH:
                if touch != True:
                    world.pause()
                    self.set_attach(rigid=attach_prim, deform=self.deform_path)
                    world.step()
                    world.play()
                    touch = True
                else:
                    a = self.gripper.get_joint_positions()
                    a[:, self.joint_z] += 0.1
                    a[:, self.rev_z] = 0
                    self.gripper.set_joint_position_targets(positions=a)
            else:
                a = self.gripper.get_joint_positions()
                a[:, self.joint_z] -= 0.1
                self.gripper.set_joint_position_targets(positions=a)



    def set_attach(self, rigid, deform):
        """
        define deformable, rigid_randomization
        """
        rigid_prim = get_prim_at_path(rigid)
        deform_path = UsdGeom.Mesh.Get(get_current_stage(), deform)
        attachment_path = deform_path.GetPath().AppendElementString("attachment")
        attachment = PhysxSchema.PhysxPhysicsAttachment.Define(
            get_current_stage(), attachment_path
        )
        attachment.GetActor0Rel().SetTargets([deform_path.GetPath()])
        attachment.GetActor1Rel().SetTargets([rigid_prim.GetPath()])
        PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
            
    def report_hit(self, hit):
        self.hit_prim.append(hit.collision)
        return True

if __name__ == "__main__":
    try:
        test = Test()
        test.main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
