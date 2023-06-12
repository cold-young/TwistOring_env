
###
# Check collision!!!!!
# 23.05.26
###

import argparse
from omni.isaac.kit import SimulationApp
import trimesh as t
simulation_app = SimulationApp({"headless": False})

import numpy as np
import os
from pxr import UsdGeom, Usd, Gf, PhysicsSchemaTools, Sdf, PhysxSchema
# import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World
# from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.stage import open_stage, add_reference_to_stage, get_current_stage
from omni.isaac.core.articulations.articulation_view import ArticulationView
from omni.isaac.core.utils.prims import create_prim, delete_prim, get_prim_at_path
# from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.isaac.core.utils.transformations import get_relative_transform, tf_matrix_from_pose
# import math
from omni.isaac.core.prims import GeometryPrimView, RigidPrimView
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_rot_matrix

# from omni.isaac.core.utils.rotations import euler_angles_to_quat
import omni.usd
import trimesh as t
import open3d as o3d
import torch
import random


class Test():
    def __init__(self):
        self._device = "cuda:0"
        self._path = os.getcwd()
        self.num_envs = 1

        self.obstacle_prims = ['/World/franka_gripper/panda_hand/geometry/panda_hand',
                               '/World/franka_gripper/panda_leftfinger/geometry/panda_leftfinger',
                               '/World/franka_gripper/panda_rightfinger/geometry/panda_rightfinger',
                               '/World/franka_gripper/panda_hand/geometry/trigger'
                               ]
        self.obstacle_roots = ['/World/franka_gripper/panda_hand/geometry',
                               '/World/franka_gripper/panda_leftfinger/geometry',
                               '/World/franka_gripper/panda_rightfinger/geometry',
                               '/World/franka_gripper/panda_hand/geometry',
                               ]

    def visualizer_setup(self, points_path="/World/Points", color=(1, 0, 0), size=0.05):
        # gripper bounding boxes
        b_color = (0, 1, 0)
        point_list = np.zeros([8, 3])
        sizes = size * np.ones(8)
        stage = omni.usd.get_context().get_stage()
        self.b_points = UsdGeom.Points.Define(stage, "/World/Bounding_0")
        self.b_points.CreatePointsAttr().Set(point_list)
        self.b_points.CreateWidthsAttr().Set(sizes)
        self.b_points.CreateDisplayColorPrimvar("constant").Set([b_color])

    def get_mesh_vertices_relative_to(self, mesh_prim, coord_prim) -> np.ndarray:
        # Vertices of the mesh in the mesh's coordinate system
        SCALE = 4
        pos, ori = self.gripper.get_world_poses()
        # from IPython import embed; embed(); exit()
        gripper_tr = np.linalg.inv(np.transpose(tf_matrix_from_pose(translation=pos[0], orientation=ori[0])))
        # franka_gripper translation, size, orientation
        rigid_prim = get_current_stage().GetPrimAtPath(mesh_prim)
        rigid_body = UsdGeom.Mesh(rigid_prim)
        local_collision_point = (np.array(rigid_body.GetPointsAttr().Get()))
        vertices = np.array(local_collision_point)
        vertices_tf_row_major = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.0)
        relative_tf_column_major = get_relative_transform(get_prim_at_path(mesh_prim),
                                                        #   get_prim_at_path('/World/franka_gripper'))
                                                          get_prim_at_path('/World'))
        
        relative_tf_row_major = np.transpose(relative_tf_column_major)
        points_in_relative_coord = vertices_tf_row_major @ relative_tf_row_major #@ gripper_tr #@ relative_tf_row_major_2
        points_in_meters = points_in_relative_coord[:, :-1]

        return points_in_meters #* SCALE


    def get_random_gripper_pose(self):
        #TODO: Add select init positions from oring collision mesh points
        pos = torch.zeros((1, 3)) 
        pos[:, 0] = random.uniform(-1, 1)
        pos[:, 1] = random.uniform(-1, 1)   
        

        ori = [0,0,0]
        ori[0] = random.randint(0, 180)
        ori[1] = random.randint(0, 180)
        ori[2] = random.randint(0, 180)
         
        random_orientations = torch.as_tensor([euler_angles_to_quat(euler_angles=ori, degrees=True)])

        return pos, random_orientations

    def main(self):
        object_usd_path = self._path + "/example/asset" + "/test_bb.usd"
        open_stage(usd_path=object_usd_path)
        
        world = World(stage_units_in_meters=1)
        self.world = world
        world.reset()
        self.gripper = ArticulationView(
            prim_paths_expr="/World/franka_gripper",name="gripper")
        world.reset()
        self.gripper.initialize()
        self.visualizer_setup()


        i = 0
        while simulation_app.is_running():
            if world.is_playing():
                if world.current_time_step_index == 0:
                    world.reset()
            world.step(render=True) 

            i += 1

            points = self.get_mesh_vertices_relative_to(
                self.obstacle_prims[1],
                self.obstacle_roots[1])

            self.b_points.GetPointsAttr().Set(points)

            if i == 500:
                print("update pcl")
                i = 0 
                pos, ori = self.get_random_gripper_pose()
                self.gripper.set_world_poses(pos, ori)
          

if __name__ == "__main__":
    try:
        test = Test()
        test.main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
