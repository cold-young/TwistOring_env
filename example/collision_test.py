
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
# from omni.physx import get_physx_scene_query_interface
# from omni.isaac.core.utils.prims import create_prim, delete_prim
from omni.isaac.core.utils.prims import create_prim, delete_prim, get_prim_at_path
# from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.isaac.core.utils.transformations import get_relative_transform
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
        self.deform_path = "/World/sexy_deform/deform"
        self.deform_root = "/World/sexy_deform"

        self.rigid_path = "/World/sexy_rigid/rigid"
        self.rigid_root = "/World/sexy_rigid"

    def visualizer_setup(self, points_path="/World/Points", color=(1, 0, 0), size=0.1):
        N, _ = np.array(self.deformable_body.GetCollisionPointsAttr().Get()).shape

        point_list = np.zeros([N, 3])
        sizes = size * np.ones(N)
        stage = omni.usd.get_context().get_stage()
        self.points = UsdGeom.Points.Define(stage, points_path)
        self.points.CreatePointsAttr().Set(point_list)
        self.points.CreateWidthsAttr().Set(sizes)
        self.points.CreateDisplayColorPrimvar("constant").Set([color])
        # return points
        color2 = (0, 1, 0)
        N2, _ = np.array(self.rigid_body.GetPointsAttr().Get()).shape
        point_list = np.zeros([N2, 3])
        sizes = size * np.ones(N2)
        stage = omni.usd.get_context().get_stage()
        self.points2 = UsdGeom.Points.Define(stage, "/World/Points2")
        self.points2.CreatePointsAttr().Set(point_list)
        self.points2.CreateWidthsAttr().Set(sizes)
        self.points2.CreateDisplayColorPrimvar("constant").Set([color2])


    def get_deform_point(self):
        deformable_prim = get_current_stage().GetPrimAtPath(self.deform_path)
        self.deformable_body = PhysxSchema.PhysxDeformableBodyAPI(deformable_prim)
        local_collision_point = (np.array(self.deformable_body.GetCollisionPointsAttr().Get()))# @ self.ORIENT.T
        # self.points.GetPointsAttr().Set((self.point_cloud_position.numpy()))  # vis
        vertices = np.array(local_collision_point)
        vertices_tf_row_major = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.0)
        # Transformation matrix from the coordinate system of the mesh to the coordinate system of the prim
        relative_tf_column_major = get_relative_transform(get_prim_at_path(self.deform_path), 
                                                          get_prim_at_path(self.deform_root))
        relative_tf_row_major = np.transpose(relative_tf_column_major)
        # Transform points so they are in the coordinate system of the top-level ancestral xform prim
        points_in_relative_coord = vertices_tf_row_major @ relative_tf_row_major
        points_in_meters = points_in_relative_coord[:, :-1]

        return points_in_meters
    
    def get_rigid_points(self):
        # scale = self.sibal.get_world_scales()
        # SCALE = [[scale[0,0],0,0],[0, scale[0,1],0],[0,0,scale[0,2]]]
        # print("sibal", SCALE)
        rigid_prim = get_current_stage().GetPrimAtPath(self.rigid_path)
        # self.rigid_body = PhysxSchema.PhysxRigidBodyAPI(rigid_prim)
        self.rigid_body = UsdGeom.Mesh(rigid_prim)
        local_collision_point = (np.array(self.rigid_body.GetPointsAttr().Get()))# @ self.ORIENT.T
        # self.points.GetPointsAttr().Set((self.point_cloud_position.numpy()))  # vis
        vertices = np.array(local_collision_point)
        vertices_tf_row_major = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.0)
        # Transformation matrix from the coordinate system of the mesh to the coordinate system of the prim
        relative_tf_column_major = get_relative_transform(get_prim_at_path(self.rigid_path), 
                                                          get_prim_at_path(self.rigid_root))
        relative_tf_row_major = np.transpose(relative_tf_column_major)
        # Transform points so they are in the coordinate system of the top-level ancestral xform prim
        points_in_relative_coord = vertices_tf_row_major @ relative_tf_row_major
        points_in_meters = points_in_relative_coord[:, :-1]
        return points_in_meters #@ SCALE
    
    def sibal_point_update(self):
        points = self.get_deform_point()
        self.points.GetPointsAttr().Set(points)



        return points
    
    def main(self):
        object_usd_path = self._path + "/example/asset" + "/test_collision.usd"
        open_stage(usd_path=object_usd_path)
        
        world = World(stage_units_in_meters=1)
        self.world = world

        world.reset()
        self.sibal = GeometryPrimView(
            prim_paths_expr="/World/sexy_rigid/rigid",
        )
        self.get_deform_point()
        self.get_rigid_points()

        self.visualizer_setup() 
        # sibal_points_update(points)

        i = 0
        while simulation_app.is_running():
            if world.is_playing():
                if world.current_time_step_index == 0:
                    world.reset()
            world.step(render=True) 
            # world.pause()
            points = self.sibal_point_update()
            
            rigid_points = self.get_rigid_points()
            rigid = t.PointCloud(rigid_points)
            a = rigid.bounding_box
            bounding_rigid_points = np.array(a.vertices)

            self.points2.GetPointsAttr().Set(bounding_rigid_points)
            i += 1 

            if i == 500:
                i = 0 
                pos = torch.zeros((1, 3)) 
                ori = [0,0,0]

                pos[:, 0] = random.uniform(-1, 1)
                pos[:, 1] = random.uniform(-1, 1)    
                # pos[:, 2] = random.uniform(-1, 1)    

                ori[0] = random.randint(0, 180)
                ori[1] = random.randint(0, 180)
                ori[2] = random.randint(0, 10)
                ori = torch.as_tensor([euler_angles_to_quat(euler_angles = ori, degrees=True)])

                self.sibal.set_world_poses(positions=pos, orientations=ori)
            # from IPython import embed; embed(); exit()

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)
            # o3d.visualization.draw_geometries([a.as_open3d, pcd])

            print("contact sexy!", any(item == True for item in a.contains(points)))
          
            # if want to check bounding box in simulation
            # simulation_points = np.array(a.bounding_box.vertices)

if __name__ == "__main__":
    try:
        test = Test()
        test.main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
