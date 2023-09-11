
###
#  gripper attach module for ring object RL env
#  check terminal rwd! 
# 23.05.23
###
import numpy as np

from pxr import UsdGeom, Usd, Gf, PhysicsSchemaTools, Sdf, PhysxSchema

from omni.isaac.core.articulations.articulation_view import ArticulationView
from omni.isaac.core.prims import GeometryPrimView
from omni.isaac.core.utils.stage import open_stage, add_reference_to_stage, get_current_stage
from omni.physx import get_physx_scene_query_interface 
from omni.isaac.core.utils.prims import create_prim, delete_prim, get_prim_at_path
from omni.isaac.core.utils.transformations import get_relative_transform, tf_matrix_from_pose, pose_from_tf_matrix

import math
import trimesh as t
import open3d as o3d
import torch

class RobotUtils():
    def __init__(self, num_envs, grip_tip):
        self.num_envs = num_envs
        self.grip_tip = grip_tip
        self.hit_prim = []

        self.deform_path = []
        self.trigger_path = []
        self.collision_path = []
        # self.collision_prim = []
        self.attach_path = []
        self.twist_trig_path = []
        self.pole_trig_path = []

        for i in range(self.num_envs):
            self.deform_path.append(f"/World/envs/env_{i}/oring_env/oring_01_05/oring")
            self.twist_trig_path.append(f"/World/envs/env_{i}/oring_env/twist_check/trigger")
            self.pole_trig_path.append(f"/World/envs/env_{i}/rigid_pole/invisible/trigger")

            self.trigger_path.append(f"/World/envs/env_{i}/Robot/panda_hand/geometry/trigger")
            self.collision_path.append(f"/World/envs/env_{i}/Robot/panda_hand/geometry/albano")
            self.attach_path.append(f"/World/envs/env_{i}/Robot/panda_hand/geometry/attach")

    def get_collision_check(self, sim):
        """get collision check between oring and trigger and set grasp"""
        self.coll_check = []
        for i, _do_path in enumerate(self.deform_path):
            self.hit_prim = []
            path_tuple = PhysicsSchemaTools.encodeSdfPath(_do_path)
            get_physx_scene_query_interface().overlap_mesh(path_tuple[0], path_tuple[1], self.report_hit, False)
            if self.trigger_path[i] in self.hit_prim:
                # print("check_contact",self.trigger_path[i])
                self.coll_check.append(True)
                if not self.grip_tip.get_collision(indices=[i]):
                    # print("Touch")
                    # sim.pause()
                    self.grip_tip.enable_collision(indices=[i])
                    sim.step()
                    # sim.play()
            else: 
                self.coll_check.append(False)
        # print(self.hit_prim)
        return self.coll_check

    def set_manual_grip(self):
        """
        manual grasp
        """
        for i, _ in enumerate(self.deform_path):
            self.grip_tip.enable_collision(indices=[i])
    
    def set_manual_grip2(self):
        for i, _ in enumerate(self.deform_path):
            self.grip_tip.disable_collision(indices=[i])

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
        # attachment.GetActor0Rel().SetTargets([rigid_prim.GetPath()])
        # attachment.GetActor1Rel().SetTargets([deform_path.GetPath()])

        PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim()) 

    def report_hit(self, hit):
        self.hit_prim.append(hit.collision)
        return True
    
    def get_oring_final_check(self, deform_points):
        """get collision check between oring and trigger and set grasp"""
        self.coll_check = []
        self.oring_no_fall = []
        self.pole_check = []
        for i in range(self.num_envs):
              
            twist_trig_prim = get_current_stage().GetPrimAtPath(self.twist_trig_path[i])
            twist_trig_mesh = UsdGeom.Mesh(twist_trig_prim)
            twist_trig_local_collision_point = (np.array(twist_trig_mesh.GetPointsAttr().Get()))
            
            vertices = np.array(twist_trig_local_collision_point)
            vertices_tf_row_major = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.0)
            relative_tf_column_major = get_relative_transform(get_prim_at_path(self.twist_trig_path[i]), 
                                                            get_prim_at_path("/World/envs/env_{}".format(i)))
            relative_tf_row_major = np.transpose(relative_tf_column_major)

            points_in_relative_coord = vertices_tf_row_major @ relative_tf_row_major
            points_in_meters = points_in_relative_coord[:, :-1]


            # make bounding box
            twist_trig_pc = t.PointCloud(points_in_meters)
            twist_trig_boundingbox = twist_trig_pc.bounding_box


            goal_trig_prim = get_current_stage().GetPrimAtPath(self.pole_trig_path[i])
            goal_trig_mesh = UsdGeom.Mesh(goal_trig_prim)
            goal_trig_local_collision_point = (np.array(goal_trig_mesh.GetPointsAttr().Get()))
            
            vertices = np.array(goal_trig_local_collision_point)
            vertices_tf_row_major = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.0)
            relative_tf_column_major = get_relative_transform(get_prim_at_path(self.pole_trig_path[i]), 
                                                            get_prim_at_path("/World/envs/env_{}".format(i)))
            relative_tf_row_major = np.transpose(relative_tf_column_major)

            points_in_relative_coord = vertices_tf_row_major @ relative_tf_row_major
            points_in_meters = points_in_relative_coord[:, :-1]


            # make bounding box
            goal_trig_pc = t.PointCloud(points_in_meters)
            goal_trig_boundingbox = goal_trig_pc.bounding_box
            
            contact_check = not any(item == True for item in twist_trig_boundingbox.contains(deform_points[i]))
            self.coll_check.append(torch.as_tensor(contact_check))

            pole_check = any(item == True for item in goal_trig_boundingbox.contains(deform_points[i]))
            self.pole_check.append(torch.as_tensor(pole_check))
            
            # check oring center in trigger 
            raw_pcds_array = np.stack(deform_points[i])  # Convert list to numpy array
            # raw_pcds_tensor = torch.from_numpy(raw_pcds_array)  # Convert numpy array to tensor
            oring_mean_point = np.mean(raw_pcds_array, axis=0) 

            oring_inner_check = twist_trig_boundingbox.contains([oring_mean_point])
            self.oring_no_fall.append(torch.as_tensor(oring_inner_check))

        return torch.as_tensor(self.coll_check, dtype=float), torch.as_tensor(self.pole_check, dtype=float), torch.as_tensor(self.oring_no_fall, dtype=float).reshape(self.num_envs)