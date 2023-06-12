# Point cloud extractor 
# Chanyoung / BG
# 2023. 06. 01 

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.transformations import get_relative_transform
from omni.isaac.core.utils.prims import (
    get_prim_at_path,
    is_prim_non_root_articulation_link,
    is_prim_path_valid,
    find_matching_prim_paths,
    get_prim_parent,
)
# from omni.isaac.core.prims import XFormPrimView
import omni.usd
from pxr import UsdGeom, PhysxSchema
import numpy as np
import torch
 
class PointCloudHandle():
    '''This is collision based pcd handle'''

    def __init__(self, 
                 deform_path: str):
        self.deform_path = find_matching_prim_paths(deform_path)
        self.deform_prim = []
        self.deformable_body = []
        for prim_path in self.deform_path:
            self.deform_prim.append(get_prim_at_path(prim_path))
        for deform_body in self.deform_prim:
            self.deformable_body.append(PhysxSchema.PhysxDeformableBodyAPI(deform_body))
        
        # visualize
        self.points = []
 
    def initialize(self):
        self.deform_prim = []
        self.deformable_body = []
        for prim_path in self.deform_path:
            self.deform_prim.append(get_prim_at_path(prim_path))
        for deform_body in self.deform_prim:
            self.deformable_body.append(PhysxSchema.PhysxDeformableBodyAPI(deform_body))
        
    def visualizer_setup(self, color=(1, 0, 0), size=0.1):
        for i, deform_body in enumerate(self.deformable_body):
            N, _ = np.array(deform_body.GetCollisionPointsAttr().Get()).shape
            point_list = np.zeros([N, 3])
            sizes = size * np.ones(N)
            stage = omni.usd.get_context().get_stage()
            point = UsdGeom.Points.Define(stage, "/World/Points/" + f"Pcd_{i}")
            point.CreatePointsAttr().Set(point_list)
            point.CreateWidthsAttr().Set(sizes)
            point.CreateDisplayColorPrimvar("constant").Set([color])
            self.points.append(point)

    def get_deform_point(self, render: bool = False):
        """Extract point clouds """
        pcds = []
        for i, deform_body in enumerate(self.deformable_body):
            local_collision_point = (np.array(deform_body.GetCollisionPointsAttr().Get())) 
            vertices = np.array(local_collision_point)
            vertices_tf_row_major = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.0)
            relative_tf_column_major = get_relative_transform(self.deform_prim[i], get_prim_at_path("/World"))
            relative_tf_row_major = np.transpose(relative_tf_column_major)
            points_in_relative_coord = vertices_tf_row_major @ relative_tf_row_major
            pcd = points_in_relative_coord[:, :-1]
            pcds.append(pcd)
        
        if render: 
            for i, points in enumerate(self.points):
                points.GetPointsAttr().Set(pcds[i])  # vis

        norm_pcds, denorm_factors = self.set_normalize(pcds)
        
        return norm_pcds, denorm_factors
        # return pcds
    
    def set_normalize(self, pcds):
        """
        Pointcloud normalize and extract scale_factor
        """
        denorm_factors = []
        norm_pcds = []
        # from IPython import embed; embed(); exit()
        for pcd in pcds:
            maxs = np.max(pcd, axis=0)
            mins = np.min(pcd, axis=0)
            scale_factor = np.max(maxs - mins)
            mean = np.mean(pcd, axis=0)
            norm_pcd = (pcd - mean)/scale_factor
            denorm_factors.append(np.hstack((mean, scale_factor)))
            norm_pcds.append(norm_pcd)
        return norm_pcds, np.array(denorm_factors)
    
    def set_denormalize(self, norm_pcds, denorm_factors, render: bool = True):
        """
        Check reconstruction point cloud 
        """
        denorm_pcds = []
        sizes = 0.01 * np.ones(1024)

        for i, norm_pcd in enumerate(norm_pcds):    
            # from IPython import embed; embed(); exit()
            denorm_pcd = norm_pcd.cpu().detach().numpy() * denorm_factors[i][-1] + denorm_factors[i][:3]
            denorm_pcds.append(denorm_pcd[0])

        if render:             
            point_list = np.zeros([1024, 3])
            for i, point in enumerate(self.points):
                point.CreatePointsAttr().Set(point_list)
                point.CreateWidthsAttr().Set(sizes)
                point.GetPointsAttr().Set(denorm_pcds[i])  # vis

        # return denorm_pcds

    def get_latent_vector(self, norm_pcds, denorm_factors, decoder: bool = False):
        gen_pcds = []
        for norm_pcd in norm_pcds:
            gen_pcd = self.model.forward(norm_pcd)
            gen_pcds.append(gen_pcd)

        if decoder:
            self.set_denormalize(gen_pcds, denorm_factors)
        # else:
            # return env.model.get_latent(xyz=env.pcd.position).to("cpu")  # N X 3, array


    def set_off_render(self):
        """
        off point cloud render 
        """
        for points in self.points:
            pnt = points.GetPointsAttr().Get()
            pnt *= 0
            points.GetPointsAttr().Set(pnt) 
