# Point cloud extractor 
# Chanyoung / BG
# 2023. 05. 31

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.transformations import get_relative_transform
from omni.isaac.core.utils.prims import (
    get_prim_at_path,
    is_prim_non_root_articulation_link,
    is_prim_path_valid,
    find_matching_prim_paths,
    get_prim_parent,
)
from omni.isaac.core.prims import XFormPrimView
import omni.usd
from pxr import UsdGeom, PhysxSchema, Gf, Usd, UsdGeom, UsdShade
import numpy as np
import torch
# from ..models import PCN as Model

# JH model
# from omni.isaac.orbit_envs.soft.Oring_entangled.models.meta_modules import PCNNeuralProcessImplicit3DHypernet as Model
from ..models.meta_modules import PCNNeuralProcessImplicit3DHypernet as Model
# from ..models.pcn_512 import PCN  # PCN

import omni.replicator.core as rep

import omni.kit.app
import open3d as o3d

import os
import random

class PointCloudHandle():
    '''This is collision based pcd handle'''

    def __init__(self, deform_path: str):
        self.deform_path = find_matching_prim_paths(deform_path)
        self.deform_prim = []
        self.deformable_body = []
        
        self.camera_path = []

        for i, prim_path in enumerate(self.deform_path):
            self.deform_prim.append(get_prim_at_path(prim_path))
            self.camera_path.append(f"/World/envs/env_{i}/oring_env/vision_sensor")

        for deform_body in self.deform_prim:
            self.deformable_body.append(PhysxSchema.PhysxDeformableBodyAPI(deform_body))
        
        self.root = "/home/bong/.local/share/ov/pkg/isaac_sim-2022.2.1/Orbit/source/extensions/omni.isaac.orbit_envs/omni/isaac/orbit_envs/soft"
        # visualize
        self.points = []

        # latent vector extract
        # self.model = PCN().to("cuda:0")
        self.model = Model().to("cuda:0")

        # self.model.load_state_dict(torch.load(self.root + "/Oring_entangled/models/checkpoint/best_l1_cd.pth"))
        # self.model.load_state_dict(torch.load(self.root + "/Oring_entangled/models/checkpoint/checkpoint_400.pth"))
        self.model.load_state_dict(torch.load(self.root + "/Oring_entangled/models/checkpoint/model_epoch_0300.pth"))


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
            # N = 10
            point_list = np.zeros([N, 3])
            sizes = size * np.ones(N)
            stage = omni.usd.get_context().get_stage()
            point = UsdGeom.Points.Define(stage, "/World/Points/" + f"Pcd_{i}")
            point.CreatePointsAttr().Set(point_list)
            point.CreateWidthsAttr().Set(sizes)
            point.CreateDisplayColorPrimvar("constant").Set([color])
            self.points.append(point)



    # fully observable
    def get_deform_point(self, rigid_poles, render: bool = False, normalize: bool = True):
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

        if normalize:
            norm_pcds, denorm_factors = self.set_normalize(pcds, rigid_poles)  # pcd.shape -> (1 X num_envs) X (N X 3)
            raw_pcds = pcds
            return norm_pcds, denorm_factors, raw_pcds
        else:
            """only raw deform point cloud"""
            return pcds
        # return pcds

    # fully observable
    def set_camera_initailize(self):
        """Extract point clouds """
        self.camera_sensors = []
        for camera_path in self.camera_path:
            rp = rep.create.render_product(camera_path, (1024, 1024))
            pointcloud_annotator = rep.AnnotatorRegistry.get_annotator("pointcloud")

            # pointcloud_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            pointcloud_annotator.attach([rp])
            # rep.orchestrator.step()
            self.camera_sensors.append(pointcloud_annotator)

    def get_deform_partial_point(self, rigid_poles, render: bool = False, normalize: bool = True):
        pcds = []
        for i, camera_sensor in enumerate(self.camera_sensors):
            data = camera_sensor.get_data()
            pcd_data = data["data"]  
            seg_data = data["info"]["pointSemantic"]  
            partial_pcd = pcd_data[seg_data==2,:] 
            downsample_pcd = self.random_sample(partial_pcd, 512)
            pcds.append(downsample_pcd)

        if render: 
            for points in pcds:
                object_pcd = o3d.geometry.PointCloud()
                object_pcd.points = o3d.utility.Vector3dVector(points)
                o3d.visualization.draw_geometries([object_pcd])
                # point_list = np.zeros([10000, 3])
                # point_list[:len(pcds[i]),:] = pcds[i]
                # points.GetPointsAttr().Set(point_list)  # vis

        if normalize:
            norm_pcds, denorm_factors = self.set_normalize(pcds, rigid_poles)  # pcd.shape -> (1 X num_envs) X (N X 3)
            # raw_pcds = pcds
        
            return norm_pcds, denorm_factors #, raw_pcds # TODO: raw_pcds is fully obs!
        else:
            """only raw deform point cloud"""
            return pcds
        # return pcds

    def get_one_deform_point(self, num_env, render: bool = False):
        """Extract point clouds """

        local_collision_point = (np.array(self.deformable_body[num_env].GetCollisionPointsAttr().Get())) 
        vertices = np.array(local_collision_point)
        vertices_tf_row_major = np.pad(vertices, ((0, 0), (0, 1)), constant_values=1.0)
        relative_tf_column_major = get_relative_transform(self.deform_prim[num_env], get_prim_at_path("/World"))
        relative_tf_row_major = np.transpose(relative_tf_column_major)
        points_in_relative_coord = vertices_tf_row_major @ relative_tf_row_major
        pcd = points_in_relative_coord[:, :-1]
        
        if render:
            self.points[num_env].GetPointsAttr().Set(pcd)  # vis

        return pcd
    
    def set_normalize(self, pcds, rigid_poles):
        """
        Pointcloud normalize and extract scale_factor
        """
        denorm_factors = []
        norm_pcds = []

        for i, pcd in enumerate(pcds):
            mins = np.min(pcd, axis=0)
            maxs = np.max(pcd, axis=0)
            scale_factor = np.max(maxs - mins)/2
            mean = np.mean(pcd, axis=0)
            norm_pcd = (pcd - mean)/scale_factor

            # related mean 
            related_mean = torch.as_tensor(mean) - rigid_poles[i]
            denorm_factors.append(np.hstack((related_mean, scale_factor)))
            #
            norm_pcds.append(norm_pcd)
            
        return norm_pcds, np.array(denorm_factors)
    
    def set_denormalize(self, norm_pcds, denorm_factors, render: bool = True):
        """
        Check reconstruction point cloud 
        """
        denorm_pcds = []
        sizes = 0.1 * np.ones(1024)

        for i, norm_pcd in enumerate(norm_pcds):    
            denorm_pcd = norm_pcd.cpu().detach().numpy() * denorm_factors[i][-1] + denorm_factors[i][:3]
            # denorm_pcd[0] /= 2 
            denorm_pcds.append(denorm_pcd[0])

        if render:             
            point_list = np.zeros([1024, 3])
            for i, point in enumerate(self.points):
                point.CreatePointsAttr().Set(point_list)
                point.CreateWidthsAttr().Set(sizes)
                point.GetPointsAttr().Set(denorm_pcds[i])  # vis

        # return denorm_pcds

    def get_decoder_pcds(self, norm_pcds, denorm_factors, decoder: bool = False):
        gen_pcds = []
        for norm_pcd in norm_pcds:
            gen_pcd = self.model.forward(norm_pcd)#PCN
            gen_pcds.append(gen_pcd)

        if decoder:
            self.set_denormalize(gen_pcds, denorm_factors)

    def get_latent_vetors(self, norm_pcds):
        """
        get latent vector from normalized pointclouds 
        norm_pcds: [num_envs, normalize pointcloud]
        latent_vectors: [num_envs, 256]
        
        """
        # latent_vectors = []
        # for norm_pcd in norm_pcds:
        #     # latent_vector = self.model.get_latent(norm_pcd) # 1, 256 # PCN
        #     latent_vector = self.model.encode(norm_pcd.unsqueeze())
        #     latent_vectors.append(latent_vector)
        # return torch.cat(latent_vectors, 0)
    
        # x, y, z stack num_envs*N*3 

        latent_vectors = self.model.encode(torch.as_tensor(norm_pcds)) # need to check
        return latent_vectors
    
    def set_off_render(self):
        """
        off point cloud render 
        """
        for points in self.points:
            pnt = points.GetPointsAttr().Get()
            pnt *= 0
            points.GetPointsAttr().Set(pnt) 

    # -- shaped reward 
    def get_chamfer_distance(self, raw_pcds, target_pcds):
        """
        Check distance(chamfer distanse) between object node and target_node. 
        Args:

        INPUT
            object_node [num_envs*N*3](npy, np.array): 
            target_nodes [num_envs*N*3](npy, np.array): 
            
        OUTPUT
            chamfer_dist [num_envs*1](npy, np.array): each node's chamfer distances.    
        """
        chamfer_dists = []
        for i, raw_pcd in enumerate(raw_pcds):    
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(raw_pcd)
            
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_pcds[i])
            # o3d.visualization.draw_geometries([object_pcd, target_pcd])

            cham_dist = np.asarray(object_pcd.compute_point_cloud_distance(target_pcd)).sum() # chamfer distance
            chamfer_dists.append(cham_dist)

        return chamfer_dists
    
    
    def save_pcds(self, norm_pcds):
        current_dir = os.getcwd()  # 현재 작업 디렉토리를 가져옴
        rand = [random.randint(1, 1000) for _ in range(5)]
        for i, norm_pcd in enumerate(norm_pcds):
            file_path = os.path.join(current_dir+"/pcds", f"normalize_pcd_{rand[i]}.xyz")  # 현재 작업 디렉토리와 파일 이름을 결합하여 파일 경로 생성

            with open(file_path, "w") as f:
                for row in norm_pcd:
                    f.write(" ".join(str(num) for num in row))
                    f.write("\n")


    def random_sample(self, pc, n):
            idx = np.random.permutation(pc.shape[0])
            if idx.shape[0] < n:
                idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
            return pc[idx[:n]]