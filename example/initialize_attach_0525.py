
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

import omni.usd

# import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core import World

# from omni.isaac.core.simulation_context import SimulationContext
# from pxr import Gf, UsdGeom

# from orbit repo
# from pxr import UsdGeom, PhysxSchema
from pxr import UsdGeom, Usd, Gf, PhysicsSchemaTools, Sdf, PhysxSchema

# from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.articulations.articulation_view import ArticulationView
from omni.isaac.core.prims import GeometryPrimView
from omni.isaac.core.utils.stage import open_stage, add_reference_to_stage, get_current_stage
from omni.physx import get_physx_scene_query_interface
# from omni.isaac.core.utils.prims import create_prim, delete_prim
from omni.isaac.core.utils.prims import create_prim, delete_prim, get_prim_at_path
from omni.isaac.core.physics_context.physics_context import PhysicsContext
from omni.isaac.core.utils.transformations import get_relative_transform
import math
from omni.isaac.core.utils.rotations import euler_angles_to_quat


class Test():
    def __init__(self):
        self._device = "cuda:0"
        self._path = os.getcwd()
        self._asset_path = self._path + "/example/asset" ###
        self.max_episode_length = 100 ###
        self.ORIENT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def init_simulation(self):
        self._scene = PhysicsContext()
        self._scene.set_broadphase_type("GPU")
        self._scene.enable_gpu_dynamics(flag=True)
        self._scene.set_friction_offset_threshold(0.01)
        self._scene.set_friction_correlation_distance(0.005)
        self._scene.enable_ccd(flag=False)

    def define_paths(self):
        self.deform_path = "/World/oring_01_05/oring"
        self.deform_root = "/World/oring_01_05"

        self.touch_prim = '/World/franka_gripper/panda_hand/geometry/trigger'
        self.attach_prim = '/World/franka_gripper/panda_hand/geometry/attach'
        self.obstacle_prims = ['/World/franka_gripper/panda_hand/geometry/panda_hand',
                               '/World/franka_gripper/panda_leftfinger/geometry/panda_leftfinger',
                               '/World/franka_gripper/panda_rightfinger/geometry/panda_rightfinger',
                               ]

    def get_hook(self):
        self.hook = ArticulationView(
            prim_paths_expr="/World/pole_env/init_hook", 
            name="hook")
    
    def get_rigid_pole(self):
        rigid_usd = self._asset_path + "/rigid_pole.usd"
        add_reference_to_stage(rigid_usd, "/World/rigid_pole")
        
        init_positions = torch.zeros((1, 3))        
        init_positions[:, 1] = random.uniform(1.2, 1.8)
        
        self.init_rigid_pos = init_positions
        
        GeometryPrimView(
            prim_paths_expr="/World/rigid_pole",
            positions=init_positions,
        )
        
        self.rigid_pole = GeometryPrimView(
            prim_paths_expr="/World/rigid_pole/pole",
            visibilities=[True],
        )

    def get_gripper(self):
        gripper_usd = self._asset_path + "/franka_gripper_fix.usd"
        add_reference_to_stage(gripper_usd, "/World/franka_gripper")
        
        # init_positions, random_orientations = self.get_random_gripper_pose()
        # euler_angles_to_quat()
        init_positions = torch.zeros((1, 3))     
        init_positions[:, 2] = 1.

        self.gripper = ArticulationView(
            prim_paths_expr="/World/franka_gripper",
            translations=init_positions, 
            # orientations=random_orientations,
            name="gripper")

    def get_random_gripper_pose(self, sliced_positions):
        #TODO: Add select init positions from oring collision mesh points

        _init_point = random.choice(sliced_positions)
        init_positions = torch.zeros((1, 3))     
        init_positions[:, 0] = _init_point[0]
        init_positions[:, 1] = _init_point[1]
        init_positions[:, 2] = _init_point[2]+0.1
        
        # init_positions[:, 2] = 0.7
        init_orientation = [0,0,0]
        init_orientation[0] = random.randint(0, 360)
        init_orientation[1] = random.randint(0, 360)
        init_orientation[2] = random.randint(0, 360)
         
        random_orientations = torch.as_tensor([euler_angles_to_quat(euler_angles = init_orientation, degrees=True)])

        return init_positions, random_orientations
    
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
    
    def position_slice(self, positions, render=True):
        slice_min = 0.2
        slice_max = 1.2
        sliced_positions = [el for el in positions if (el[1] >= slice_min) and (el[1] <= slice_max)]

        if render:
            N, _ = np.array(self.deformable_body.GetCollisionPointsAttr().Get()).shape
            point_list = np.zeros([N, 3])
            point_list[:len(sliced_positions), :] = sliced_positions

            self.points.GetPointsAttr().Set(point_list)  # vis
        
        return sliced_positions


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
    
    def initalize(self):
        """
        Need to use world.reset()
        """
        self.hook.initialize()
        self.joint_x = self.hook.get_dof_index("pris_x")
        self.joint_y = self.hook.get_dof_index("pris_y")
        self.joint_z = self.hook.get_dof_index("pris_z")
        self.rev_x = self.hook.get_dof_index("rev_x")

        self.gripper.initialize()
        self.gripper_joint_z = self.gripper.get_dof_index("joint_z")
        self.gripper_rev_z = self.gripper.get_dof_index("rev_z")

    ## reset functions ##
    
    def get_pre_reset(self):
        delete_prim("/World/rigid_pole")
        delete_prim("/World/pole_env")
        delete_prim("/World/oring_01_05")
        # delete_prim("/World/oring_01_05")
        
        object_usd_path = self._asset_path  + "/oring_task_env_01_05_default.usd"
        open_stage(usd_path=object_usd_path)
        
        self.get_hook()
        self.get_rigid_pole()
        self.get_gripper()
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
        print("delete hook")
        
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

    def main(self):
        object_usd_path = self._asset_path + "/oring_task_env_01_05_default.usd"
        open_stage(usd_path=object_usd_path)
        
        world = World(stage_units_in_meters=1)
        self.world = world
        self.init_simulation()
        
        self.get_hook()
        self.get_rigid_pole()
        self.get_gripper()

        world.reset()
        self.initalize()
        
        self.define_paths()

        self.get_deform_point()
        self.visualizer_setup()

        i = self.max_episode_length
        TOUCH = False

        self.hit_prim = []
        path_tuple = PhysicsSchemaTools.encodeSdfPath(self.deform_path)
    
        while simulation_app.is_running():
            if world.is_playing():
                if world.current_time_step_index == 0:
                    world.reset()
            world.step(render=True) 
            
            ## RESET PART ##
            if i == self.max_episode_length: # randomize reset (max_episode_length)
                self.get_pre_reset()
                self.get_reset()
                world.step()
                i = 0

                positions = self.get_deform_point()

                # from IPython import embed; embed()
                from IPython import embed; embed(); exit()
                # cloud = g.trimesh.points.PointCloud(points)

                import trimesh as t
                # create trimesh object by loading mesh
                # transfer the pose 
                # check the trimesh object if it contains positions of the point clooud.
                #any(item == True for item in c.contains(positions))

                self.visualizer_setup()
                sliced_positions = self.position_slice(positions=positions, render=True)

                while TOUCH == False:
                    get_physx_scene_query_interface().overlap_mesh(path_tuple[0], 
                                                                   path_tuple[1], 
                                                                   self.report_hit, False)

                    obstacle_contact = all(element not in self.hit_prim for element in self.obstacle_prims)
                    # world.play()
                    if obstacle_contact:
                        # print("no contact gripper head with oring")
                        if self.touch_prim in self.hit_prim:
                            TOUCH = True
                            print("jotno")
                        else:
                            print("amugotdo..")
                            # sliced_positions
                            # world.pause()
                            positions = self.get_deform_point()
                            sliced_positions = self.position_slice(positions=positions, render=True)


                            pos, ori = self.get_random_gripper_pose(sliced_positions)
                            self.gripper.set_world_poses(pos, ori)
                            world.step()
                            self.hit_prim = []

                    else:
                        print('sibal obastacle!')
                        # world.pause()
                        positions = self.get_deform_point()
                        sliced_positions = self.position_slice(positions=positions, render=True)
                        pos, ori = self.get_random_gripper_pose(sliced_positions)
                        self.gripper.set_world_poses(pos, ori)
                        world.step()
                        # world.pause()
                        self.hit_prim = []

                if TOUCH:
                    world.step()
                    world.pause()
                    self.set_attach(rigid=self.attach_prim, deform=self.deform_path)
                    world.step()
                    world.play()
                    TOUCH = False
                    self.hit_prim = []
                    delete_prim("/World/Points")
            ## RESET PART ##

            ## MOVE RANDOM ACTION
            state = self.gripper.get_joint_positions()
            state[:, 0] += random.uniform(-.1, .1)
            state[:, 1] += random.uniform(-.1, .1)
            state[:, 2] += random.uniform(-.1, .1)
            state[:, 3] += random.uniform(-.1, .1)
            state[:, 4] += random.uniform(-.1, .1)
            state[:, 5] += random.uniform(-.1, .1)
            
            self.gripper.set_joint_position_targets(state)
            i += 1

            
                
            
if __name__ == "__main__":
    try:
        test = Test()
        test.main()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
