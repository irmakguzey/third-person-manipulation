# Environment to play with the robotic hand for having a fingertip
# based action space 
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch 
from isaacgym.torch_utils import *

import math 
import numpy as np 
import torch

from .sim_env import SimulationEnv

class SpongeFlippingEnv(SimulationEnv):
    def __init__(self, asset_root): 
        super().__init__(asset_root=asset_root)

        self.set_home_state()
        self.viewer = None

     # Set Home state
    def set_home_state(self):  
        self.hand_home_state = torch.tensor([
            [-0.00137183, -0.22922094, 0.7265581, 0.79128325,
            0.9890924, 0.37431374, 0.36866143, 0.77558154,
            0.00662423, -0.23064502, 0.73253167, 0.7449019, 
            0.08261403, -0.15844858, 0.82595366, 0.7666822 ]]) 
        
        self.object_home_state = torch.tensor([0.2, 1.7, 0.0, -1.3, -0.707, 0, 0])
        self.endeff_home_state = torch.tensor([0.2, 1.5, 0.0, 0, 0, 0.8509035, 0.525322]) 

    def load_urdfs(self): 
        print('** Loading URDFs **')
        actor_asset_file = "allegro_hand_description/urdf/model_only_hand.urdf"
        table_asset_file = "allegro_hand_description/urdf/table.urdf"
        cube_asset_file = "allegro_hand_description/urdf/cube_multicolor.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments =  False 
        asset_options.use_mesh_materials = True
        asset_options.disable_gravity = True
        
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset_options.flip_visual_attachments = False
        table_asset_options.collapse_fixed_joints = True
        table_asset_options.disable_gravity = True
        
        self.actor_asset = self.gym.load_urdf(self.sim, self.asset_root, actor_asset_file, asset_options)
        self.table_asset = self.gym.load_urdf(self.sim, self.asset_root, table_asset_file, table_asset_options)

        object_asset_options = gymapi.AssetOptions()
        self.object_asset= self.gym.load_urdf(self.sim, self.asset_root, cube_asset_file, object_asset_options)

    def set_poses(self): 
        # Actor pose 
        self.actor_pose = gymapi.Transform() 
        self.actor_pose.p = gymapi.Vec3(0.5, 1.5, 0.0)
        self.actor_pose.r = gymapi.Quat(-0.707, -0.707, 0, 0)

        # Table pose 
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        self.table_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        # Object pose
        self.object_pose = gymapi.Transform()
        self.object_pose.p = gymapi.Vec3() 
        self.object_pose.p.x = self.actor_pose.p.x
        pose_dy, pose_dz = 0, -0.05
        self.object_pose.p.y = self.actor_pose.p.y + pose_dy
        self.object_pose.p.z = self.actor_pose.p.z + pose_dz
            
    def create_camera_sensors(self): 

        print('  ** Creating camera sensors **')
        camera_props = gymapi.CameraProperties() 
        camera_props.horizontal_fov = 35
        camera_props.width = 480
        camera_props.height = 480
        camera_props.enable_tensors = True

        # Create the camera sensor
        self.camera_handle = self.gym.create_camera_sensor(self.env, camera_props) # To be used in receiving the camera image
        print('  Created camera sensor')
        
        # Actually set the camera position
        camera_position = gymapi.Vec3(0.2, 2.3, 0.0)
        camera_target = gymapi.Vec3(0.2, 1.5, 0.0)
        self.gym.set_camera_location(self.camera_handle, self.env, camera_position, camera_target)
        print('  Set camera location')
        self.gym.start_access_image_tensors(self.sim)   
        print('  Started access to image tensors')

    def create_handlers_and_indices(self):
        # Create the object handlers to control each component
        self.set_poses()
        self.actor_handle = self.gym.create_actor(self.env, self.actor_asset, self.actor_pose, 'actor', 0, 1)
        self.table_handle = self.gym.create_actor(self.env, self.table_asset, self.table_pose, 'table', 0, 1)
        self.object_handle = self.gym.create_actor(self.env, self.object_asset, self.object_pose, 'cube', 0, 0, 0)

        # Get the indices / num dofs
        self.num_dofs = self.gym.get_asset_dof_count(self.actor_asset)
        print('  Num DOFs: {}'.format(self.num_dofs))
        self.actor_idx = self.gym.get_actor_index(self.env, self.actor_handle, gymapi.DOMAIN_SIM)
        self.object_idx = self.gym.get_actor_index(self.env, self.object_handle, gymapi.DOMAIN_SIM)