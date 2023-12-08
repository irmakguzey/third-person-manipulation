# Script that has essential simulation initializations

from isaacgym import gymapi, gymutil
from isaacgym import gymtorch 
from isaacgym.torch_utils import *

import math
import numpy as np
import torch

def create_simulation(): 
    # Initial lines to create the environment 
    sim_params = gymapi.SimParams()
    physics_engine = gymapi.SIM_PHYSX
    gym = gymapi.acquire_gym()

    # Simulation common parameters
    sim_params.dt = 1/60
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z 
    sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0)

    # Physics engine parameters 
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    compute_device_id=1

    # Creating the sim with these parameters 
    sim = gym.create_sim(compute_device_id, 1, physics_engine, sim_params)

    # Add ground
    plane_params = gymapi.PlaneParams() 
    gym.add_ground(sim, plane_params)

    return sim, gym

def load_urdf(sim, gym, asset_root='/home/irmak/Workspace/third-person-manipulation/3rd_person_man/environment/urdf'):
    
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments =  False #asset_descriptors[asset_id].flip_visual_attachments
    asset_options.use_mesh_materials = True
    asset_options.disable_gravity = True

    actor_asset_file = 'allegro_hand_description/model_only_hand.urdf'
    table_asset_file = 'table.urdf' # For now we only have the robot hand and a table

    actor_asset = gym.load_urdf(sim, asset_root, actor_asset_file, asset_options)
    table_asset = gym.load_urdf(sim, asset_root, table_asset_file, asset_options)

    return actor_asset, table_asset


def create_environment(sim, gym, actor_asset, table_asset, spacing = 2.5):

    # Environment parameters 
    env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    # Create the environment
    env = gym.create_env(sim, env_lower, env_upper, 1) # We will only have 1 environment anyways so setting the number of rows to 1

    # Set the camera parameters 
    set_camera_params(sim, gym, env)

    # Create the object handlers to control each component
    actor_pose = set_actor_pose()
    table_pose = set_table_pose()
    actor_handle = gym.create_actor(env, actor_asset, actor_pose, 'actor', 0, 1)
    table_handle = gym.create_actor(env, table_asset, table_pose, 'table', 0, 1)

    # Color the hand
    num_dofs = gym.get_asset_dof_count()
    color_hand(gym, env, num_dofs, actor_handle)

    # Set the actor properties such as stiffness / damping and etc 
    props = gym.get_actor_dof_properties(env, actor_handle)
    props["stiffness"] =[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
    props["damping"] =  [0.1,0.1,0.1,0.1,0.1,0,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    props["friction"] = [0.01]*16
    props["armature"] = [0.001]*16
    props = set_control_mode(props = props, mode = 'Position_Velocity')
    gym.set_actor_dof_properties(env, actor_handle, props) 

    return env, gym, sim

def set_camera_params(sim, gym, env): 
    camera_props = gymapi.CameraProperties() 
    camera_props.horizontal_fov = 35
    camera_props.width = 480
    camera_props.height = 480
    camera_props.enable_tensors = True
    camera_handle = gym.create_camera_sensor(env, camera_props)
    camera_position = gymapi.Vec3(1.06,1.6 , -0.02) #Camera Position #gymapi.Vec3(1,1.2, 0.0)
    camera_target = gymapi.Vec3(1.03,1.3 , -0.02)   #Camera Target 
    gym.set_camera_location(camera_handle, env, camera_position, camera_target)
    gym.start_access_image_tensors(sim)   


def set_actor_pose(): 
    actor_pose = gymapi.Transform() 
    actor_pose.p = gymapi.Vec3(1, 1.2, 0.0)
    actor_pose.r = gymapi.Quat(-0.707, -0.707, 0, 0)

    return actor_pose 

def set_table_pose(): 
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.7, 0.0, 0.3)
    table_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    return table_pose 

def color_hand(gym, env, num_dofs, actor_handle): 
    for j in range(num_dofs+13):   
        if j!=20 and j!=15 and j!=10 and j!=5 : 
            gym.set_rigid_body_color(env, actor_handle,j, gymapi.MESH_VISUAL, gymapi.Vec3(0.15, 0.15, 0.15))


def set_control_mode(self, props, mode=None):
    for k in range(self.num_dofs):
        if mode is not None:
            if mode=='Position':
                props["driveMode"][k] = gymapi.DOF_MODE_POS
            elif mode=='Velocity':
                props["driveMode"][k] = gymapi.DOF_MODE_VEL
            elif mode=='Effort':
                props["driveMode"][k] = gymapi.DOF_MODE_EFFORT
            elif mode=='Position_Velocity':
                props["driveMode"][k] = gymapi.DOF_MODE_POS   

    return props
