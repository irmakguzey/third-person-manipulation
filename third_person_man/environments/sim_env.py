# Class for any simulated environment

from isaacgym import gymapi, gymutil
from isaacgym import gymtorch 
from isaacgym.torch_utils import *

import math
import numpy as np
import torch

class SimulationEnv: 
    def __init__(self, asset_root):

        # Create the simulation
        self.create_simulation()

        # Load the object / robot urdfs
        self.load_urdfs(asset_root = asset_root)

        # Create the environment
        self.create_environment(spacing = 2.5)

        # Initialize State Vectors
        self.initialize_states()

    def create_simulation(self): 
        print('** Creating Simulation **')
        
        # Initial lines to create the environment 
        sim_params = gymapi.SimParams()
        physics_engine = gymapi.SIM_PHYSX
        self.gym = gymapi.acquire_gym()

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
        compute_device_id = 1 # This is required for running the simulation on the background as well
        graphics_device_id = 1
        # Creating the sim with these parameters 
        self.sim = self.gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)
        print('Simulation Created')

        # Add ground
        plane_params = gymapi.PlaneParams() 
        self.gym.add_ground(self.sim, plane_params)
        print('Plane Added')

    def load_urdfs(self, asset_root): 

        print('** Loading URDFs **')

        # asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True
        # asset_options.flip_visual_attachments =  False #asset_descriptors[asset_id].flip_visual_attachments
        # asset_options.use_mesh_materials = True
        # asset_options.disable_gravity = True

        # table_asset_options = gymapi.AssetOptions()
        # table_asset_options.fix_base_link = True
        # table_asset_options.flip_visual_attachments =  False #asset_descriptors[asset_id].flip_visual_attachments
        # table_asset_options.collapse_fixed_joints = True
        # table_asset_options.disable_gravity = True

        # # print('table asset options: {}'.format(table_asset_options))

        # actor_asset_file = 'allegro_hand_description/urdf/model_only_hand.urdf'
        # table_asset_file = 'allegro_hand_description/urdf/table.urdf' # For now we only have the robot hand and a table

        # self.actor_asset = self.gym.load_urdf(self.sim, asset_root, actor_asset_file, asset_options)
        # print(f'  Loaded the actor at {actor_asset_file}')
        # self.table_asset = self.gym.load_urdf(self.sim, asset_root, table_asset_file, table_asset_options)

        asset_options = gymapi.AssetOptions() # NOTE: Test this
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments =  False #asset_descriptors[self.asset_id].flip_visual_attachments
        asset_options.use_mesh_materials = True
        asset_options.disable_gravity = True
        
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset_options.flip_visual_attachments = False
        table_asset_options.collapse_fixed_joints = True
        table_asset_options.disable_gravity = True

        # Get asset file
        self.asset_root = "/home/irmak/Workspace/third-person-manipulation/third_person_man/models"
        self.asset_file = "allegro_hand_description/urdf/model_only_hand.urdf"
        self.table_asset= "allegro_hand_description/urdf/table.urdf"
        self.cube_asset= "allegro_hand_description/urdf/cube_multicolor.urdf"
        print("Loading asset '%s' from '%s'" % (self.asset_file, self.asset_root)) 
        
        self.actor_asset = self.gym.load_urdf(self.sim, self.asset_root, self.asset_file, asset_options)
        self.table_asset = self.gym.load_urdf(self.sim, self.asset_root, self.table_asset, table_asset_options)

        object_asset_options = gymapi.AssetOptions()
        self.object_asset= self.gym.load_urdf(self.sim, self.asset_root, self.cube_asset,object_asset_options)

    def create_environment(self, spacing = 2.5): 
        print('** Creating Environment **')
        # Environment parameters 
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # Create the environment
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1) # We will only have 1 environment anyways so setting the number of rows to 1

        # Set the camera parameters 
        self.create_camera_sensors()

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

        # Color the hand
        self.color_hand()

        # Set the actor properties such as stiffness / damping and etc 
        props = self.gym.get_actor_dof_properties(self.env, self.actor_handle)
        props["stiffness"] =[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
        props["damping"] =  [0.1,0.1,0.1,0.1,0.1,0,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
        props["friction"] = [0.01]*16
        props["armature"] = [0.001]*16
        props = self.set_control_mode(props = props, mode = 'Position_Velocity')
        self.gym.set_actor_dof_properties(self.env, self.actor_handle, props) 
        print('Environment created')

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
        camera_position = gymapi.Vec3(1.06,1.6 , -0.02)
        camera_target = gymapi.Vec3(1.03,1.3 , -0.02)
        self.gym.set_camera_location(self.camera_handle, self.env, camera_position, camera_target)
        print('  Set camera location')
        self.gym.start_access_image_tensors(self.sim)   
        print('  Started access to image tensors')
    
    def set_poses(self): 
        # Actor pose 
        self.actor_pose = gymapi.Transform() 
        self.actor_pose.p = gymapi.Vec3(1, 1.2, 0.0)
        self.actor_pose.r = gymapi.Quat(-0.707, -0.707, 0, 0)

        # Table pose 
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(0.7, 0.0, 0.3)
        self.table_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        # Object pose
        self.object_pose = gymapi.Transform()
        self.object_pose.p = gymapi.Vec3() 
        self.object_pose.p.x =self.actor_pose.p.x
        pose_dy, pose_dz = 0, -0.05
        self.object_pose.p.y = self.actor_pose.p.y + pose_dy
        self.object_pose.p.z = self.actor_pose.p.z + pose_dz

    def initialize_states(self): 
        self.root_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).view(-1,13)
        print('Root state tensor: {}'.format(self.root_state_tensor))

    def color_hand(self): 
        for j in range(self.num_dofs+13):   
            if j!=20 and j!=15 and j!=10 and j!=5 : # TODO: Make sure this is correct?
                self.gym.set_rigid_body_color(self.env, self.actor_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.15, 0.15, 0.15))

    def set_control_mode(self, props, mode = None): 
        for k in range(self.num_dofs):
            if mode is not None:
                if mode == 'Position':
                    props["driveMode"][k] = gymapi.DOF_MODE_POS
                elif mode == 'Velocity':
                    props["driveMode"][k] = gymapi.DOF_MODE_VEL
                elif mode == 'Effort':
                    props["driveMode"][k] = gymapi.DOF_MODE_EFFORT
                elif mode == 'Position_Velocity':
                    props["driveMode"][k] = gymapi.DOF_MODE_POS   

        return props
    
    # Set Hand position
    def set_hand_position(self, position): # position: (1,16)
        self.gym.set_dof_position_target_tensor(self.sim,  gymtorch.unwrap_tensor(position))

    # Set Hand velocity
    def set_hand_velocity(self, velocity):
        self.gym.set_dof_velocity_target_tensor(self.sim,  gymtorch.unwrap_tensor(velocity))

    # Get Hand properties 
    def get_hand_properties(self):
        dof_props = self.gym.get_asset_dof_properties(self.asset)
        return dof_props
    
    # Get Hand position
    def get_hand_position(self): 
        state = np.zeros(self.num_dofs)
        for i in range(self.num_dofs):
            state[i] = self.gym.get_dof_position(self.env, i)
        
        return state 
    
    # Get Hand velocities
    def get_hand_velocity(self):
        state = np.zeros(self.num_dofs)
        for i in range(self.num_dofs): 
            state[i] = self.gym.get_dof_velocity(self.env, i)

        return state 

    def get_endeff_position(self):
        state = self.gym.acquire_actor_root_state_tensor(self.sim)
        state = gymtorch.wrap_tensor(state)
        position = state.numpy()[0,0:7]
        return position
    
    def get_endeff_state(self): 
        self.root_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).view(-1,13)
        endeff_state = self.root_state_tensor[self.actor_idx,:]
        return endeff_state

    def set_endeff_position(self, position): # 7 dimensional position for the root tensor, 3 translation, 4 rotation 
        self.root_state_tensor[self.actor_idx, :7] = to_torch(np.array(position), device='cpu')
        actor_indices = to_torch([self.actor_idx], dtype=torch.int32, device='cpu')
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(actor_indices), # NOTE: actor_indices is different than object_indices
            len(actor_indices)
        )

    def set_endeff_velocity(self, velocity): # 6 dimensional velocity, for orientation velocity is 3
        self.root_state_tensor[self.actor_idx, 7:] = to_torch(np.array(velocity), device='cpu') 
        self.actor_indices=to_torch([self.actor_idx], dtype=torch.int32, device='cpu')
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(self.actor_indices),
            len(self.actor_indices)
        )

    def get_object_state(self):
        self.root_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).view(-1,13)
        object_state = self.root_state_tensor[self.object_idx, :]
        return object_state

    def set_object_position(self, position):
        self.root_state_tensor[self.object_idx, :7] = to_torch(np.array(position), device='cpu')
        object_indices = to_torch([self.object_idx], dtype=torch.int32, device='cpu')
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, 
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(object_indices),
            len(object_indices)
        )

    # Get DOF lower limits
    def get_lower_limits(self):
        dof_props=self.get_dof_properties()
        lower_limits = dof_props['lower']
        lower_limits=to_torch(lower_limits)
        return lower_limits.to("cpu")

    # Get DOF upper limits
    def get_upper_limits(self):
        dof_props=self.get_dof_properties()
        upper_limits = dof_props['upper']
        upper_limits=to_torch(upper_limits)  
        return upper_limits.to("cpu")
    
    # Get observations - image
    def get_image(self): 
        color_image = self.gym.get_camera_image_gpu_tensor(
            self.sim,
            self.env,
            self.camera_handle,
            gymapi.IMAGE_COLOR
        )
        color_image = gymtorch.wrap_tensor(color_image)
        color_image = color_image.cpu().numpy()
        print('color_image.shape: {}'.format(color_image.shape))
        color_image = color_image[:,:,[0,1,2]] # NOTE: Make sure that this is what is needed?

        return np.transpose(color_image, (2,0,1)) 
    
    def compute_observation(self, obs_type):
        # Refresh all the simulation states
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if obs_type == 'image':
            return self.get_image()
        
        if obs_type == 'position': 
            hand_position = self.get_hand_position() 
            endeff_position = self.get_endeff_position()
            position = np.concatenate([hand_position, endeff_position], axis=0)
            print('position.shape: {}'.format(position.shape))
            return position
    
        if obs_type == 'velocity': 
            hand_velocity = self.get_hand_velocity()
            endeff_velocity = self.get_endeff_velocity()
            velocity = np.concatenate([hand_velocity, endeff_velocity], axis=0)
            print('velocity.shape: {}'.format(velocity.shape))

            return velocity 
        
        return None
    
    def render(self): 
        return self.get_image()
        
