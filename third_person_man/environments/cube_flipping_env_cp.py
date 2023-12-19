from isaacgym import gymapi, gymutil
import numpy as np

from isaacgym import gymtorch
from isaacgym.torch_utils import *
import torch

class CubeFlippingSim():

        def __init__(self, num_envs =1,num_per_row = 1,spacing = 2.5,show_axis=0, cam_pose=gymapi.Vec3(0,1,0), env_path=None, log_file=None,log_conf={},full_log=False,env_suite='banana', flag=0, control_mode= 'Position_Velocity', is_kinova=False,**kwargs):#gymapi.Vec3(2,4,5)):
                
                sim_params = gymapi.SimParams()
                physics_engine=gymapi.SIM_PHYSX
                self.gym=gymapi.acquire_gym()
                
                self.num_per_row=num_per_row
                self.spacing=spacing
                
                self.env_lower = gymapi.Vec3(-self.spacing, 0.0, -self.spacing)
                self.env_upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)
                
                self.object_indices=[]
                
                self.device = "cpu"
                
                self.control_mode=control_mode

        # set common parameters
                sim_params.dt = self.dt= 1/60
                sim_params.substeps = 2
                sim_params.up_axis = gymapi.UP_AXIS_Z
                sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0)
        # set PhysX-specific parameters
                if physics_engine==gymapi.SIM_PHYSX:
                        sim_params.physx.use_gpu = True 
                        sim_params.physx.solver_type = 1
                        sim_params.physx.num_position_iterations = 6
                        sim_params.physx.num_velocity_iterations = 1
                        sim_params.physx.contact_offset = 0.01
                        sim_params.physx.rest_offset = 0.0
                        compute_device_id=1
                        graphics_device_id=1

        # set Flex-specific parameters
                elif physics_engine==gymapi.SIM_FLEX:
                        sim_params.flex.solver_type = 5
                        sim_params.flex.num_outer_iterations = 4
                        sim_params.flex.num_inner_iterations = 20
                        sim_params.flex.relaxation = 0.8
                        sim_params.flex.warm_start = 0.5
                        compute_device_id=0
                        graphics_device_id=0
                        
        # create sim with these parameters
                print("Creating Sim")
              
                self.sim = self.gym.create_sim(compute_device_id, 1, physics_engine, sim_params)
                #self.sim_device="cuda:0"

        # Add ground
                self.plane_params = gymapi.PlaneParams()
                self.gym.add_ground(self.sim, self.plane_params)
                
        # create viewer #
                #self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
                self.viewer = None
                #Get the camera pose and place the camera there
                self.cam_pose = cam_pose
                if self.viewer is None:
                        print("*** Failed to create viewer")
                        #quit()
        # set asset options
                asset_options = gymapi.AssetOptions()
                asset_options.fix_base_link = True
                asset_options.flip_visual_attachments =  False #asset_descriptors[self.asset_id].flip_visual_attachments
                asset_options.use_mesh_materials = True
                asset_options.disable_gravity = True
                
                table_asset_options = gymapi.AssetOptions()
                table_asset_options.fix_base_link = True
                table_asset_options.flip_visual_attachments = False
                table_asset_options.collapse_fixed_joints = True
                table_asset_options.disable_gravity = True

                #get asset file
                self.asset_root = "/home/aadhithya/tactile-learning/envs/dexterous_env/urdf"
                self.asset_file = "allegro_hand_description/urdf/model_only_hand.urdf"
                self.table_asset= "allegro_hand_description/urdf/table.urdf"
                self.cube_asset= "allegro_hand_description/urdf/cube_multicolor.urdf"
                print("Loading asset '%s' from '%s'" % (self.asset_file, self.asset_root)) 
                
                self.asset = self.gym.load_urdf(self.sim, self.asset_root, self.asset_file, asset_options)
                self.table_asset = self.gym.load_urdf(self.sim, self.asset_root, self.table_asset, table_asset_options)
        
                object_asset_options = gymapi.AssetOptions()
                self.object_asset= self.gym.load_urdf(self.sim, self.asset_root, self.cube_asset,object_asset_options)
               
                
                self.num_dofs=self.get_dof_count()
                print("Num DOFS", self.num_dofs)
               
                # Call Function to create and load the environment              
                self.load()
                self.root_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
                print("Root state tensor",self.root_state_tensor)
                self.root_state_tensor = self.root_state_tensor.view(-1, 13)
                self.object_indices=to_torch(self.object_indices, dtype=torch.int32,device='cpu')
                
                self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
                self.rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
                print("Initialisation complete")

                self.home_position=torch.zeros((1, self.num_dofs),dtype=torch.float32, device='cpu')
               
                self.set_home_state()

        #Function to set camera params      
        def set_camera_params(self):
                self.camera_props = gymapi.CameraProperties()
                self.camera_props.horizontal_fov = 35
                self.camera_props.width = 480
                self.camera_props.height = 480
                self.camera_props.enable_tensors = True
                print('pre camera')
                self.camera_handle = self.gym.create_camera_sensor(self.env, self.camera_props)
                print('post camera')
                camera_position = gymapi.Vec3(1.06,1.6 , -0.02) #Camera Position #gymapi.Vec3(1,1.2, 0.0)
                camera_target = gymapi.Vec3(1.03,1.3 , -0.02)   #Camera Target 
                self.gym.set_camera_location(self.camera_handle, self.env, camera_position, camera_target)
                self.camera_handles.append(self.camera_handle)
                self.gym.start_access_image_tensors(self.sim)   


        #Set Home state
        def set_home_state(self): 
                self.home_state = torch.zeros(1,16) 
                self.home_state[0]=torch.tensor([-0.00137183, -0.22922094,  0.7265581 ,  0.79128325,0.9890924 ,  0.37431374,  0.36866143,
                                        0.77558154,  0.00662423,
                                        -0.23064502,  0.73253167,  0.7449019 ,  0.08261403, -0.15844858,
                                        0.82595366,  0.7666822 ])  
  
                   
        #Set actor pose
        def set_actor_pose(self):
  
                self.actor_pose = gymapi.Transform()
                self.actor_pose.p = gymapi.Vec3(1,1.2, 0.0)
                self.actor_pose.r = gymapi.Quat(-0.707,-0.707, 0,0)

        #Set Object Initial Pose
        def set_init_object_pose(self):
                self.object_pose = gymapi.Transform()
                self.object_pose.p = gymapi.Vec3()
                                                        
                self.object_pose.p.x =self.actor_pose.p.x
                pose_dy, pose_dz = 0, -0.05

                self.object_pose.p.y = self.actor_pose.p.y + pose_dy
                self.object_pose.p.z = self.actor_pose.p.z + pose_dz

        #Color the fingers of the Hand

        def color_hand(self):
                for j in range(self.num_dofs+13):   
                        if j!=20 and j!=15 and j!=10 and j!=5 : 
                                self.gym.set_rigid_body_color(self.env, self.actor_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.15, 0.15, 0.15))

        #Load the asset

        def load(self):
                
                #self.dof_props=self.get_dof_properties()
                self.camera_handles = []
                self.object_handles=[]
                print("Loading Assets")
        
                self.env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)
                print("Env Created")

                #Set camera parameters
                self.set_camera_params()        
                self.set_actor_pose()

                #Table Pose
                self.table_pose = gymapi.Transform()
                self.table_pose.p = gymapi.Vec3(0.7, 0.0, 0.3)
                self.table_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)


                #Create Actor Handles
                self.actor_handle = self.gym.create_actor(self.env, self.asset, self.actor_pose, "actor", 0, 1)
                self.table_handle = self.gym.create_actor(self.env, self.table_asset, self.table_pose, "table", 0, 1)
                
                #Set Object Init Pose
                self.set_init_object_pose()
                
                #Create Objects
                self.object_handle = self.gym.create_actor(self.env, self.object_asset,self.object_pose, "cube",0, 0, 0)
                
                object_idx = self.gym.get_actor_index(self.env, self.object_handle, gymapi.DOMAIN_SIM)
                self.object_indices.append(object_idx)                        
                
                # Set Color for the Hand Urdf coz Urdf is fully white
                self.color_hand()
                
                #Gets Actor DOF properties
                props = self.gym.get_actor_dof_properties(self.env, self.actor_handle)
                props["stiffness"] =[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]
                props["damping"] =  [0.1,0.1,0.1,0.1,0.1,0,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
                props["friction"] = [0.01]*16
                props["armature"] = [0.001]*16
                props = self.set_control_mode(props = props, mode = 'Position_Velocity')
                self.gym.set_actor_dof_properties(self.env, self.actor_handle, props) 

        #Get dof names
                        
        def get_dof_names(self):
                dof_names = self.gym.get_asset_dof_names(self.asset)
                return dof_names

        #Get Dof properties 
        def get_dof_properties(self):
                dof_props = self.gym.get_asset_dof_properties(self.asset)
                return dof_props

        #Get DOF count
        def get_dof_count(self):
                num_dofs = self.gym.get_asset_dof_count(self.asset)
                return num_dofs
        

        # Get DOF States
        def get_dof_states(self):
                dof_states = np.zeros(self.num_dofs, dtype=gymapi.DofState.dtype)
                return dof_states
        

        #Get DOF positions
        def get_dof_positions(self):
                #self.dof_states = self.gym.get_actor_dof_states(self.envs[i], self.actor_handles[i], gymapi.STATE_NONE)
                self.position=np.zeros(self.num_dofs)
                for i in range(self.num_dofs):
                        self.position[i]=self.gym.get_dof_position(self.env,i)
                return self.position
        
        #Get DOF velocities
        def get_dof_velocities(self):
                #self.dof_states = self.gym.get_actor_dof_states(self.envs[i], self.actor_handles[i], gymapi.STATE_NONE)
                self.velocity=np.zeros(self.num_dofs)
                for i in range(self.num_dofs):
                        self.velocity[i]=self.gym.get_dof_velocity(self.env,i)
                return self.velocity


        #Get DOF types
        def get_dof_types(self):
                dof_types = [self.gym.get_asset_dof_type(self.asset, i) for i in range(self.num_dofs)]
                return dof_types
        
        #Create Sim Viewer
        def create_viewer(self):
                viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
                if viewer is None:
                        print("*** Failed to create viewer")
                        quit()
                return viewer

        #Get Num environments
        def get_num_envs(self):
                return self.num_envs 
        

        #Set Object State at the start of each episode                         
        def set_object_state(self):
                #Set Object State
                self.root_state_tensor[self.object_indices[0],0:3]=to_torch([1,1.3,0.06],dtype=torch.float,device='cpu') #Cube Position 
                self.root_state_tensor[self.object_indices[0],3:7]=to_torch([-1.3,-0.707, 0, 0],dtype=torch.float,device='cpu') #Cube Orientation

        
        # This Function is used for resetting the Environment
        def reset(self):
                #self.load()
                                   
                # self.set_home_state()
                self.set_position(self.home_state)  

                # Set initial Object State
                self.set_object_state()       
                self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                       gymtorch.unwrap_tensor(self.root_state_tensor),
                                                       gymtorch.unwrap_tensor(self.object_indices), len(self.object_indices))    
                
                # Code For Simulating and Stepping Graphics
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                self.gym.refresh_dof_state_tensor(self.sim)
                self.gym.step_graphics(self.sim)
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)
                        
                #Get Observation
                self.obs={}
                self.obs['pixels'] = self.compute_observation(observation= 'image')
                self.obs['features'] = self.compute_observation(observation= 'position')
              
                return self.obs
                

       
        def compute_reward(self):
                reward=0
                return reward

        def terminal_state(self):
                done=0
                return done

        #Get DOF lower limits
        def get_lower_limits(self):
                dof_props=self.get_dof_properties()
                lower_limits = dof_props['lower']
                lower_limits=to_torch(lower_limits)
                return lower_limits.to("cpu")

        #Get DOF upper limits
        def get_upper_limits(self):
                dof_props=self.get_dof_properties()
                upper_limits = dof_props['upper']
                upper_limits=to_torch(upper_limits)  
                return upper_limits.to("cpu")
        
        #Step Function
        def step(self,action):
                action=to_torch(action,dtype=torch.float, device='cpu') 
                self.next_state=np.zeros(self.num_dofs)
                self.next_update_time = 0.1
                self.frame = 0
                t = self.gym.get_sim_time(self.sim)
                # print('self.actions: {}'.format(self.actions))             
                self.set_position(action)
                
                #Function for simulating
                for i in range(1):
                        
                        self.gym.simulate(self.sim)
                        self.gym.fetch_results(self.sim, True)
                        self.gym.refresh_dof_state_tensor(self.sim)
                      
                # step rendering
                self.gym.step_graphics(self.sim)
                self.gym.render_all_camera_sensors(self.sim)

                self.gym.draw_viewer(self.viewer, self.sim, False)
                
            
                #Compute Next Image state and Next position
               
                self.nextstate=self.compute_observation(observation='image')
                self.nextposition = self.compute_observation(observation='position')
              
                self.obs={}
                
                self.obs['pixels']=self.nextstate
                self.obs['features']=self.nextposition                
               
               
                self.reward, self.done, infos = 0, False, {'is_success': False} 
                self.gym.clear_lines(self.viewer)
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                
                #print("Done")

                return self.obs,self.done,self.reward, infos

        #Function for computing Observation
        def compute_observation(self, observation):
                self.gym.refresh_dof_state_tensor(self.sim)
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_rigid_body_state_tensor(self.sim) 

                if observation=='image':
                        for i in range(1):
                            self.color_image =self.gym.get_camera_image_gpu_tensor(self.sim,self.env, self.camera_handles[i], gymapi.IMAGE_COLOR)
                            self.color_image=gymtorch.wrap_tensor(self.color_image)
                            self.color_image=self.color_image.cpu().numpy()
                            self.color_image=self.color_image[:,:,[0,1,2]]
                            
                        state= np.transpose(self.color_image, (2,0,1))
                    
                elif observation=='position':
                        
                        state=np.zeros(self.num_dofs)
                        for i in range(self.num_dofs):
                                state[i]=self.gym.get_dof_position(self.env,i)  

                elif observation=='velocity':
                        state=np.zeros(self.num_dofs)
                        for i in range(self.num_dofs):
                                state[i]=self.gym.get_dof_velocity(self.env,i) 
                       
                elif observation=='full_state':
                        for i in range(2*self.num_dofs):
                                if i<self.num_dofs:
                                        state[i]=self.gym.get_dof_position(self.env,i)  
                                else:
                                        state[i]=self.gym.get_dof_velocity(self.env,i)  


               
                return state
        
        #Get Hand position
        def get_dof_position(self):
                self.state=self.compute_observation(observation='position')[6:]
                return self.state
        
        #Get Arm position
        def get_arm_position(self):
                self.state=self.compute_observation(observation='position')[0:6]
                return self.state
        
        #Get Arm Velocity
        def get_arm_velocity(self):
                self.state=self.compute_observation(observation='velocity')[0:6]
                return self.state
        
        #Get full position
        def get_state(self):
                self.state=self.compute_observation(observation='position')
                return self.state

        def update_log(self):
                self.log.add('state', self.get_state().tolist())

        
        def get_time(self):
                return self.gym.get_elapsed_time(self.sim)

        #Get Cartesian Position of Table 
        def get_table_cartesian(self):
                self.table_handle = self.gym.find_actor_rigid_body_handle(self.env, self.table_handle, "base_link")
                self.table_pose = self.gym.get_rigid_transform(self.env, self.table_handle)
                self.table_position = [self.table_pose.p.x, self.table_pose.p.y, self.table_pose.p.z]   
                return self.table_position
        

        #Get Hand effector position
        def get_cartesian_position(self):
                self.end_eff_handle = self.gym.find_actor_rigid_body_handle(self.env, self.actor_handle, "kinova_end_effector")
                self.end_eff_pose = self.gym.get_rigid_transform(self.env, self.end_eff_handle)
                self.end_eff_position = np.array([self.end_eff_pose.p.x, self.end_eff_pose.p.y, self.end_eff_pose.p.z])
                self.end_eff_rotation = np.array([self.end_eff_pose.r.x, self.end_eff_pose.r.y, self.end_eff_pose.r.z, self.end_eff_pose.r.w])
                self.end_eff_pos= np.concatenate((self.end_eff_position,self.end_eff_rotation))
                return self.end_eff_pos

       # Set DOF position
        def set_position(self, position):
                self.gym.set_dof_position_target_tensor(self.sim,  gymtorch.unwrap_tensor(position))

       # Set DOF velocity
        def set_velocity(self,velocity):
                self.gym.set_dof_velocity_target_tensor(self.sim,  gymtorch.unwrap_tensor(velocity))


        # # Control Mode of DOF operation
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

        # Render the image 
        def render(self, mode='rbg_array', width=0, height=0):
                return self.compute_observation(observation='image')


if __name__ == '__main__':
    env = CubeFlippingSim()
    env.reset()
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    color_image = env.render()
    print('color_image.shape: {}'.format(color_image.shape))
    plt.imshow(np.transpose(color_image, (1,2,0)) )
    plt.savefig('ex_image_cube.jpg') 
