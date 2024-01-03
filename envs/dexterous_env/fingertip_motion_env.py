# Environment that only moves the fingers 

from isaacgym import gymapi, gymutil
from isaacgym import gymtorch 
from isaacgym.torch_utils import *

import cv2
import math 
import numpy as np 
import torch

from scipy.spatial.transform import Rotation

from .dexterous_simulation_env import DexterousSimulationEnv

class FingertipMotionEnv(DexterousSimulationEnv):
    def __init__(self): 
        super().__init__()

        self.set_home_state()
        self.viewer = None

        # Initialize the KDL solver
        from holobot.robot.allegro.allegro_kdl import AllegroKDL
        self.hand_solver = AllegroKDL()

    def set_home_state(self): 
        self.hand_home_state = torch.tensor([
            [-0.00137183, -0.22922094, 0.7265581, 0.79128325,
            0.9890924, 0.37431374, 0.36866143, 0.77558154,
            0.00662423, -0.23064502, 0.73253167, 0.7449019, 
            0.08261403, -0.15844858, 0.82595366, 0.7666822 ]]) 

        self.endeff_home_state = torch.tensor([0.2, 1.5, 0.0, 0, 0, 0.7071068, 0.7071068 ]) 

    def load_urdfs(self):
        actor_asset_file = "allegro_hand_description/urdf/model_only_hand.urdf"
        table_asset_file = "allegro_hand_description/urdf/table.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
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

    def set_poses(self):
        # Actor pose 
        self.actor_pose = gymapi.Transform() 
        self.actor_pose.p = gymapi.Vec3(0.2, 1.5, 0.0)
        self.actor_pose.r = gymapi.Quat(0, 0, 0.7071068, 0.7071068 )

        # Table pose 
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        self.table_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

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
        camera_position = gymapi.Vec3(0.18, 2.3, 0.0)
        camera_target = gymapi.Vec3(0.2, 1.5, 0.0)
        self.gym.set_camera_location(self.camera_handle, self.env, camera_position, camera_target)
        print('  Set camera location')
        self.gym.start_access_image_tensors(self.sim)   
        print('  Started access to image tensors')

        # Get the camera pose - view matrix and invert it to transform points from camera to world space
        self.camera_pose = np.linalg.pinv(np.transpose(np.matrix(self.gym.get_camera_view_matrix(self.sim, self.env, self.camera_handle))))

        # Get the intrinsics matrix of the camera
        self.intrinsic_matrix = self.compute_camera_intrinsics_matrix(
            image_width = camera_props.width, 
            image_heigth = camera_props.height, 
            horizontal_fov = camera_props.horizontal_fov
        )

    def compute_camera_intrinsics_matrix(self, image_width, image_heigth, horizontal_fov):
        vertical_fov = (image_heigth / image_width * horizontal_fov) * np.pi / 180
        horizontal_fov *= np.pi / 180

        f_x = (image_width / 2.0) / np.tan(horizontal_fov / 2.0)
        f_y = (image_heigth / 2.0) / np.tan(vertical_fov / 2.0)

        K = np.asarray([
            [-f_x, 0.0, image_width / 2.0], 
            [0.0, f_y, image_heigth / 2.0],
            [0.0, 0.0, 1.0]], dtype=np.float32)

        return K

    def create_handlers_and_indices(self):
        # Create the object handlers to control each component
        self.set_poses()
        self.actor_handle = self.gym.create_actor(self.env, self.actor_asset, self.actor_pose, 'actor', 0, 1)
        self.table_handle = self.gym.create_actor(self.env, self.table_asset, self.table_pose, 'table', 0, 1)

        # Get the indices / num dofs
        self.num_dofs = self.gym.get_asset_dof_count(self.actor_asset)
        print('  Num DOFs: {}'.format(self.num_dofs))
        self.actor_idx = self.gym.get_actor_index(self.env, self.actor_handle, gymapi.DOMAIN_SIM)

    def reset(self):
        # Reset
        self.set_hand_position(self.hand_home_state)  
        self.set_endeff_position(self.endeff_home_state)
        
        # Code For Simulating and Stepping Graphics
        self.simulate_and_render()
                
        # Get Observation
        obs = {}
        obs['pixels'] = self.compute_observation(obs_type = 'image') 
        obs['features'] = self.compute_observation(obs_type = 'position')
        
        return obs

    def _turn_frame_to_homo_mat(self, rvec, tvec):

        homo_mat = np.zeros((4, 4))
        homo_mat[:3, :3] = rvec
        homo_mat[:3, 3] = tvec
        homo_mat[3, 3] = 1

        return homo_mat
    
    def _turn_quat_to_rot(self, quat):
        rot = Rotation.from_quat(quat) 
        return rot.as_matrix()
    
    def _get_home_endeff_pose(self, relative_to_camera=True):
        # We are going to calculate the desired pose 
        # wrt the beginning pose
        # This is the home endeff position
        home_tvec = self.endeff_home_state[:3]
        home_rvec = self._turn_quat_to_rot(self.endeff_home_state[3:])
        H_EH_O = self._turn_frame_to_homo_mat(rvec = home_rvec, tvec = home_tvec) # Home position of end effector

        if relative_to_camera:
            # Calculate the current endeff pose wrt the camera
            H_C_O = self.camera_pose
            H_EH_C = np.linalg.pinv(H_C_O) @ H_EH_O
            return H_EH_C
        else:
            return H_EH_O
    
    def _get_current_endeff_pose(self, relative_to_camera=True):
        position = self.get_endeff_position()

        # Turn position to homo matrix
        tvec = position[:3]
        rvec = self._turn_quat_to_rot(position[3:])
        H_ET_O = self._turn_frame_to_homo_mat(rvec= rvec, tvec=tvec)

        if relative_to_camera:
            # Calculate the current endeff pose wrt the camera
            H_C_O = self.camera_pose
            H_ET_C = np.linalg.pinv(H_C_O) @ H_ET_O
            return H_ET_C
        else: 
            return H_ET_O


    def calculate_fingertip_positions(self, features):
        fingertip_poses = []
        # Calculate translational and rotational fingertip positions
        for i, finger_type in enumerate(['index', 'middle', 'ring', 'thumb']):
            finger_tvec, finger_rvec = self.hand_solver.finger_forward_kinematics(
                finger_type, features[i*4:(i+1)*4]
            )

            # Stack tvec and rvec
            fingertip_pose = self._turn_frame_to_homo_mat(
                rvec=finger_rvec,
                tvec=finger_tvec)
            # print('fingertip_pose.shape: {}'.format(fingertip_pose))
            fingertip_poses.append(fingertip_pose)

        fingertip_poses = np.stack(fingertip_poses, axis=0)

        return fingertip_poses
    
    def calculate_fingertip_positions_wrt_world(self, features, endeff_pose='home_pose'):
        # If from_data is set to true endeff pose will be taken from the data collected
        if endeff_pose == 'home_pose': 
            H_E_O = self._get_home_endeff_pose(
                relative_to_camera = False
            )
        elif endeff_pose == 'current_pose':
            H_E_O = self._get_current_endeff_pose(
                relative_to_camera = False
            )

        fingertip_poses = self.calculate_fingertip_positions(features = features)
        fingertip_poses_wrt_world = []
        for H_F_E in fingertip_poses: # Homo to take fingertip pose frame to the end effector frame
            H_F_O = H_E_O @ H_F_E  # Homo to take fingertip pose frame to the origin 
            fingertip_poses_wrt_world.append(H_F_O)

        fingertip_poses_wrt_world = np.stack(fingertip_poses_wrt_world, axis=0)
        return fingertip_poses_wrt_world
    
    def get_action_fingertip_poses(self, features, endeff_pose='home_pose', return_projected_poses=False): 
        # Get the endeffector matrix
        if endeff_pose == 'home_pose': 
            H_E_O = self._get_home_endeff_pose(
                relative_to_camera = False
            )
            if return_projected_poses:
                H_E_C = self._get_home_endeff_pose(
                    relative_to_camera = True
                )
        elif endeff_pose == 'current_pose':
            H_E_O = self._get_current_endeff_pose(
                relative_to_camera = False
            )
            if return_projected_poses:
                H_E_C = self._get_current_endeff_pose(
                    relative_to_camera = True
                )

        fingertip_poses = self.calculate_fingertip_positions(features = features)
        fingertip_poses_wrt_world = []
        if return_projected_poses:
            projected_fingertip_poses = []
        for H_F_E in fingertip_poses: # Homo to take fingertip pose frame to the end effector frame
            H_F_O = H_E_O @ H_F_E  # Homo to take fingertip pose frame to the origin 
            fingertip_poses_wrt_world.append(H_F_O)

            if return_projected_poses:
                # Project the fingertip position
                H_F_C = H_E_C @ H_F_E 
                rvec, tvec = H_F_C[:3, :3], H_F_C[:3, 3]

                # Translate these fingertip points to image
                fingertip_2d = self.project_axes(rvec, tvec, self.intrinsic_matrix)
                projected_fingertip_poses.append(fingertip_2d)

        fingertip_poses_wrt_world = np.stack(fingertip_poses_wrt_world, axis=0)

        if return_projected_poses:
            projected_fingertip_poses = np.stack(projected_fingertip_poses, axis=0)
            return fingertip_poses_wrt_world, projected_fingertip_poses

        return fingertip_poses_wrt_world
        

    def get_projected_fingertip_positions(self, features, endeff_pose='home_pose'):

        # Get the fingertip positions in 3d
        fingertips_3d = self.calculate_fingertip_positions(features)

        # Calculate the camera pose wrt the camera
        if endeff_pose == 'home_pose':
            H_E_C = self._get_home_endeff_pose(
                relative_to_camera = True
            )
        elif endeff_pose == 'current_pose':
            H_E_C = self._get_current_endeff_pose(
                relative_to_camera = True
            )        

        fingertips_2d = []
        # Get the distance of the fingertips to the camera 
        for finger_id in range(len(fingertips_3d)):
            H_F_E = fingertips_3d[finger_id] # Homo to take finger frame to endeffector

            # We want the homo to take finger frame to camera frame
            H_F_C = H_E_C @ H_F_E

            # Extract the tvec and rvecs on this homo
            rvec, tvec = H_F_C[:3, :3], H_F_C[:3, 3]

            # Translate these fingertip points to image
            fingertip_2d = self.project_axes(rvec, tvec, self.intrinsic_matrix)
            fingertips_2d.append(fingertip_2d)

        fingertips_2d = np.stack(fingertips_2d, axis=0)
        return fingertips_2d
    
    def get_projected_endeffector_position(self, endeff_pose='home_pose'):

        if endeff_pose == 'home_pose': 
            H_E_C = self._get_home_endeff_pose(
                relative_to_camera = True
            )
        elif endeff_pose == 'current_pose':
            H_E_C = self._get_current_endeff_pose(
                relative_to_camera = True
            )

        rvec, tvec = H_E_C[:3, :3], H_E_C[:3, 3]
        eef_2d = self.project_axes(rvec, tvec, self.intrinsic_matrix)

        return [eef_2d]
            
    def project_axes(self, rvec, tvec, intrinsic_matrix, scale=0.05, dist=None):
        """
        Draw a 6dof axis (XYZ -> RGB) in the given rotation and translation
        :param img - rgb numpy array
        :rotation_vec - euler rotations, numpy array of length 3,
                        use cv2.Rodrigues(R)[0] to convert from rotation matrix
        :t - 3d translation vector, in meters (dtype must be float)
        :K - intrinsic calibration matrix , 3x3
        :scale - factor to control the axis lengths
        :dist - optional distortion coefficients, numpy array of length 4. If None distortion is ignored.
        """
        # img = img.astype(np.float32)
        dist = np.zeros(4, dtype=float) if dist is None else dist
        points = scale * np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
        axis_points, _ = cv2.projectPoints(points, rvec, tvec, intrinsic_matrix, dist)
        return axis_points
