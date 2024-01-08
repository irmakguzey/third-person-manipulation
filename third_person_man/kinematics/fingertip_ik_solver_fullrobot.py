import matplotlib.pyplot as plt
import numpy as np
import os

from copy import deepcopy as copy 
from klampt import WorldModel, IKSolver
from klampt.model import ik
from klampt.model.subrobot import SubRobotModel
from klampt.math import se3
from pathlib import Path
from tqdm import tqdm 
from scipy.spatial.transform import Rotation

# from third_person_man.utils import turn_frames_to_homo, turn_homo_to_frames

class FingertipChain:
    def __init__(self, robot_model, base_link_name, tip_link_name, base_position=[0, 0, 0, 0.5, 0.5, 0.5, 0.5]): 
        self.robot = robot_model 
        self.base_link = self.robot.link(base_link_name)
        self.tip_link = self.robot.link(tip_link_name)

        # self.subrobot = SubRobotModel(
        #     robot = robot_model,
        #     links = self.get_beginning_joint_indices()
        # )

        # self.tip_link = self.subrobot.link(tip_link_name)
        # self.robot = self.subrobot

        # print('self.subrobot.getConfig: {}'.format(self.subrobot.getConfig()))

        # print('self.subrobot.getJacobian: {}'.format(self.subrobot.link(3).getPositionJacobian([0,0,0])))

        self.num_links = len(self.get_joint_indices())
        print('Chain initiated number of links: {}'.format(self.num_links))

        self.set_base_position(base_position = np.asarray(base_position)) # For now we are not going to change the base position

    def get_current_position(self): 
        _, current_tvec = self.tip_link.getTransform()
        return current_tvec 
    
    # def get_tip_transform(self):

    def get_current_pose(self): 
        from third_person_man.utils import turn_frames_to_homo
        current_rvec, current_tvec = self.tip_link.getTransform()
        current_rvec = np.asarray(current_rvec).reshape(3,3)
        current_pose = turn_frames_to_homo(
            rvec = current_rvec, 
            tvec = current_tvec
        )
        return current_pose 
    
    def get_current_orientation(self): 
        current_rvec, _ = self.tip_link.getTransform()
        return np.asarray(current_rvec).reshape(3,3)
    
    def get_current_state(self, compute_type='position'):
        if compute_type == 'position': return self.get_current_position()
        if compute_type == 'orientation': return self.get_current_orientation() 
        if compute_type == 'all': return self.get_current_pose()

    def get_jacobian(self, compute_type='position'):
        # Get the whole jacobian of the tip link\
        # tip_jacobian: 6xn (n: number of DOFs) jacobian matrix
        if compute_type == 'position': return self.get_position_jacobian()
        if compute_type == 'orientation': return self.get_orientation_jacobian() 
        if compute_type == 'all': return self.get_pose_jacobian()        
    
    def get_pose_jacobian(self): # Returns a 6 dimensional jacobian
        return self.tip_link.getJacobian([0,0,0]) # plocal is 0 wrt the link 

    def get_position_jacobian(self): # 3 dimenisional jacobian 
        # tip_jacobian: 3xn (n: number of DOFs) position jacobian matrix
        return self.tip_link.getPositionJacobian([0,0,0])

    def get_orientation_jacobian(self): # 3 dimensional jacobian
        # tip_jacobian: 3xn (n: number of DOFs) orientation jacobian matrix
        return self.tip_link.getOrientationJacobian()
    
    # Returns error in position vectors and axis angles
    def get_current_error(self, desired, error_type='position'):
        if error_type == 'position':
            current_position = self.get_current_position()
            transform = desired - current_position
            return transform
        
        if error_type == 'orientation':
            current_orientation = self.get_current_orientation() # 3,3 rotation matrix
            error = np.linalg.pinv(current_orientation) @ desired 
            # Turn this transform to axis angle
            from third_person_man.utils import quat2axisangle
            error_axis_angle = quat2axisangle(Rotation.from_matrix(error).as_quat())
            return error_axis_angle
        
        if error_type == 'all':
            current_pose = self.get_current_pose()
            transform = np.linalg.pinv(current_pose) @ desired 

            from third_person_man.utils import turn_homo_to_frames, quat2axisangle
            rvec, tvec = turn_homo_to_frames(transform)
            rvec_axis_angle = quat2axisangle(Rotation.from_matrix(rvec).as_quat())
            error = np.concatenate([rvec_axis_angle, tvec], axis=0) # Jacobian returns the orientation first and then position
            return error

    def set_base_position(self, base_position):
        from third_person_man.utils import quat2axisangle
        
        # base_position: [7,] [position + quat] of the base_link
        # NOTE: If this won't work, should first send things wrt the base position and not the world
        endeff_position = base_position[:3]
        endeff_axis_angle = Rotation.from_quat(base_position[3:]).as_euler('zyx')
        # endeff_axis_angle = quat2axisangle(quat = base_position[3:]) # Rotation.from_quat(base_position[3:]).as_euler('zxy') # NOTE: Again - so not sure of this...
        endeff_config = np.concatenate([endeff_position, endeff_axis_angle], axis=0)

        # Get the robot config 
        robot_config = self.robot.getConfig()

        # Traverse through the base link's joints and set them accordingly 
        for i in range(self.base_link.getIndex()):
            robot_config[i] = endeff_config[i]

        # Set the config 
        self.robot.setConfig(robot_config)
    

    def set_joint_positions(self, joint_positions):
        # Will get the joint positions and set the config of the robot accordingly
        # Will traverse through the chain until it finds the base link

        # Get the joint indices of the chain 
        joint_indices = self.get_joint_indices()

        # Get the robot config 
        robot_config = self.robot.getConfig() 

        # Traverse through the config and the indices and change it
        for desired_joint_id, actual_joint_id in enumerate(joint_indices): 
            desired_joint_pos = joint_positions[desired_joint_id]
            robot_config[actual_joint_id] = desired_joint_pos

        # Set the config 
        self.robot.setConfig(robot_config)

    def get_joint_positions(self):
        # Get the joint positions from robot
        joint_positions = []

        # Get the joint indices of the chain
        joint_indices = self.get_joint_indices()

        # Get the robot config to traverse
        robot_config = self.robot.getConfig()

        # Traverse through them and get the positions
        for joint_id in joint_indices:
            joint_positions.append(
                robot_config[joint_id]
            )

        return np.asarray(joint_positions)

    # Returns the indices of the joints of the chain
    # These indices can be used to set and get the joint positions 
    def get_joint_indices(self):
        joint_indices = [] # NOTE: Joint indices for the whole robot model
        parent_link = None 
        while True: 
            if parent_link is None:
                current_link = self.tip_link 
            else:
                current_link = parent_link 

            if current_link.getName() == 'base_link': # We don't want to go further than the base_link
                break

            # Get the id if the joint is not fixed 
            joint_type = self.robot.getJointType(current_link.getName())
            if joint_type == 'normal':
                joint_id = current_link.getIndex()
                joint_indices.append(
                    joint_id
                )

            # Get the parent id
            parent_id = current_link.getParent() 
            if parent_id == -1: break 
            parent_link = self.robot.link(parent_id)

        # Revert the joint indices 
        joint_indices.reverse()

        return joint_indices

        # return list(range(4))
    
    def get_beginning_joint_indices(self):
        joint_indices = [] # NOTE: Joint indices for the whole robot model
        parent_link = None 
        while True: 
            if parent_link is None:
                current_link = self.tip_link 
            else:
                current_link = parent_link 

            if current_link.getName() == 'base_link': # We don't want to go further than the base_link
                break

            # Get the id if the joint is not fixed 
            joint_type = self.robot.getJointType(current_link.getName())
            if joint_type == 'normal':
                joint_id = current_link.getIndex()
                joint_indices.append(
                    joint_id
                )

            # Get the parent id
            parent_id = current_link.getParent() 
            if parent_id == -1: break 
            parent_link = self.robot.link(parent_id)

        # Revert the joint indices 
        joint_indices.reverse()

        return joint_indices
    
    # Have the single jacobian step
    def single_jacobian_inverse_step(self, desired, compute_type='position'):
        # compute_type: (position, orientation, all)

        # Get the current error
        x_e = self.get_current_error(desired=desired, error_type=compute_type)

        # Get the current jacobian and the necessary change
        current_jacobian = self.get_jacobian(compute_type=compute_type)
        # print('current_jacobian: {}, x_e: {}'.format(current_jacobian, x_e))
        all_joints_pos_change = np.linalg.pinv(current_jacobian) @ x_e
        # print('all_joints_pos_change: {}'.format(all_joints_pos_change))

        # Return the change in our current joint
        joint_indices = self.get_joint_indices()
        # print('joint_indices: {}'.format(joint_indices))
        joint_pos_change = []
        for joint_id in joint_indices:
            joint_pos_change.append(all_joints_pos_change[joint_id])

        # print('np.asarray(joint_pos_change): {}'.format(
        #     np.asarray(joint_pos_change)
        # ))
        return np.asarray(joint_pos_change)

    
    def inverse_kinematics(self, desired, threshold=1e-3, max_iterations=1000, compute_type='position', pbar=None): 
        # Will return the relative change and the calculated joint angles
        robot_config = copy(self.robot.getConfig())
        curr_joint_positions = self.get_joint_positions()
        delta_joint_positions = np.zeros(self.num_links)
        
        errors = []

        learning_rate = 1 # NOTE: Might change this afterwards
        iteration_joint_positions = [] # This is for debugging

        # Get the jacobians with objectives
        for i in range(max_iterations): 
            # Calculate the 
            curr_joint_pos_change = self.single_jacobian_inverse_step(
                desired = desired,
                compute_type = compute_type 
            )
            
            # Add the current position change in the joints
            curr_joint_positions += learning_rate * curr_joint_pos_change
            delta_joint_positions += learning_rate * curr_joint_pos_change
            iteration_joint_positions.append(curr_joint_positions)

            # Set the joint positions to have the forward kinematics and the jacobian
            self.set_joint_positions(curr_joint_positions)

            # Calculate the error to break if it's found
            curr_error = self.get_current_error(desired = desired, error_type = compute_type)
            errors.append(curr_error)

            if not pbar is None: # Means debugs
                pbar.update(1)
                pbar.set_description('Iteration: {}, Error: {}'.format(i, curr_error))

            if (np.abs(curr_error) < threshold).all():
                break

        errors = np.asarray(errors)
        iteration_joint_positions = np.stack(iteration_joint_positions, axis=0) # Shape: (1000, 4)
        # if len(errors) == 1: 
        #     print('errors: {}'.format(errors))
        return curr_joint_positions, delta_joint_positions, pbar, errors, iteration_joint_positions

class FingertipIKSolver:
    def __init__(self, urdf_path, desired_finger_types):
        # Initialize the world and the robot
        self.world = WorldModel()
        self.robot = self.world.loadRobot(
            fn = urdf_path
        )

        urdf_fingertip_mappings = dict(
            index = 'link_3.0_tip',
            middle = 'link_15.0_tip',
            ring = 'link_7.0_tip',
            thumb = 'link_11.0_tip'
        )

        self.fingertip_link_mappings = {} 
        for finger_type in desired_finger_types:
            self.fingertip_link_mappings[finger_type] = urdf_fingertip_mappings[finger_type]

        # Create the chains
        self.chains = {}
        for finger_type, finger_link_name in self.fingertip_link_mappings.items():
            self.chains[finger_type] = FingertipChain(
                robot_model = self.robot, 
                base_link_name = 'base_link',
                tip_link_name = finger_link_name,
                # base_position = np.zeros(7) 
            )

    def move_to_pose(self, poses, demo_action=None): 
        from third_person_man.utils import turn_homo_to_frames
        # poses: 4 fingertip poses as homogenous matrices: (4, 4, 4,) - poses will always come with all the fingers on it
        # assumption is that indexing: 0: index, 1: middle, 2: ring, 3: thumb
        # if demo_hand_action is not None it means that we will only apply IK on
        # some of the fingers

        # Here are the instructions for the method
        # Traverse through the poses
        # For each pose find the required angles to apply using inverse kinematics
        # Get the joint angles and return them stacked - should only use the hand for now

        joint_positions = []
        iteration_joint_positions = []
        errors = []
        for i, finger_type in enumerate(['index', 'middle', 'ring', 'thumb']):
            desired_pose = poses[i] # For now this is wrt the world, so we should set the base position

            # NOTE: For now we are only testing position
            if finger_type in self.fingertip_link_mappings.keys(): # Will check the keys
                _, desired_tvec = turn_homo_to_frames(matrix = desired_pose)

                # current_tvec = self.chains[finger_type].get_current_position()
                # print('current_tvec from ik: {}, current_tvec from fingertip_position: {}, desired_tvec: {}'.format(
                #     current_tvec, desired_tvec))

                # # Get the joint positions
                # current_state = self.chains[finger_type].get_joint_positions()


                # _, base_position = self.robot.link('base_link').getTransform()
                # print('base_position: {}'.format(base_position))

                finger_joint_positions, _, _, finger_errors, finger_iteration_joint_positions = self.chains[finger_type].inverse_kinematics(
                    desired = desired_tvec,
                    compute_type = 'position',
                    threshold = 1e-3,
                )
            else: # Just get it from the demo_hand_action
                finger_joint_positions = np.zeros(4)
                finger_iteration_joint_positions = np.zeros((1000, 4))
                finger_errors = np.zeros((1000, 3))

            # We should make pad iteration joint positions to reach the max iteration number 
            if finger_iteration_joint_positions.shape[0] < 1000:
                # Pad errors and iteration joint positions
                finger_iteration_joint_positions = np.pad(
                    finger_iteration_joint_positions,
                    ((0, 1000 - finger_iteration_joint_positions.shape[0]), (0,0)),
                    'constant',
                    constant_values=((0,finger_iteration_joint_positions[-1,:]), (0,0)))

                finger_errors = np.pad(
                    finger_errors,
                    ((0, 1000 - finger_errors.shape[0]), (0,0)),
                    'constant',
                    constant_values=((0,finger_errors[-1,:]), (0,0)))

            joint_positions.append(finger_joint_positions)
            iteration_joint_positions.append(finger_iteration_joint_positions) # (4, 1000, 4)
            errors.append(finger_errors)

        joint_positions = np.concatenate(joint_positions, axis=0)
        iteration_joint_positions = np.concatenate(iteration_joint_positions, axis=1) # Concatenate in the finger positions
        errors = np.stack(errors, axis=0)

        endeff_position = [0.2, 1.5, 0.0, 0, 0, 0.7071068, 0.7071068 ]
        return joint_positions, endeff_position, errors, iteration_joint_positions
    
    def set_positions(self, joint_positions, endeff_position): # This is removed from the eq for now

        # Set these for each of the chains separately
        for i, finger_type in enumerate(['index', 'middle', 'ring', 'thumb']): 
            if finger_type in self.fingertip_link_mappings.keys():
                finger_joint_pos = joint_positions[4*i:4*(i+1)]
                self.chains[finger_type].set_joint_positions(joint_positions=finger_joint_pos)
            # self.chains[finger_type].set_base_position(base_position = endeff_position)
