# from klampt import WorldModel, IKSolver
# from klampt.model import ik
# from klampt.math import se3

# Going to use IKPy
import numpy as np

from copy import deepcopy as copy 
from ikpy import chain
from scipy.spatial.transform import Rotation

from third_person_man.utils import get_yaml_data, get_path_in_package

class FingertipIKSolver:
    def __init__(self):

        # Get the urdf path
        urdf_path = get_path_in_package("kinematics/assets/allegro_hand_right.urdf")

        # Loading Allegro Hand configs
        self.hand_configs = get_yaml_data(get_path_in_package("kinematics/configs/allegro_info.yaml"))
        self.finger_configs = get_yaml_data(get_path_in_package("kinematics/configs/allegro_link_info.yaml"))

        # Parsing chains from the urdf file
        self.chains = {} 
        for finger in self.hand_configs['fingers'].keys():
            self.chains[finger] = chain.Chain.from_urdf_file(
                urdf_path,
                name = finger 
                # base_elements = [
                #     self.finger_configs['links_info']['base']['link'], 
                #     # self.finger_configs['links_info'][finger]['link']
                # ], 
                # name = finger
            )

        print('self.chains[index].links: {}'.format(self.chains['ring'].links))

        # Get the list of joint names
        joint_names = self.chains['index'].links  # Exclude the base link

        # Alternatively, you can iterate over the joints and get more information
        for joint_name, joint in zip(joint_names, self.chains['index'].joints):
            print(f"Joint Name: {joint_name}")
            print(f"Joint Type: {joint.type}")
            print(f"Joint Limits: {joint.limits}")
            print("\n")

    def _turn_homo_to_frame(self, homomatrix): 
        rvec = homomatrix[:3, :3]
        tvec = homomatrix[:3, 3]
        return rvec, tvec 
    
    def finger_forward_kinematics(self, finger_type, input_angles):
        # Checking if the number of angles is equal to 4
        if len(input_angles) != self.hand_configs['joints_per_finger']:
            print('Incorrect number of angles')
            return 

        # Checking if the input finger type is a valid one
        if finger_type not in self.hand_configs['fingers'].keys():
            print('Finger type does not exist')
            return
        
        # Clipping the input angles based on the finger type
        finger_info = self.finger_configs['links_info'][finger_type]
        for iterator in range(len(input_angles)):
            if input_angles[iterator] > finger_info['joint_max'][iterator]:
                input_angles[iterator] = finger_info['joint_max'][iterator]
            elif input_angles[iterator] < finger_info['joint_min'][iterator]:
                input_angles[iterator] = finger_info['joint_min'][iterator]

        # Padding values at the beginning and the end to get for a (1x6) array
        input_angles = list(input_angles)
        input_angles.insert(0, 0)
        input_angles.append(0)

        # Performing Forward Kinematics 
        output_frame = self.chains[finger_type].forward_kinematics(input_angles)
        return output_frame[:3, 3], output_frame[:3, :3]
    
    def finger_inverse_kinematics(self, finger_type, input_position, initial_position = None):
        # Checking if the input figner type is a valid one
        if finger_type not in self.hand_configs['fingers'].keys():
            print('Finger type does not exist')
            return
        
        if initial_position is not None:
            # Checking if the number of angles is equal to 4
            if len(initial_position) != self.hand_configs['joints_per_finger']:
                print('Incorrect seed array length')
                return 

            # Clipping the input angles based on the finger type
            finger_info = self.finger_configs['links_info'][finger_type]
            for iterator in range(len(initial_position)):
                if initial_position[iterator] > finger_info['joint_max'][iterator]:
                    initial_position[iterator] = finger_info['joint_max'][iterator]
                elif initial_position[iterator] < finger_info['joint_min'][iterator]:
                    initial_position[iterator] = finger_info['joint_min'][iterator]

            # Padding values at the beginning and the end to get for a (1x6) array
            initial_position = list(initial_position)
            initial_position.insert(0, 0)
            initial_position.append(0)

        output_angles = self.chains[finger_type].inverse_kinematics(
                target_position = input_position,
                initial_position = initial_position)
            

        # ))
        return output_angles[1:5]
    


    # def move_to_pose(self, poses): 
    #     # poses: 4 fingertip poses as homogenous matrices: (4, 4, 4,)
    #     # assumption is that indexing: 0: index, 1: middle, 2: ring, 3: thumb
    #     # pass 
    #     finger_indices = [
    #         'link_d_tip', # Index finger tip
    #         'link_h_tip', # Middle finger tip
    #         'link_l_tip', # Ring finger tip
    #         'link_p_tip'
    #     ]

    #     # Set the palm to its original place - NOTE: This is just a test
    #     endeff_rvec = Rotation.from_quat([0, 0, 0.7071068, 0.7071068]).as_matrix()
    #     endeff_tvec = [0.2, 1.5, 0.0]
    #     self.robot.link('palm_link').setTransform(
    #         R = endeff_rvec.ravel(), 
    #         t = endeff_tvec
    #     )
    #     # Add a constraint for the palm
    #     self.set_endeff_position(endeff_position=[0.2, 1.5, 0.0, 0, 0, 0.7071068, 0.7071068 ])

    #     # Get the current robot config to set it afterwards - aka we don't want to 
    #     # change the actual robot config
    #     curr_config = self.robot.getConfig()

    #     # Create a new solver for this 
    #     solver = IKSolver(self.robot)

    #     # Add objectives to the solver 
    #     for i,fingertip_pose in enumerate(poses):
    #         fingertip_rvec, fingertip_tvec = self._turn_homo_to_frame(fingertip_pose)
    #         fingertip_objective = ik.objective(
    #             body = self.robot.link(finger_indices[i]),
    #             R = fingertip_rvec.ravel(), 
    #             t = fingertip_tvec,
    #         )

    #         solver.add(fingertip_objective)

    #     print('Solving the objectives for fingers: {}'.format(
    #         solver.solve()
    #     ))

    #     robot_positions = self.get_positions()
    #     hand_actions, endeff_action = robot_positions[:16], robot_positions[-7:]

    #     # Set the old config of the robot 
    #     self.robot.setConfig(curr_config)

    #     # Return the actions
    #     return hand_actions, endeff_action

    # def get_positions(self): 
    #     # Get endeffector position
    #     endeff_rvec, endeff_tvec = self.robot.link('palm_link').getTransform()
    #     endeff_rvec = np.asarray(endeff_rvec).reshape(3,3) # The robot link returns (9,) array for rotation
    #     endeff_quat = Rotation.from_matrix(endeff_rvec).as_quat()
    #     endeff_position = np.concatenate([endeff_tvec, endeff_quat])

    #     # Get hand joint positions
    #     robot_config = self.robot.getConfig()
    #     hand_joint_positions = []
    #     for i in range(4):
    #         curr_finger_joint_positions = np.asarray(robot_config[6+5*i : 6+5*(i+1)-1])
    #         hand_joint_positions.append(
    #             curr_finger_joint_positions
    #         )
    #     hand_joint_positions = np.concatenate(hand_joint_positions, axis=0)

    #     positions = np.concatenate([hand_joint_positions, endeff_position], axis=0)

    #     return positions

    # def set_positions(self, endeff_position, joint_positions): 
    #     # Set joint positions
    #     self.set_joint_positions(joint_positions=joint_positions)

    #     # Set the endeff position
    #     self.set_endeff_position(endeff_position=endeff_position)

    # def set_joint_positions(self, joint_positions):
    #     robot_config = copy(self.robot.getConfig())
    #     for i in range(4): 
    #         robot_config[6+5*i : 6+5*(i+1)-1] = joint_positions[4*i : 4*(i+1)]
    #     self.robot.setConfig(robot_config)

    # def set_endeff_position(self, endeff_position):
    #     endeff_rvec = Rotation.from_quat(endeff_position[3:]).as_matrix()
    #     endeff_tvec = endeff_position[:3]
    #     ik.solve( # Calculate the inverse kinematics and move the object
    #         ik.objective(
    #             body = self.robot.link('palm_link'), 
    #             R = endeff_rvec.ravel(),
    #             t = endeff_tvec
    #         )
    #     )

    # def get_link_positions(self): 
    #     return np.asarray(self.robot.getConfig())

if __name__ == '__main__': 
    solver = FingertipIKSolver()
