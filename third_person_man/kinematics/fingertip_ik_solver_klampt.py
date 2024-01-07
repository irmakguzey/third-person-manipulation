from klampt import WorldModel, IKSolver
from klampt.model import ik
from klampt.math import se3

from copy import deepcopy as copy 

import numpy as np

from scipy.spatial.transform import Rotation

class FingertipIKSolver:
    def __init__(self, urdf_path):
        # Initialize the world and the robot
        self.world = WorldModel()
        # robot_id = self.world.loadRobot(fn = urdf_path)
        # self.robot = self.world.robot(robot_id)
        self.robot = self.world.loadRobot(
            fn = urdf_path
        )

        # print('self.robot.getLink(link15.0_tip): {}'.format(
        #     self.robot.link('link_15.0_tip')
        # ))


        # TODO: Set the beginning position of the end effector and the hand pose
        # and etc

        # NOTE: Config Info:
        # First 3: Translational axis
        # Next 3: Rotational axes 
        # Next 5: 4 knucle to tip revolute joints, 5th last fixed joint
        # ... tips go similarly

        # self.fingertip_links = [
        #     'link_d_tip', # Index finger tip
        #     'link_h_tip', # Middle finger tip
        #     'link_l_tip', # Ring finger tip
        #     'link_p_tip'
        # ]
        # self.base_link_name = 'palm_link'

        self.fingertip_link_mappings = dict(
            index = 'link_3.0_tip',
            middle = 'link_15.0_tip',
            ring = 'link_7.0_tip',
            thumb = 'link_11.0_tip'
        )
        # self.fingertip_links = [
        #     # 'link_3.0_tip', # Index
        #     'link_11.0_tip', # Middle
        #     # 'link_7.0_tip', # Ring
        #     # 'link_11.0_tip', # Thumb
        # ]
        # self.fingertip_links = [
        #     'link_3.0_tip', # Middle
        # ]
        self.base_link_name = 'base_link'

        print('self.robot.config: {}'.format(
            self.robot.getConfig()
        ))

    def _turn_homo_to_frame(self, homomatrix): 
        rvec = homomatrix[:3, :3]
        tvec = homomatrix[:3, 3]
        return rvec, tvec 

    def move_to_pose(self, poses): 
        # poses: 4 fingertip poses as homogenous matrices: (4, 4, 4,)
        # assumption is that indexing: 0: index, 1: middle, 2: ring, 3: thumb
        # pass 
        # Set the palm to its original place - NOTE: This is just a test
        # Add a constraint for the palm
        # self.set_endeff_position(endeff_position=[0.2, 1.5, 0.0, 0, 0, 0.7071068, 0.7071068 ])

        # Get the current robot config to set it afterwards - aka we don't want to 
        # change the actual robot config
        curr_config = self.robot.getConfig()

        # Create a new solver for this 
        solver = IKSolver(self.robot)

        # Add objectives to the solver 
        for i, finger_type in enumerate(['index', 'middle', 'ring', 'thumb']):
            fingertip_pose = poses[i]
            fingertip_rvec, fingertip_tvec = self._turn_homo_to_frame(fingertip_pose)
            
            fingertip_link_name = self.fingertip_link_mappings[finger_type]
            fingertip_objective = ik.objective(
                body = self.robot.link(fingertip_link_name),
                # R = fingertip_rvec.ravel(),
                # t = fingertip_tvec,
                local = [0, 0, 0],
                world = fingertip_tvec
            )

            solver.add(fingertip_objective)

            # Print the pose of each finger that is added
            self.print_parents(finger_type = finger_type)

            # Print the desired

        # Solve the objective with multiple restarts 
        self._solve_objective(
            solver = solver, debug_action = 'Solving for Fingertip Positions'
        )

        robot_positions = self.get_positions()
        hand_actions, endeff_action = robot_positions[:16], robot_positions[-7:]

        # Set the old config of the robot 
        self.robot.setConfig(curr_config)

        # Return the actions
        return hand_actions, endeff_action
    
    def print_parents(self, finger_type):
        # After motion get the fingertip positions
        # Get all the parent positions of whatever the link's finger is
        parent_link = None
        while True:
            if parent_link is None:
                current_link = self.robot.link(self.fingertip_link_mappings[finger_type])
            else:
                current_link = parent_link
            current_link_world_direction = current_link.getWorldDirection([0,0,1])
            # current_link_world_quat = Rotation.from_rotvec(current_link_world_direction).as_quat()
            
            print('current link name: {}, index: {}, world position: {}, current_link_world_direction: {}, robot_config[current_link.getIndex()]: {}'.format(
                current_link.getName(),
                current_link.getIndex(),
                np.asarray(current_link.getWorldPosition([0,0,0])),
                current_link_world_direction,
                self.robot.getConfig()[current_link.getIndex()]
            ))

            # if current_link.getName() == 'base_link':
            #     # For now we will only print until base_link
            #     break

            parent_id = current_link.getParent()
            if parent_id == -1:
                break 
            parent_link = self.robot.link(parent_id)

            # print('parent_link name: {}, index: {}, world position: {}'.format(
            #     parent_link.getName(), parent_link.getIndex(), np.asarray(parent_link.getWorldPosition([0,0,0]))
            # ))

        print('-----')

    # def get_fingertip_poses(self, relative_to_world=True):
    #     # relative_to_world, if not true will return relative to the palm link
    #     fingertip_poses = []
    #     for finger_link in self.finger_indices:

        
    #     if relative_to_world:



    def get_positions(self): 
        # Get endeffector position
        endeff_rvec, endeff_tvec = self.robot.link(self.base_link_name).getTransform()
        endeff_rvec = np.asarray(endeff_rvec).reshape(3,3) # The robot link returns (9,) array for rotation
        endeff_quat = Rotation.from_matrix(endeff_rvec).as_quat()
        endeff_position = np.concatenate([endeff_tvec, endeff_quat])

        # Get hand joint positions
        robot_config = self.robot.getConfig()
        # print('robot_config: {}'.format(np.asarray(robot_config)))
        hand_joint_positions = []
        for i in range(4):
            # print('6+5*i: {}, 6+5*(i+1)-1: {}'.format(
            #     6+5*i, 6+5*(i+1)-1
            # ))
            curr_finger_joint_positions = np.asarray(robot_config[6+5*i : 6+5*(i+1)-1])
            hand_joint_positions.append(
                curr_finger_joint_positions
            )
        hand_joint_positions = np.concatenate(hand_joint_positions, axis=0)
        print('from robot_config hand_joint_positions: {}'.format(hand_joint_positions))
        positions = np.concatenate([hand_joint_positions, endeff_position], axis=0)

        return positions

    def set_positions(self, endeff_position, joint_positions): 
        # Set joint positions
        self.set_joint_positions(joint_positions=joint_positions)

        # Set the endeff position
        self.set_endeff_position(endeff_position=endeff_position)

    def set_joint_positions(self, joint_positions):
        print('joint_positions: {}'.format(joint_positions))
        robot_config = copy(self.robot.getConfig())
        for i in range(4): 
            robot_config[6+5*i : 6+5*(i+1)-1] = joint_positions[4*i : 4*(i+1)]
        self.robot.setConfig(robot_config)

        print('after set_joint_positions, robot config: {}'.format(
            np.asarray(self.robot.getConfig())
        ))

    def set_endeff_position(self, endeff_position):
        endeff_rvec = Rotation.from_quat(endeff_position[3:]).as_matrix()
        endeff_tvec = endeff_position[:3]

        solver = IKSolver(self.robot)
        solver.add(
            ik.objective(
                body = self.robot.link(self.base_link_name), 
                R = endeff_rvec.ravel(),
                t = endeff_tvec
            )
        )

        self._solve_objective(
            solver = solver, debug_action='Moving the EEF to desired pose '
        ) # This usually gets solved in 0 restart

    def _solve_objective(self, solver, debug_action=None):
        numRestarts = 1000
        solved = False
        for i in range(numRestarts):
            solver.sampleInitial()
            solver.setMaxIters(1)
            solver.setTolerance(1e-5)
            res = solver.solve()
            if res:
                solved = True
                break

        if not debug_action is None:
            print('{}: {} in {} restart'.format(
                debug_action, solved, i+1
            ))

    def get_link_positions(self): 
        return np.asarray(self.robot.getConfig())

# def load_model_klampt(): # These are the tryouts with klampt
#     robot_urdf_path = '/home/irmak/Workspace/third-person-manipulation/models/allegro_hand_description/urdf/model_only_hand.urdf'

#     world = WorldModel()
#     robot = world.loadRobot(robot_urdf_path)

#     print(world.numRobots(), robot.getID(), robot.getName(), len(robot.getConfig()), robot.getJointLimits())

#     print(f'numlinks: {robot.numLinks()},\
#           one finger tip: {robot.link(self.base_link_name)}')
    

#     print('Pre IK Position: {}'.format(robot.link('link_h_tip').getTransform()))
#     for link_id in range(robot.numLinks()):
#         print('link ID: {}, link Name: {}, transform: {}'.format(
#             link_id, robot.link(link_id).getName(), robot.link(link_id).getTransform()
#         ))

#     # Example ik objective
#     move_tip_obj = ik.objective(
#         body = robot.link(self.base_link_name),
#         # R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
#         R = [0.6806479, -0.5827119,  0.4440330, 
#              0.6544765,  0.7560006, -0.0111195,
#              -0.3292097,  0.2981776,  0.8959414],
#         t = [-0.1, 2.0, 1.0]) # Old position: [0.0, 0.0, 0.1362]
    
#     print(ik.solve(move_tip_obj))

#     print('Post IK Position: {}'.format(robot.link(self.base_link_name).getTransform()))
#     for link_id in range(robot.numLinks()):
#         print('link ID: {}, link Name: {}, transform: {}'.format(
#             link_id, robot.link(link_id).getName(), robot.link(link_id).getTransform()
#         ))

#     print('robot.getConfig(): {}'.format(robot.getConfig()))


# if __name__ == '__main__': 
#     load_model_ikpy()
