# Script to randomly get data from the trajectory replay
# And ask the IK to solve and plot the multiple iterations

import cv2
import glob
import hydra
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np 
import os
import random 

from pathlib import Path
from tqdm import tqdm

from third_person_man.kinematics import FingertipIKSolver

np.set_printoptions(precision=3, suppress=True)

random.seed(10)

class FingertipIKTester: 
    def __init__(self,
                 data_path, 
                 demo_num,
                 representations, 
                 env_cfg,
                 num_of_examples, # Number of random datapoints to test
                 module_name='fingertip_ik_tester',
                 desired_finger_types=['index', 'middle', 'ring', 'thumb']):

        # Set the initial position on the envronment 
        # NOTE: For now we are going to use step and not reinitialization of the 
        # environment - if that doesn't work well we should try to see if we can do something else 

        # Initialize the environment 
        self.env = hydra.utils.instantiate(env_cfg) 

        from third_person_man.utils import get_path_in_package
        self.solver = FingertipIKSolver(
            urdf_path=get_path_in_package('kinematics/assets/allegro_hand_right.urdf'),
            desired_finger_types=desired_finger_types
        )
        self.desired_finger_types = desired_finger_types

        # Load the data
        from third_person_man.utils import load_data # Will give an error for the torch imports 
        roots = glob.glob(f'{data_path}/demonstration_*')
        self.data = load_data(roots=roots, demos_to_use=[demo_num], representations=representations)

        # Reset the environment
        _ = self.env.reset()

        # Set the variables 
        self.module_name = module_name 
        self.num_of_examples = num_of_examples
        # self.iteration_idxs = [1, 5, 10, 20, 50, 100, 150, 200, 500, 999] # Number of iterations that will be shown
        # self.iteration_idxs = [1, 5, 10, 20, 50, 100, 150]
        self.iteration_idxs = [0,1,2,5,9,10,20,50,99]
        # The last column will plot the error in all of the iterations 

        # Create the directory to plot 
        work_dir = os.path.dirname(__file__)
        self.out_dir = Path(work_dir) / f'outs/{module_name}'
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def sample_data_points(self):
        # Returns, two current data points

        # Sample random indices
        sample_idx = random.choices(range(len(self.data['hand_actions']['indices'])-1), k=self.num_of_examples) # Sample from -1 number of ids
        print('sample_idx: {}'.format(sample_idx))

        # Sample the data points - action and the image 
        data_points = []
        for sample_id in sample_idx:
            demo_id, action_id = self.data['hand_actions']['indices'][sample_id]
            current_state = self.data['hand_actions']['values'][demo_id][action_id]

            next_demo_id, next_action_id = self.data['hand_actions']['indices'][sample_id+1]
            assert demo_id == next_demo_id, 'There should only a single demo in this test'
            next_state = self.data['hand_actions']['values'][next_demo_id][next_action_id]

            data_points.append(
                np.stack([current_state, next_state], axis=0) # (2,16)
            )

        data_points = np.stack(data_points, axis=0) # (num_examples, 2, 16)

        return data_points, sample_idx
    
    def turn_data_point_to_states(self, data_point):
        # Gets a single data_point (2,16) and returns the next state
        # into a pose and sends the current state as is: 
        # current_state will be applied to the env and the IK solver directly
        # then iterations will be plotted

        current_state, next_state = data_point[0], data_point[1]
        next_pose = self.env.calculate_fingertip_positions(features = next_state)

        return current_state, next_pose, next_state

    # Gets a single datapoint, changes the IK solver accordingly
    # Gets the iteration joint positions
    # Steps and renders each step, returns the images and the total errors for the iterations
    def process_single_data_point(self, data_point, state_id):

        # Get the current state and the desired pose for this data_point
        current_state, next_pose, next_state = self.turn_data_point_to_states(data_point)
        
        # print('next_pose: {}'.format(next_pose))
        
        current_state_used, next_state_used = self.get_used_states(
            current_state=current_state, 
            next_state=next_state
        )

        # Get the iteration joint positions and the error
        # print('current_state_used: {}'.format(current_state_used))
        self.solver.set_positions(
            joint_positions = current_state_used, 
            endeff_position = None
        )

        # Get the current joint positions from the IK, compare it with the actual
        # current state
        print('row id: {}'.format(state_id))
        thumb_current_state_ik = self.solver.chains['thumb'].get_joint_positions()
        print('actual current_state: {}, IK current_state: {}'.format(
            current_state_used[-4:], np.asarray(thumb_current_state_ik)
        ))
        
        # Get the current poses from env and the IK, compare them 
        from third_person_man.utils import turn_homo_to_frames
        current_pose = self.env.calculate_fingertip_positions(features = current_state)
        _, current_tvec = turn_homo_to_frames(matrix = current_pose[-1])
        
        # thumb_tvec_ik = self.solver.chains['thumb'].get_current_pose()
        # print('actual thumb position: {}, IK thumb position: {}'.format(
        #     current_tvec, np.asarray(thumb_tvec_ik)
        # ))
        finger_poses = []
        for finger_type in ['index', 'middle', 'ring', 'thumb']:
            if finger_type in self.solver.chains.keys(): 
                finger_pose = self.solver.chains[finger_type].get_current_pose()
            else:
                finger_pose = np.zeros((4,4))
            finger_poses.append(finger_pose)

        finger_poses = np.stack(finger_poses, axis=0)

        # Step for the first timestep and plot the 
        first_img = self.plot_single_step(
            current_state = current_state_used, 
            next_state = next_state_used,
            ik_fingertip_poses = finger_poses
        )
        imgs = [first_img]

        
        # NOTE: We don't need to get the used poses here because the solver is set to use different 
        # fingers as well


        final_joint_position, _, errors, iteration_joint_positions, all_ik_fingertip_poses = self.solver.move_to_pose(
            poses = next_pose,
            env = self.env # For debugging
        )

        # Iterate in the iteration joint positions and plot them 
        for iteration_id in self.iteration_idxs:
            # Get the joint positions for that iteration
            if iteration_id > len(all_ik_fingertip_poses):
                iteration_id = -1
            iteration_joint_position = iteration_joint_positions[iteration_id]
            # print(iteration_joint_position.shape)
            iteration_used_state = self.get_used_state(state = iteration_joint_position)

            # Step and render the states 
            iteration_img = self.plot_single_step(
                current_state = iteration_used_state, 
                next_state = next_state_used,
                ik_fingertip_poses = all_ik_fingertip_poses[iteration_id]
            )
            # print('iteration_img.shape: {}'.format(iteration_img.shape))
            imgs.append(iteration_img)

        # Set the final state as well
        final_used_state = self.get_used_state(state = final_joint_position)
        final_img = self.plot_single_step(
            current_state = final_used_state, 
            next_state = next_state_used
        )
        imgs.append(final_img)

        print('----')
        return imgs, errors

    # Gets an axs and a row id, single data_point and plots the iterations of the IK solver
    def plot_single_data_point(self, axs, row_id, data_point):
        
        # Process the iterations and get the images / errors for that row
        row_imgs, row_errors = self.process_single_data_point(data_point = data_point, state_id=row_id)
        # print('len(row_imgs): {}, row_errors.shape: {}'.format(
        #     len(row_imgs), row_errors.shape
        # ))

        # Plot the images on the axs 
        for column_id in range(len(row_imgs)+3): # +1 is for errors

            # Set the titles of the axs
            if column_id > 0 and column_id < len(self.iteration_idxs)+1:
                axs[row_id, column_id].set_title(f'Iteration: {self.iteration_idxs[column_id-1]}')
            else:
                if column_id == 0: axs[row_id, column_id].set_title('First State')
                elif column_id == len(self.iteration_idxs)+1: axs[row_id, column_id].set_title('Last State')
            
            if column_id < len(row_imgs): # Plot the images
                axs[row_id, column_id].imshow(row_imgs[column_id])
                axs[row_id, column_id].get_xaxis().set_ticks([])
                axs[row_id, column_id].get_yaxis().set_ticks([])
            else: # Plot the errors for axes
                axis_id = column_id - len(row_imgs)
                axs[0, column_id].set_title(f'Errors in Axis: {axis_id}')
                finger_id = ['index', 'middle', 'ring', 'thumb'].index(self.desired_finger_types[0]) # Finger that is used only
                axs[row_id, column_id].plot(np.abs(row_errors[finger_id][:, axis_id]))

        return axs
    
    def plot_iterations(self, plot_name): # plot_name: Name of the plot to save

        # Create the axs to plot 
        _, axs = plt.subplots(
            nrows = self.num_of_examples, 
            ncols = len(self.iteration_idxs) + 5, # 2 for beginning and ending state, 3 for error axes
            figsize = (50,50)
        )

        # Sample datapoints to test 
        data_points, sample_idx = self.sample_data_points()
        print('data_points.shape: {}'.format(data_points.shape))

        pbar = tqdm(total = self.num_of_examples)

        for row_id in range(self.num_of_examples): 
            data_point = data_points[row_id]
            
            # Plot the row 
            axs = self.plot_single_data_point(
                axs = axs, 
                row_id = row_id, 
                data_point = data_point
            )

            axs[row_id, 0].set_ylabel('Sample ID: {}'.format(
                sample_idx[row_id]
            ))

            pbar.update(1)
            pbar.set_description('Plotted example: {}'.format(sample_idx[row_id]))


        pbar.close()

        # Dump the plot 
        plt.savefig(f'{self.out_dir}/{plot_name}.png', bbox_inches='tight')
        plt.close() 

    # Steps to the environment and plots the observation with the desired and the current fingertip poses
    def plot_single_step(self, current_state, next_state, ik_fingertip_poses=None):
        
        # Render the environment and get the first image 
        action = np.concatenate([current_state, np.zeros(7)], axis=0)
        obs, _, _, _ = self.env.step(action)
        # Get the transformed fingertip positions
        current_fingertip_poses = self.env.get_projected_fingertip_positions(
            obs['features'], # NOTE: Not sure if this is the case
            endeff_pose = 'current_pose')
        desired_fingertip_poses = self.env.get_projected_fingertip_positions(
            features = next_state,
            endeff_pose = 'current_pose' # We are not moving the end effector for now so everything should be wrt the current pose
        )
        if not ik_fingertip_poses is None:
            ik_fingertip_poses = self.env.get_projected_poses(
                poses = ik_fingertip_poses, 
                endeff_pose = 'current_pose'
            )

        img = self.record_fingertip_poses(
            obs = obs, 
            current_fingertip_poses = current_fingertip_poses,
            desired_fingertip_poses = desired_fingertip_poses,
            ik_fingertip_poses = ik_fingertip_poses
        )

        return img
    
    # Gets the joint positions with all the joint positions inputted - turns them into
    # 0s padded if the finger is not being used
    def get_used_states(self, current_state, next_state):
        # Make the current_state such that only desired fingers have actions on them
        current_state_used = []
        next_state_used = []
        for finger_id, finger_type in enumerate(['index', 'middle', 'ring', 'thumb']):
            if finger_type in self.desired_finger_types:
                current_state_used.append(current_state[finger_id*4:(finger_id+1)*4])
                next_state_used.append(next_state[finger_id*4:(finger_id+1)*4])
            else:
                current_state_used.append(np.zeros(4))
                next_state_used.append(np.zeros(4))

        current_state_used = np.concatenate(current_state_used, axis=0)
        next_state_used = np.concatenate(next_state_used, axis=0)

        return current_state_used, next_state_used
    
    def get_used_state(self, state):
        state_used = []
        for finger_id, finger_type in enumerate(['index', 'middle', 'ring', 'thumb']):
            if finger_type in self.desired_finger_types:
                state_used.append(state[finger_id*4:(finger_id+1)*4])
            else:
                state_used.append(np.zeros(4))

        state_used = np.concatenate(state_used, axis=0)

        return state_used


    # Gets a single fingertip pose and an observation and plots the desired 
    # and current fingertip poses on the image, returns the image
    def record_fingertip_poses(self,
                               obs,
                               current_fingertip_poses,
                               desired_fingertip_poses=None,
                               ik_fingertip_poses=None):
        
        from third_person_man.utils import plot_axes
        
        fingertips_2d_used = []
        for finger_id, finger_type in enumerate(['index', 'middle', 'ring', 'thumb']):
            if finger_type in self.desired_finger_types:
                fingertips_2d_used.append(current_fingertip_poses[finger_id])

        # Plot the fingertips on top of the image
        img = np.multiply(
            np.ones(np.transpose(obs['pixels'], (1,2,0)).shape, np.uint8),
            np.transpose(obs['pixels'], (1,2,0))
        )
        img = plot_axes(axes=fingertips_2d_used, img=img)

        # Plot desired fingertip features
        if not desired_fingertip_poses is None:
            desired_ft_poses_used = []
            for finger_id, finger_type in enumerate(['index', 'middle', 'ring', 'thumb']):
                if finger_type in self.desired_finger_types:
                    desired_ft_poses_used.append(desired_fingertip_poses[finger_id])
            img = plot_axes(axes=desired_ft_poses_used, img=img, color_set=2)

        if not ik_fingertip_poses is None:
            used_ik_ft_poses = []
            for finger_id, finger_type in enumerate(['index', 'middle', 'ring', 'thumb']):
                if finger_type in self.desired_finger_types:
                    used_ik_ft_poses.append(ik_fingertip_poses[finger_id])
            img = plot_axes(axes=used_ik_ft_poses, img=img, color_set=3)

        return img