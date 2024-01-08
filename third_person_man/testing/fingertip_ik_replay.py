# Script to: 
# 1) Load a demo
# 2) Calculate the fingertip poses of that demo through timesteps
# 3) Try to replay that demo but with IK solver applying the actions

import numpy as np 
np.set_printoptions(precision=2, suppress=True)

import cv2
import os
import glob 
import hydra 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

from pathlib import Path
from tqdm import tqdm

from scipy.spatial.transform import Rotation

from third_person_man.testing import FingertipReplay
from third_person_man.kinematics import FingertipIKSolver


class FingertipIKReplay(FingertipReplay):
    def __init__(self,
                 data_path,
                 demo_num,
                 representations,
                 env_cfg,
                 module_name='fingertip_ik_replay',
                 desired_finger_types=['index', 'middle', 'ring', 'thumb']
    ): 
        super().__init__(
            data_path=data_path,
            demo_num=demo_num,
            representations=representations,
            env_cfg = env_cfg,
            module_name=module_name,
        )

        from third_person_man.utils import get_path_in_package
        self.solver = FingertipIKSolver(
            urdf_path = get_path_in_package('kinematics/assets/allegro_hand_right.urdf'),
            desired_finger_types=desired_finger_types
        )
        self.desired_finger_types = desired_finger_types

        # Set the initial positions
        obs = self.env.reset()

        self.solver.set_positions(
            joint_positions = obs['features'][:-7],
            endeff_position = np.zeros(7) # For now we are going to ignore the endeffector position 
        ) 

    def save_timestep(self, state_id):
        # Get the action from the data
        demo_id, action_id = self.data['hand_actions']['indices'][state_id]
        demo_hand_action = self.data['hand_actions']['values'][demo_id][action_id]
        demo_action = np.concatenate([demo_hand_action, np.zeros(7)], axis=0)

        # Calculate the fingertip poses wrt to the origin
        action_fingertip_poses, projected_fingertip_poses = self.env.get_action_fingertip_poses(
            features = demo_action,
            endeff_pose = 'home_pose',
            return_projected_poses = True)
        
        action_fingertip_poses = self.env.calculate_fingertip_positions(features = demo_action) # NOTE: This is only for testing

        # Find the necessary actions through the IK solver
        hand_action, _, errors, _, _ = self.solver.move_to_pose(
            poses = action_fingertip_poses,
            demo_action = demo_action)

        # Apply these actions to the environment
        action = np.concatenate([hand_action, np.zeros(7)], axis=0) # NOTE: This is for now
        obs, _, _, _ = self.env.step(action)
        self.record_fingertip_poses(
            obs = obs,
            desired_fingertip_poses = projected_fingertip_poses,
            errors = errors,
            state_id = state_id # For debugging purposes
        )

        # Apply the final positions from the environment
        # on the IK solver as well
        self.solver.set_positions(
            joint_positions = obs['features'][:-7],
            endeff_position = obs['features'][-7:] 
        ) 
        # for finger_id, finger_type in enumerate(['index', 'middle', 'ring', 'thumb']):
        #     if not finger_type in self.desired_finger_types:
        #         demo_action[finger_id*4:(finger_id+1)*4] = np.zeros(4) # Send zeros if a finger is not in the used fingertips
        # self.env.step(demo_action) # Move everything back to the demo so that we could experiment on it one by one
        # self.solver.set_positions(
        #     joint_positions = demo_action[:-7],
        #     endeff_position = demo_action[-7:]
        # )

    def record_fingertip_poses(self, obs, desired_fingertip_poses=None, errors=None, state_id=None): # TODO: Delete this and use one from the other replay module
        # Get the transformed fingertip positions
        fingertips_2d = self.env.get_projected_fingertip_positions(
            obs['features'],
            endeff_pose = 'current_pose')
        fingertips_2d_used = []
        for finger_id, finger_type in enumerate(['index', 'middle', 'ring', 'thumb']):
            if finger_type in self.desired_finger_types:
                fingertips_2d_used.append(fingertips_2d[finger_id])

        # Plot the fingertips on top of the image
        img = np.multiply(
            np.ones(np.transpose(obs['pixels'], (1,2,0)).shape, np.uint8),
            np.transpose(obs['pixels'], (1,2,0))
        )
        img = self.plot_axes(axes=fingertips_2d_used, img=img)

        # Plot desired fingertip features
        if not desired_fingertip_poses is None:
            desired_ft_poses_used = []
            for finger_id, finger_type in enumerate(['index', 'middle', 'ring', 'thumb']):
                if finger_type in self.desired_finger_types:
                    desired_ft_poses_used.append(desired_fingertip_poses[finger_id])
            img = self.plot_axes(axes=desired_ft_poses_used, img=img, color_set=2)

        # Plot the end effector position - not anymore
        # eef_2d = self.env.get_projected_endeffector_position(
        #     endeff_pose = 'current_pose'
        # )
        # img = self.plot_axes(axes=eef_2d, img=img, color_set=2)

        # Get the image and write to the video recorder
        self.video_recorder.record(
            obs = np.transpose(img, (2,0,1))
        )

        # Plot the errors and concatenate to the video
        error_plt_dir = '/home/irmak/Workspace/third-person-manipulation/third_person_man/testing/outs/fingertip_ik_replay/error_plots'
        finger_types = ['index', 'middle', 'ring', 'thumb']
        fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(10,10))
        for finger_id in range(len(errors)):
            if finger_types[finger_id] in self.desired_finger_types: # Set the finger 
                axs[finger_id, 0].set_ylabel(f'{finger_types[finger_id]} - USED')
            else: 
                axs[finger_id, 0].set_ylabel(f'{finger_types[finger_id]} - NA')

            for axis_id in range(errors[finger_id].shape[1]):
                if finger_id == 0: # Set the title
                    axs[finger_id, axis_id].set_title('Axis: {}'.format(axis_id))

                if len(errors[finger_id][:, axis_id]) == 1: # Plot the errors
                    axs[finger_id, axis_id].plot(range(5), np.zeros(5))
                else: 
                    axs[finger_id, axis_id].plot(np.abs(errors[finger_id][:, axis_id]))
                    
        
        plt.savefig(f'{error_plt_dir}/errors_plot.png',
                    bbox_inches='tight')
        fig.clf()
        plt.close()

        # Dump the image as well
        plt.axis('off')
        plt.imsave(f'{error_plt_dir}/errors_hand.png', img)

        # Read and concat the images
        from third_person_man.utils import concat_imgs
        recorded_img = concat_imgs(
            img1 = cv2.imread(f'{error_plt_dir}/errors_hand.png'), 
            img2 = cv2.imread(f'{error_plt_dir}/errors_plot.png'),
            orientation = 'horizontal'
        )

        cv2.imwrite(f'{error_plt_dir}/visualization/state_{str(state_id).zfill(2)}.png', recorded_img)
        

    def save_trajectory(self, title='fingertip_ik_replay.mp4'):
        obs = self.env.reset()
        self.video_recorder.init(obs = obs['pixels'])

        pbar = tqdm(total=len(self.data['hand_actions']['indices']))

        for state_id in range(len(self.data['hand_actions']['indices'])):
            self.save_timestep(state_id)
            pbar.update(1)
            pbar.set_description(f'State ID: {state_id}')

        pbar.close()
        self.video_recorder.save(title)

        from third_person_man.utils import turn_images_to_video
        turn_images_to_video(
            viz_dir = '/home/irmak/Workspace/third-person-manipulation/third_person_man/testing/outs/fingertip_ik_replay/error_plots/visualization',
            video_fps = 10,
            video_name = title
        )
