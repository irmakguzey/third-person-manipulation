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

# from third_person_man.utils import get_yaml_data, get_path_in_package

class FingertipIKReplay(FingertipReplay):
    def __init__(self, data_path, demo_num, representations, env_cfg, module_name='fingertip_ik_replay'): 
        super().__init__(
            data_path=data_path,
            demo_num=demo_num,
            representations=representations,
            env_cfg = env_cfg,
            module_name=module_name
        )

        # TODO: Take this path from the env?
        self.solver = FingertipIKSolver(
            urdf_path = '/home/irmak/Workspace/third-person-manipulation/models/allegro_hand_description/urdf/model_only_hand.urdf'
        )

        # Set the initial positions
        obs = self.env.reset()

        self.solver.set_positions(
            joint_positions = obs['features'][:-7],
            endeff_position = obs['features'][-7:] 
        ) 

        # Print the obs and robot config positions
        print('Obs Features: {}, Robot Config Features: {}'.format(
            obs['features'], self.solver.get_positions()
        ))

    def save_timestep(self, state_id):
        # Get the action from the data
        demo_id, action_id = self.data['hand_actions']['indices'][state_id]
        demo_hand_action = self.data['hand_actions']['values'][demo_id][action_id]
        demo_action = np.concatenate([demo_hand_action, np.zeros(7)], axis=0)

        # Calculate the fingertip poses wrt to the origin
        action_fingertip_poses = self.env.calculate_fingertip_positions_wrt_world(
            features = demo_action,
            endeff_pose = 'home_pose')

        # Find the necessary actions through the IK solver
        hand_action, endeff_action = self.solver.move_to_pose(poses = action_fingertip_poses)

        # Apply these actions to the environment
        action = np.concatenate([hand_action, endeff_action], axis=0)
        # action = np.concatenate([hand_action, np.zeros(7)], axis=0) # NOTE: This is for now
        obs, _, _, _ = self.env.step(action)
        self.record_fingertip_poses(obs = obs, desired_fingertip_features = demo_action)

        # Apply the final positions from the environment
        # on the IK solver as well
        self.solver.set_positions(
            joint_positions = obs['features'][:-7],
            endeff_position = obs['features'][-7:] 
        ) 

    def record_fingertip_poses(self, obs, desired_fingertip_features=None): # TODO: Delete this and use one from the other replay module
        # Get the transformed fingertip positions
        fingertips_2d = self.env.get_projected_fingertip_positions(
            obs['features'],
            endeff_pose = 'current_pose')

        # Plot the fingertips on top of the image
        # img = np.transpose(np.float32(obs['pixels']) / 255., (1,2,0))
        img = np.multiply(
            np.ones(np.transpose(obs['pixels'], (1,2,0)).shape, np.uint8),
            np.transpose(obs['pixels'], (1,2,0))
        )
        img = self.plot_axes(axes=fingertips_2d, img=img)

        # Plot desired fingertip features
        if not desired_fingertip_features is None:
            desired_fingertips_2d = self.env.get_projected_fingertip_positions(
                desired_fingertip_features,
                endeff_pose = 'current_pose')
            img = self.plot_axes(axes=desired_fingertips_2d, img=img, color_set=2)

        # Get the image and write to the video recorder
        self.video_recorder.record(
            obs = np.transpose(img, (2,0,1))
        )

    def save_trajectory(self, title='fingertip_ik_replay.mp4'):
        obs = self.env.reset()
        self.video_recorder.init(obs = obs['pixels'])

        pbar = tqdm(total=len(self.data['hand_actions']['indices']))

        for state_id in range(len(self.data['hand_actions']['indices'])):
            self.save_timestep(state_id)
            pbar.update(1)
            pbar.set_description(f'State ID: {state_id}')

            if state_id > 5: 
                break

        self.video_recorder.save(title)
        pbar.close()