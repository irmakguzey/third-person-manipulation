# Script to plot the trajectory of the fingertip with respect to the camera
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

from .trajectory_replay import TrajectoryReplay

class FingertipReplay(TrajectoryReplay):
    def __init__(self, data_path, demo_num, representations, env_cfg, module_name='fingertip_replay'): 
        super().__init__(
            data_path = data_path,
            demo_num = demo_num,
            representations = representations,
            env_cfg = env_cfg,
            module_name = module_name
        )


    def save_timestep(self, state_id): 
        # Get the action from the data
        demo_id, action_id = self.data['hand_actions']['indices'][state_id]
        hand_action = self.data['hand_actions']['values'][demo_id][action_id]
        
        # Just as a test we are adding mock actions
        state_div = state_id / len(self.data['hand_actions']['indices'])
        buffer = 0.5
        if state_div < 0.33:
            res = state_div / 0.33 * buffer
            res_x =  res if state_div < 0.165 else buffer-res
            res_y, res_z = 0, 0
        elif state_div < 0.66:
            res = (state_div-0.33) / 0.33 * buffer
            res_y = res if state_div < (0.33+0.165) else buffer-res
            res_x, res_z = 0, 0
        else: 
            res = (state_div-0.66) / 0.33 * buffer
            res_z = res if state_div < (0.66+0.165) else buffer-res
            res_x, res_y = 0, 0
        arm_action = np.asarray(
            [0.2-res_x, 1.5+res_y, 0.0-res_z, 0, 0, 0.7071068, 0.7071068 ]
        )
        residual_action = np.array([
            [-res_x],
            [res_y],
            [-res_z]
        ])
        print('residual_action: {}'.format(residual_action))
        action = np.concatenate([hand_action, arm_action], axis=0)
        # action = np.concatenate([hand_action, np.zeros(7)], axis=0)

        # Apply to the environment
        obs, _, _, _ = self.env.step(action)
        self.record_fingertip_poses(obs=obs)
        

    def record_fingertip_poses(self, obs):
        # Get the transformed fingertip positions
        fingertips_2d = self.env.get_projected_fingertip_positions(
            obs['features'],
            endeff_pose = 'current_pose'
        )

        # Plot the fingertips on top of the image
        # img = np.transpose(np.float32(obs['pixels']) / 255., (1,2,0))
        img = np.multiply(
            np.ones(np.transpose(obs['pixels'], (1,2,0)).shape, np.uint8),
            np.transpose(obs['pixels'], (1,2,0))
        )
        img = self.plot_axes(axes=fingertips_2d, img=img)

        # Plot the end effector position
        eef_2d = self.env.get_projected_endeffector_position(
            endeff_pose = 'current_pose'
        )
        img = self.plot_axes(axes=eef_2d, img=img, color_set=2)

        home_eef_2d = self.env.get_projected_endeffector_position(
            endeff_pose = 'home_pose'
        )
        img = self.plot_axes(axes=home_eef_2d, img=img, color_set=2)

        # Get the image and write to the video recorder
        self.video_recorder.record(
            obs = np.transpose(img, (2,0,1))
        )
    
    def plot_axes(self, axes, img, color_set=1):
        for axis in axes:
            axis = axis.astype(int)
            if color_set == 1:
                img = cv2.line(img, tuple(axis[3].ravel()), tuple(axis[0].ravel()), (255, 0, 0), 3)
                img = cv2.line(img, tuple(axis[3].ravel()), tuple(axis[1].ravel()), (0, 255, 0), 3)
                img = cv2.line(img, tuple(axis[3].ravel()), tuple(axis[2].ravel()), (0, 0, 255), 3)
        
            elif color_set == 2:

                img = cv2.line(img, tuple(axis[3].ravel()), tuple(axis[0].ravel()), (255, 165, 0), 3) # Orange
                img = cv2.line(img, tuple(axis[3].ravel()), tuple(axis[1].ravel()), (128, 128, 0), 3) # Green
                img = cv2.line(img, tuple(axis[3].ravel()), tuple(axis[2].ravel()), (138, 43, 226), 3) # Purple
        

        return img

    def save_trajectory(self, title='fingertip_replay.mp4'):
        obs = self.env.reset()
        self.video_recorder.init(obs = obs['pixels'])

        pbar = tqdm(total=len(self.data['hand_actions']['indices']))

        for state_id in range(len(self.data['hand_actions']['indices'])):
            self.save_timestep(state_id)
            pbar.update(1)
            pbar.set_description(f'State ID: {state_id}')

            # if state_id > 5: 
            #     break

        self.video_recorder.save(title)
        pbar.close()