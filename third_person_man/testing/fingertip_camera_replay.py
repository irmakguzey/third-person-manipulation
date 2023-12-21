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
    def __init__(self, data_path, demo_num, representations, env_cfg): 
        # Initialize the environment required
        self.env = hydra.utils.instantiate(env_cfg) 

        # Load the data
        from third_person_man.utils import load_data # Will give an error for the torch imports 
        roots = glob.glob(f'{data_path}/demonstration_*')
        self.data = load_data(roots=roots, demos_to_use=[demo_num], representations=representations)
        self.state_id = 0

        # Test if the initialization is proper
        obs = self.env.reset()

        # Initialize the video recorder
        from third_person_man.utils import VideoRecorder
        work_dir = os.path.dirname(__file__)
        self.video_recorder = VideoRecorder(
            save_dir = Path(work_dir) / 'outs/fingertip_replay',
        )

        plt.imshow(np.transpose(obs['pixels'], (1,2,0)))
        plt.savefig(f'{work_dir}/outs/fingertip_replay/reset.jpg')


    def save_timestep(self, state_id): 
        # Get the action from the data
        demo_id, action_id = self.data['hand_actions']['indices'][state_id]
        hand_action = self.data['hand_actions']['values'][demo_id][action_id]
        action = np.concatenate([hand_action, np.zeros(7)], axis=0)

        # Apply to the environment
        obs, _, _, _ = self.env.step(action)
        # print(f'in save_timestep: obs.shape: {obs["pixels"].shape}')
        # print('features: {}'.format(obs["features"]))

        # Get the transformed fingertip positions
        fingertips_2d = self.env.get_projected_fingertip_positions(obs['features'])
        
        work_dir = os.path.dirname(__file__)
        plt.imshow(np.transpose(obs['pixels'], (1,2,0)))
        plt.savefig(f'{work_dir}/outs/fingertip_replay/pre_fingertips_obs.jpg')

        # Plot the fingertips on top of the image
        # img = np.transpose(np.float32(obs['pixels']) / 255., (1,2,0))
        img = np.multiply(
            np.ones(np.transpose(obs['pixels'], (1,2,0)).shape, np.uint8),
            np.transpose(obs['pixels'], (1,2,0))
        )
        img = self.plot_fingertips(fingertips_2d=fingertips_2d, img=img)

        work_dir = os.path.dirname(__file__)
        plt.imshow(img)
        plt.savefig(f'{work_dir}/outs/fingertip_replay/post_fingertips_obs.jpg')

        # Get the image and write to the video recorder
        self.video_recorder.record(
            obs = np.transpose(img, (2,0,1))
        )

    def plot_fingertips(self, fingertips_2d, img):
        # print(img)
        # print('img.shape: {}'.format(img.shape))
        # print(img)
        for fingertip_pos in fingertips_2d:
            fingertip_pos = fingertip_pos.astype(int)
            img = cv2.line(img, tuple(fingertip_pos[3].ravel()), tuple(fingertip_pos[0].ravel()), (255, 0, 0), 3)
            img = cv2.line(img, tuple(fingertip_pos[3].ravel()), tuple(fingertip_pos[1].ravel()), (0, 255, 0), 3)
            img = cv2.line(img, tuple(fingertip_pos[3].ravel()), tuple(fingertip_pos[2].ravel()), (0, 0, 255), 3)
        return img


    def save_trajectory(self, title='fingertip_replay.mp4'):

        obs = self.env.reset()
        print('obs.shape: {}'.format(obs["pixels"].shape))
        print('obs.features after resetting: {}'.format(obs["features"]))
        self.video_recorder.init(obs = obs['pixels'])

        pbar = tqdm(total=len(self.data['hand_actions']['indices']))

        for state_id in range(len(self.data['hand_actions']['indices'])):
            # print(f'State ID: {state_id}')
            self.save_timestep(state_id)

            pbar.update(1)
            pbar.set_description(f'State ID: {state_id}')

        self.video_recorder.save(title)
        pbar.close()