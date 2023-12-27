# Script to replay a saved trajectory on the simulation environment and then save it to a video
import os
import glob 
import hydra 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

from tqdm import tqdm

from pathlib import Path

class TrajectoryReplay:
    def __init__(self, data_path, demo_num, representations, env_cfg, module_name='trajectory_replay'):

        # Initialize the environment required
        self.env = hydra.utils.instantiate(env_cfg) 

        # Load the data
        from third_person_man.utils import load_data # Will give an error for the torch imports 
        roots = glob.glob(f'{data_path}/demonstration_*')
        self.data = load_data(roots=roots, demos_to_use=[demo_num], representations=representations)
        self.state_id = 0

        obs = self.env.reset()
        # Initialize the video recorder
        from third_person_man.utils import VideoRecorder
        work_dir = os.path.dirname(__file__)
        self.video_recorder = VideoRecorder(
            save_dir = Path(work_dir) / f'outs/{module_name}',
            fps = 20
        )

        # Test if the initialization is proper
        plt.imshow(np.transpose(obs['pixels'], (1,2,0)))
        plt.savefig(f'{work_dir}/outs/{module_name}/env_reset.jpg')


    def save_timestep(self, state_id): 
        # Get the action from the data
        demo_id, action_id = self.data['hand_actions']['indices'][state_id]
        hand_action = self.data['hand_actions']['values'][demo_id][action_id]
        action = np.concatenate([hand_action, np.zeros(7)], axis=0)

        # Apply to the environment
        obs, _, _, _ = self.env.step(action)
        print(f'in save_timestep: obs.shape: {obs["pixels"].shape}')
        print('features: {}'.format(obs["features"]))

        # Get the image and write to the video recorder
        self.video_recorder.record(
            obs = obs['pixels']
        )

    def save_trajectory(self, title='cube_flipping_trajectory_endeff.mp4'):
        obs = self.env.reset()
        self.video_recorder.init(obs = obs['pixels'])

        pbar = tqdm(total=len(self.data['hand_actions']['indices']))

        for state_id in range(len(self.data['hand_actions']['indices'])):
            self.save_timestep(state_id)
            pbar.update(1)
            pbar.set_description(f'State ID: {state_id}')

        self.video_recorder.save(title)
        pbar.close()


@hydra.main(version_base=None, config_path='../../configs', config_name='testing')
def main(cfg) -> None: 
    cfg = cfg.trajectory_replay
    # env = hydra.utils.instantiate(cfg.env_cfg)
    # print('env: {}'.format(env))
    traj_replay = TrajectoryReplay(
        data_path = cfg.data_path, 
        demo_num = cfg.demo_num, 
        representations = cfg.representations, 
        env_cfg = cfg.env_cfg
    )
    
    traj_replay.save_trajectory()

if __name__ == '__main__': 
    main()



