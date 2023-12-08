# Environment to play with the robotic hand for having a fingertip
# based action space 
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch 
from isaacgym.torch_utils import *

import math 
import numpy as np 
import torch

from .sim_env import SimulationEnv

class FingertipMotionEnv(SimulationEnv):
    def __init__(self, asset_root): 
        super().__init__(asset_root=asset_root)

        self.set_home_state()
        self.endeff_state = self.get_endeff_state()
        self.viewer = None

     # Set Home state
    def set_home_state(self):  
        self.home_state = torch.tensor([-0.00137183, -0.22922094, 0.7265581, 0.79128325,
                                        0.9890924, 0.37431374, 0.36866143, 0.77558154,
                                        0.00662423, -0.23064502, 0.73253167, 0.7449019, 
                                        0.08261403, -0.15844858, 0.82595366, 0.7666822 ]) 

    # This Function is used for resetting the Environment
    def reset(self):
        # Reset
        self.set_hand_position(self.home_state)  
        self.set_endeff_position(self.endeff_state[:7]) 
        
        # Code For Simulating and Stepping Graphics
        self.simulate_and_render()
                
        # Get Observation
        obs = {}
        obs['pixels'] = self.compute_observation(obs_type = 'image') 
        obs['features'] = self.compute_observation(obs_type = 'position')
        print('AFTER RESET OBS: {} | {}'.format(obs['pixels'].shape, obs['features'].shape))
        
        return obs

    # Step Function - TODO
    def step(self, action): # NOTE: This code piece is for moving the robot with positions

        # Set the action position to the desired ones
        action = to_torch(action, dtype=torch.float, device='cpu') 

        self.set_hand_position(action[:-7])
        self.set_endeff_position(action[-7:])
        
        # Simulate and render
        self.simulate_and_render()
        
        # Compute observations
        obs = {}
        obs['pixels'] = self.compute_observation(obs_type = 'image') 
        obs['features'] = self.compute_observation(obs_type = 'position')
        print('AFTER STEP OBS: {} | {}'.format(obs['pixels'].shape, obs['features'].shape))

        reward, done, infos = 0, False, {'is_success': False} 
        
        return obs, done, reward, infos
        
        # self.nextstate=self.compute_observation(observation='image')
        # self.nextposition = self.compute_observation(observation='position')
        
        # self.obs={}
        
        # self.obs['pixels']=self.nextstate
        # self.obs['features']=self.nextposition                
        
        
        # # self.gym.clear_lines(self.viewer)
        # # self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # #print("Done")

        # return self.obs,self.done,self.reward, infos

    
    def simulate_and_render(self): 
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)   
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)