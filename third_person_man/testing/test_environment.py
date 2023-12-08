# Script for testing this environment
import os 

os.environ['MESA_VK_DEVICE_SELECT'] = '10de:24b0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from third_person_man.environments import FingertipMotionEnv

def test_env_initialization(asset_root='/home/irmak/Workspace/third-person-manipulation/third_person_man/environments'):
    env = FingertipMotionEnv(asset_root=asset_root)
    return env

if __name__ == '__main__': 
    test_env_initialization()