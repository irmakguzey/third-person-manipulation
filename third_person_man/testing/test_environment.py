# Script for testing this environment
import os 

os.environ['MESA_VK_DEVICE_SELECT'] = '10de:24b0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from third_person_man.environments import FingertipMotionEnv

def test_env_initialization(asset_root='/home/irmak/Workspace/third-person-manipulation/third_person_man/environments'):
    env = FingertipMotionEnv(asset_root=asset_root)
    return env

def test_image(env): 
    # from PIL import Image
    # arr = env.render()
    # im = Image.fromarray(arr)
    # im.save("ex_image.jpg")
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    color_image = env.render()
    # print('color_image.shape: {}'.format(color_image.shape))
    plt.imshow(np.transpose(color_image, (1,2,0)) )
    plt.savefig('ex_image.jpg')

if __name__ == '__main__': 
    # env = test_env_initialization()
    # test_image(env)

    env = FingertipMotionEnv(asset_root="/home/irmak/Workspace/third-person-manipulation/third_person_man/models")
    env.reset()
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    color_image = env.render()
    # print('color_image.shape: {}'.format(color_image.shape))
    plt.imshow(np.transpose(color_image, (1,2,0)) )
    plt.savefig('ex_image_cube.jpg') 
    