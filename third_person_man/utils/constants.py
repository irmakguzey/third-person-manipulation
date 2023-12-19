# Alexnet means and stds
TACTILE_IMAGE_MEANS = [0.485, 0.456, 0.406]
TACTILE_IMAGE_STDS = [0.229, 0.224, 0.225]

# These constants are used to clamp 
TACTILE_PLAY_DATA_CLAMP_MIN = -1000
TACTILE_PLAY_DATA_CLAMP_MAX = 1000

# Constans for camera images
VISION_IMAGE_MEANS = [0.4191, 0.4445, 0.4409]
VISION_IMAGE_STDS = [0.2108, 0.1882, 0.1835]

MODALITY_TYPES = {
    'allegro': 'hand',
    'kinova': 'arm',
    'franka': 'arm',
    'image': 'image',
    'tactile': 'tactile' # TODO: This could be changed to reskin / xela
}