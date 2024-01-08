import cv2
import os
import numpy as np 

def plot_axes(axes, img, color_set=1):
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

        elif color_set == 3: 

            img = cv2.line(img, tuple(axis[3].ravel()), tuple(axis[0].ravel()), (255, 153, 153), 3) # Light Red
            img = cv2.line(img, tuple(axis[3].ravel()), tuple(axis[1].ravel()), (204, 255, 204), 3) # Green
            img = cv2.line(img, tuple(axis[3].ravel()), tuple(axis[2].ravel()), (153, 255, 255), 3) # Light Blue

    return img

def concat_imgs(img1, img2, orientation='horizontal'): # Or it could be vertical as well
    metric_id = 0 if orientation == 'horizontal' else 1
    max_metric = max(img1.shape[metric_id], img2.shape[metric_id])
    min_metric = min(img1.shape[metric_id], img2.shape[metric_id])
    scale = min_metric / max_metric
    large_img_idx = np.argmax([img1.shape[metric_id], img2.shape[metric_id]])

    if large_img_idx == 0: 
        img1 = cv2.resize(
            img1, 
            (int(img1.shape[1]*scale),
             int(img1.shape[0]*scale))
        )
    else: 
        img2 = cv2.resize(
            img2, 
            (int(img2.shape[1]*scale),
             int(img2.shape[0]*scale))
        )

    concat_img = cv2.hconcat([img1, img2]) if orientation == 'horizontal' else cv2.vconcat([img1, img2])
    return concat_img

def turn_images_to_video(viz_dir, video_fps, video_name='visualization.mp4'):
    video_path = os.path.join(viz_dir, video_name)
    if os.path.exists(video_path):
        os.remove(video_path)
    os.system('ffmpeg -r {} -i {}/%*.png -vf setsar=1:1 {}'.format(
        video_fps, # fps
        viz_dir,
        video_path
    ))

def turn_video_to_images(dir_path, video_name, images_dir_name, images_fps):
    images_path = os.path.join(dir_path, images_dir_name)
    video_path = os.path.join(dir_path, video_name)
    os.makedirs(images_path, exist_ok=True)
    os.system(f'ffmpeg -i {video_path} -vf fps={images_fps} {images_path}/out%d.png')