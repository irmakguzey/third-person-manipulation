import cv2
import os
import numpy as np 

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