import cv2
import imageio
import numpy as np


class VideoRecorder:
    def __init__(self, save_dir, render_size=256, fps=20):
        assert save_dir is not None, 'Save Directory in VideoRecorder cannot be None'
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs):
        self.frames = []
        # self.record(obs)
        self.record_realsense(obs)

    def record(self, obs):
        frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                            dsize=(self.render_size, self.render_size),
                            interpolation=cv2.INTER_CUBIC)
        self.frames.append(frame)

    def record_realsense(self, img):
        frame = img
        self.frames.append(frame)

    def save(self, file_name):
        path = self.save_dir / file_name
        imageio.mimsave(str(path), self.frames, fps=self.fps)
