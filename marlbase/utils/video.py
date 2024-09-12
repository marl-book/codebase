from pathlib import Path

import imageio
import torch


class VideoRecorder:
    def __init__(self, fps=30):
        self.fps = fps
        self.frames = []

    def reset(self):
        self.frames = []

    def record_frame(self, env):
        frame = env.unwrapped.render()
        self.frames.append(frame)

    def save(self, filename):
        imageio.mimsave(f"{filename}", self.frames, fps=self.fps)
