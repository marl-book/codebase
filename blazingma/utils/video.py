import imageio

class VideoRecorder:
    def __init__(self, fps=30):
        self.fps = fps
        self.frames = []

    def reset(self):
        self.frames = []

    def record_frame(self, env):
        self.frames.append(env.render(mode='rgb_array'))

    def save(self, filename):
        imageio.mimsave(f'{filename}.mp4', self.frames, fps=self.fps)
