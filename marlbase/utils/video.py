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


def record_episodes(env, act_func, n_timesteps, path):
    recorder = VideoRecorder()
    done = True

    for _ in range(n_timesteps):
        if done:
            obs, _ = env.reset()
            done = False
        else:
            with torch.no_grad():
                obs, _, done, truncated, _ = env.step(act_func(obs))
            done = done or truncated
        recorder.record_frame(env)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    recorder.save(path)
