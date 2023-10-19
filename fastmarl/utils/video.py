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
        self.frames.append(env.render(mode="rgb_array"))

    def save(self, filename):
        imageio.mimsave(f"{filename}", self.frames, fps=self.fps)


def record_episodes(env, act_func, n_timesteps, path):
    recorder = VideoRecorder()
    done = True

    for _ in range(n_timesteps):
        if done:
            obs = env.reset()
            done = False
            recorder.record_frame(env)
        else:
            with torch.no_grad():
                obs, _, done, _ = env.step(act_func(obs))
            recorder.record_frame(env)

    env.close()
    Path(path).parents[0].mkdir(parents=True, exist_ok=True)
    recorder.save(path)
