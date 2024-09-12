import numpy as np
import gymnasium as gym


class SMACliteWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = self.env.unwrapped.n_agents

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        actions = [int(act) for act in actions]
        obs, reward, terminated, truncated, info = self.env.step(actions)
        info["action_mask"] = np.array(
            self.env.unwrapped.get_avail_actions(), dtype=np.float32
        )
        return obs, [reward] * self.n_agents, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        obs, info = self.env.reset(seed=seed, options=options)
        info["action_mask"] = np.array(
            self.env.unwrapped.get_avail_actions(), dtype=np.float32
        )
        return obs, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
