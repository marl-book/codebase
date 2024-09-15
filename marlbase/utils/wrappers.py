"""
A collection of environment wrappers for multi-agent environments
"""

from collections import deque
from time import perf_counter

import gymnasium as gym
from gymnasium import ObservationWrapper, spaces
import numpy as np


class RecordEpisodeStatistics(gym.Wrapper):
    """Multi-agent version of RecordEpisodeStatistics gym wrapper"""

    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.t0 = perf_counter()
        self.episode_reward = np.zeros(self.unwrapped.n_agents)
        self.episode_length = 0
        self.reward_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.episode_reward = 0
        self.episode_length = 0
        self.t0 = perf_counter()
        return obs, info

    def step(self, action):
        observation, reward, done, truncated, info = super().step(action)
        self.episode_reward += np.array(reward, dtype=np.float32)
        self.episode_length += 1
        if done or truncated:
            info["episode_returns"] = self.episode_reward
            if len(self.episode_reward) == self.unwrapped.n_agents:
                for i, agent_reward in enumerate(self.episode_reward):
                    info[f"agent{i}/episode_returns"] = agent_reward
            info["episode_length"] = self.episode_length
            info["episode_time"] = perf_counter() - self.t0

            self.reward_queue.append(self.episode_reward)
            self.length_queue.append(self.episode_length)
        return observation, reward, done, truncated, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        ma_spaces = []
        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]
        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class ObserveID(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        agent_count = env.unwrapped.n_agents
        for obs_space in self.observation_space:
            assert (
                isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 1
            ), "ObserveID wrapper assumes flattened observation space."
        self.observation_space = gym.spaces.Tuple(
            tuple(
                [
                    gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=((x.shape[0] + agent_count),),
                        dtype=x.dtype,
                    )
                    for x in self.observation_space
                ]
            )
        )

    def observation(self, observation):
        observation = np.stack(observation)
        observation = np.concatenate(
            (np.eye(self.unwrapped.n_agents, dtype=observation.dtype), observation),
            axis=1,
        )
        return [o.squeeze() for o in np.split(observation, self.unwrapped.n_agents)]


class CooperativeReward(gym.RewardWrapper):
    def reward(self, reward):
        return self.unwrapped.n_agents * [sum(reward)]


class StandardizeReward(gym.RewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stdr_wrp_sumw = np.zeros(self.unwrapped.n_agents, dtype=np.float32)
        self.stdr_wrp_wmean = np.zeros(self.unwrapped.n_agents, dtype=np.float32)
        self.stdr_wrp_t = np.zeros(self.unwrapped.n_agents, dtype=np.float32)
        self.stdr_wrp_n = 0

    def reward(self, reward):
        # based on http://www.nowozin.net/sebastian/blog/streaming-mean-and-variance-computation.html
        # update running avg and std
        weight = 1.0

        q = reward - self.stdr_wrp_wmean
        temp_sumw = self.stdr_wrp_sumw + weight
        r = q * weight / temp_sumw

        self.stdr_wrp_wmean += r
        self.stdr_wrp_t += q * r * self.stdr_wrp_sumw
        self.stdr_wrp_sumw = temp_sumw
        self.stdr_wrp_n += 1

        if self.stdr_wrp_n == 1:
            return reward

        # calculate standardized reward
        var = (self.stdr_wrp_t * self.stdr_wrp_n) / (
            self.stdr_wrp_sumw * (self.stdr_wrp_n - 1)
        )
        stdr_rew = (reward - self.stdr_wrp_wmean) / (np.sqrt(var) + 1e-6)
        return stdr_rew


class ClearInfo(gym.Wrapper):
    def step(self, action):
        observation, reward, done, truncated, _ = self.env.step(action)
        return observation, reward, done, truncated, {}
