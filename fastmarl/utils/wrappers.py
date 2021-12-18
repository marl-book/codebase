"""
A collection of environment wrappers for multi-agent environments
"""
from collections import deque, Iterable
from time import perf_counter

import gym
import numpy as np
from gym import ObservationWrapper, spaces


def is_wrapped_by(env, wrapper_class):
    while True:
        if isinstance(env, wrapper_class):
            return True
        if not hasattr(env, "env"):
            return False
        env = env.env

class RecordEpisodeStatistics(gym.Wrapper):
    """Multi-agent version of RecordEpisodeStatistics gym wrapper"""

    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.t0 = perf_counter()
        self.episode_reward = np.zeros(self.n_agents)
        self.episode_length = 0
        self.reward_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.episode_reward = 0
        self.episode_length = 0
        self.t0 = perf_counter()

        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.episode_reward += np.array(reward, dtype=np.float64)
        self.episode_length += 1
        if all(done):
            info["episode_returns"] = self.episode_reward
            if len(self.episode_reward) == self.n_agents:
                for i, agent_reward in enumerate(self.episode_reward):
                    info[f"agent{i}/episode_returns"] = agent_reward
            info["episode_length"] = self.episode_length
            info["episode_time"] = perf_counter() - self.t0

            self.reward_queue.append(self.episode_reward)
            self.length_queue.append(self.episode_length)
        return observation, reward, done, info


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


class SquashDones(gym.Wrapper):
    r"""Wrapper that squashes multiple dones to a single one using all(dones)"""

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, all(done), info

class ObserveID(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        agent_count = env.n_agents
        self.observation_space = gym.spaces.Tuple(tuple([gym.spaces.Box(low=-np.inf, high=np.inf, shape=((x.shape[0] + agent_count),), dtype=x.dtype) for x in self.observation_space]))
    def observation(self, observation):
        observation = np.stack(observation)
        observation = np.concatenate((np.eye(self.n_agents, dtype=observation.dtype), observation), axis=1)
        return [o.squeeze() for o in np.split(observation, self.n_agents)]

class GlobalizeReward(gym.RewardWrapper):
    def reward(self, reward):
        return self.n_agents * [sum(reward)]

class CooperativeReward(gym.RewardWrapper):
    def reward(self, reward):
        return [sum(reward)]

class StandardizeReward(gym.RewardWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stdr_wrp_sumw = np.zeros(self.n_agents, dtype=np.float32)
        self.stdr_wrp_wmean = np.zeros(self.n_agents, dtype=np.float32)
        self.stdr_wrp_t = np.zeros(self.n_agents, dtype=np.float32)
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


class TimeLimit(gym.wrappers.TimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done)
            done = len(observation) * [True]
        return observation, reward, done, info


class ClearInfo(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, {}


class SMACCompatible(gym.Wrapper):
    def get_avail_actions(self):
        return [np.ones(x.n) for x in self.action_space]

    def get_state(self):
        return [np.zeros(5) for x in self.observation_space]


class Monitor(gym.wrappers.Monitor):
    def _after_step(self, observation, reward, done, info):
        if not self.enabled:
            return done

        if done and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
            self.reset_video_recorder()
            self.episode_id += 1
            self._flush()

        # Record stats
        self.stats_recorder.after_step(observation, sum(reward), done, info)
        # Record video
        self.video_recorder.capture_frame()

        return done
