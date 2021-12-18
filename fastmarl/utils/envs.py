import gym
from fastmarl.utils import wrappers as mwrappers
from omegaconf import DictConfig
from functools import partial
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
import numpy as np
import random


def async_reset(envs):
    time_limit = 25
    obs = envs.reset()
    parallel_envs = obs[0].shape[0]

    async_array = [i * time_limit // parallel_envs for i in range(parallel_envs)]
    if isinstance(envs, DummyVecEnv):
        for i in range(parallel_envs):
            for j in range(async_array[i]):
                _ = envs.envs[i].step(envs.action_space.sample())
        obs, _, _, _ = envs.step(
            [envs.action_space.sample() for _ in range(parallel_envs)]
        )
        return obs
    elif isinstance(envs, SubprocVecEnv):
        for i in range(parallel_envs):
            for j in range(async_array[i]):
                envs.remotes[i].send(("step", envs.action_space.sample()))
                _ = envs.remotes[i].recv()
        obs, _, _, _ = envs.step(
            [envs.action_space.sample() for _ in range(parallel_envs)]
        )
        return obs


def _make_parallel_envs(
    name, parallel_envs, dummy_vecenv, wrappers, time_limit, clear_info, observe_id, seed, **kwargs
):
    def _env_thunk(seed):
        env = gym.make(name, **kwargs)
        if clear_info:
            env = mwrappers.ClearInfo(env)
        if time_limit:
            env = mwrappers.TimeLimit(env, time_limit)
        if observe_id:
            env = mwrappers.ObserveID(env)
        for wrapper in wrappers:
            env = getattr(mwrappers, wrapper)(env)
        env.seed(seed)
        return env

    if seed is None:
        seed = random.randint(0, 99999)

    env_thunks = [partial(_env_thunk, seed + i) for i in range(parallel_envs)]
    if dummy_vecenv:
        envs = DummyVecEnv(env_thunks)
        envs.buf_rews = np.zeros(
            (parallel_envs, len(envs.observation_space)), dtype=np.float32
        )
    else:
        envs = SubprocVecEnv(env_thunks, start_method="fork")

    return envs


def _make_env(name, time_limit, clear_info, observe_id, wrappers, seed, **kwargs):
    env = gym.make(name, **kwargs)
    if clear_info:
        env = mwrappers.ClearInfo(env)
    if time_limit:
        env = mwrappers.TimeLimit(env, time_limit)
    if observe_id:
        env = mwrappers.ObserveID(env)
    for wrapper in wrappers:
        env = getattr(mwrappers, wrapper)(env)
    env.seed(seed)
    return env


def make_env(seed, **env):
    env = DictConfig(env)
    if "parallel_envs" in env and env.parallel_envs:
        return _make_parallel_envs(**env, seed=seed)
    return _make_env(**env, seed=seed)
