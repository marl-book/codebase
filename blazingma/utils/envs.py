import gym
from blazingma.utils import wrappers as mwrappers
from omegaconf import DictConfig
from functools import partial
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
import numpy as np
import random

def _make_parallel_envs(name, parallel_envs, dummy_vecenv, wrappers, time_limit, seed):
    def _env_thunk(seed):
        env = gym.make(name)
        if time_limit:
            env = mwrappers.TimeLimit(env, time_limit)
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


def _make_env(name, time_limit, wrappers, seed, **kwargs):
    env = gym.make(name)
    if time_limit:
        env = mwrappers.TimeLimit(env, time_limit)
    for wrapper in wrappers:
        env = getattr(mwrappers, wrapper)(env)
    env.seed(seed)
    return env

def make_env(seed, **env):
    env = DictConfig(env)
    if env.parallel_envs:
        return _make_parallel_envs(**env, seed=seed)
    return _make_env(**env, seed=seed)

