from functools import partial
import random

import gymnasium as gym
from omegaconf import DictConfig

from marlbase.utils import wrappers as mwrappers
from marlbase.utils.smaclite_wrapper import SMACliteWrapper


def _make_parallel_envs(
    name,
    parallel_envs,
    wrappers,
    time_limit,
    clear_info,
    observe_id,
    seed,
    enable_video,
    **kwargs,
):
    def _env_thunk(seed):
        if "smaclite" in name:
            import smaclite  # noqa

            env = gym.make(
                name,
                seed=seed,
                render_mode="rgb_array" if enable_video else None,
                **kwargs,
            )
            env = SMACliteWrapper(env)
        else:
            env = gym.make(
                name, **kwargs, render_mode="rgb_array" if enable_video else None
            )
        if clear_info:
            env = mwrappers.ClearInfo(env)
        if time_limit:
            env = gym.wrappers.TimeLimit(env, time_limit)
        if observe_id:
            env = mwrappers.ObserveID(env)
        for wrapper in wrappers:
            env = getattr(mwrappers, wrapper)(env)
        env.reset(seed=seed)
        return env

    if seed is None:
        seed = random.randint(0, 99999)

    envs = gym.vector.AsyncVectorEnv(
        [partial(_env_thunk, seed + i) for i in range(parallel_envs)]
    )

    return envs


def _make_env(
    name, time_limit, clear_info, observe_id, wrappers, seed, enable_video, **kwargs
):
    if "smaclite" in name:
        import smaclite  # noqa

        env = gym.make(
            name,
            seed=seed,
            render_mode="rgb_array" if enable_video else None,
            **kwargs,
        )
        env = SMACliteWrapper(env)
    else:
        env = gym.make(
            name, render_mode="rgb_array" if enable_video else None, **kwargs
        )
    if clear_info:
        env = mwrappers.ClearInfo(env)
    if time_limit:
        env = gym.wrappers.TimeLimit(env, time_limit)
    if observe_id:
        env = mwrappers.ObserveID(env)
    for wrapper in wrappers:
        wrapper = (
            getattr(mwrappers, wrapper)
            if hasattr(mwrappers, wrapper)
            else getattr(gym.wrappers, wrapper)
        )
        env = wrapper(env)

    env.reset(seed=seed)
    return env


def make_env(seed, enable_video=False, **env_config):
    env_config = DictConfig(env_config)
    if "parallel_envs" in env_config and env_config.parallel_envs:
        return _make_parallel_envs(**env_config, enable_video=enable_video, seed=seed)
    return _make_env(**env_config, enable_video=enable_video, seed=seed)
