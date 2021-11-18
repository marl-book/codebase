from copy import deepcopy
import math
from collections import deque

import gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from cpprb import ReplayBuffer, create_before_add_func, create_env_dict
from blazingma.dqn.model import QNetwork, VDNetwork
from blazingma.utils import wrappers
from utils.video import record_episodes
from copy import deepcopy


def _plot_epsilon(eps_sched, total_steps):
    import matplotlib.pyplot as plt

    x = np.arange(0, 1, 0.001) * total_steps
    y = list(map(eps_sched, x))
    plt.plot(x, y)
    plt.show()


def _epsilon_schedule(eps_start, eps_end, eps_decay, total_steps):
    eps_decay = (eps_start - eps_end) / total_steps * eps_decay

    def _thunk(steps_done):
        return eps_end + (eps_start - eps_end) * math.exp(-eps_decay * steps_done)
    return _thunk


def _evaluate(env, model, eval_episodes, greedy_epsilon):
    infos = []
    for j in range(eval_episodes):
        done = False
        obs = env.reset()
        while not done:
            with torch.no_grad():
                act = model.act(obs, greedy_epsilon)
            obs, _, done, info = env.step(act)

        infos.append(info)

    return infos


def main(env, logger, **cfg):
    cfg = DictConfig(cfg)

    # replay buffer:
    env_dict = create_env_dict(env)
    
    force_coop = wrappers.is_wrapped_by(env, wrappers.CooperativeReward)
    if not force_coop: env_dict["rew"]["shape"] = env.n_agents
    rb = ReplayBuffer(cfg.buffer_size, env_dict)
    before_add = create_before_add_func(env)

    model = hydra.utils.instantiate(cfg.model, env.observation_space, env.action_space, cfg) # TODO: improve config structure to make this cleaner

    # Logging
    logger.watch(model)

    # epsilon
    eps_sched = _epsilon_schedule(cfg.eps_start, cfg.eps_end, cfg.eps_decay, cfg.total_steps)

    # _plot_epsilon(eps_sched, cfg.total_steps)
    # training loop:
    obs = env.reset()

    for j in range(cfg.total_steps + 1):

        if j % cfg.eval_interval == 0:
            infos = _evaluate(env, model, cfg.eval_episodes, cfg.greedy_epsilon)
            infos.append(
                {'updates': j, 'environment_steps': j, 'epsilon': eps_sched(j)}
            )
            logger.log_metrics(infos)

        act = model.act(obs, epsilon=eps_sched(j))

        next_obs, rew, done, info = env.step(act)

        if (
            cfg.use_proper_termination
            and done
            and info.get("TimeLimit.truncated", False)
        ):
            del info["TimeLimit.truncated"]
            proper_done = False
        elif cfg.use_proper_termination == "ignore":
            proper_done = False
        else:
            proper_done = done

        rb.add(
            **before_add(obs=obs, act=act, next_obs=next_obs, rew=rew, done=proper_done)
        )

        if j > cfg.training_start:
            batch = rb.sample(cfg.batch_size)
            batch = {
                k: torch.from_numpy(v).to(cfg.model.device) for k, v in batch.items()
            }
            model.update(batch)

        if done:
            obs = env.reset()
        else:
            obs = next_obs

        if cfg.video_interval and j % cfg.video_interval == 0:
            record_episodes(
                deepcopy(env),
                lambda x: model.act(x, cfg.greedy_epsilon),
                cfg.video_frames,
                f"./videos/step-{j}.mp4",
            )


if __name__ == "__main__":
    main()
