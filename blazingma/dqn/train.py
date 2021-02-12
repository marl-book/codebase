import logging
import math
import random
import time
from collections import deque

import gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import optim
from tqdm import tqdm

from cpprb import ReplayBuffer, create_before_add_func, create_env_dict
from gymma import wrappers
from blazingma.dqn.model import QNetwork
from blazingma.utils.loggers import Logger


def _squash_info(info):
    info = [i for i in info if i]
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")
    for key in keys:
        mean = np.mean([np.array(d[key]).sum() for d in info if key in d])
        new_info[key] = mean
    return new_info


def _plot_epsilon(eps_sched, total_steps):
    import matplotlib.pyplot as plt

    x = np.arange(0, 1, 0.001) * total_steps
    y = list(map(eps_sched, x))
    plt.plot(x, y)
    plt.show()


def _epsilon_schedule(eps_start, eps_end, eps_decay):
    def _thunk(steps_done):
        return eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)

    return _thunk


def _evaluate(env, model, eval_episodes, greedy_epsilon):
    infos = []
    for j in range(eval_episodes):
        done = False
        obs = env.reset()
        while not done:
            act = model.act(obs, greedy_epsilon)
            obs, _, done, info = env.step(act)
            env.render()

        infos.append(info)

    return infos

@hydra.main(config_name="config")
def main(cfg: DictConfig):

    logger = Logger("blazing-ma", cfg)

    torch.set_num_threads(1)

    # env config:
    env = gym.make("gymma:" + cfg.env.name)
    if cfg.env.time_limit:
        env = wrappers.TimeLimit(env, cfg.env.time_limit)
    for wrapper in cfg.env.wrappers:
        env = getattr(wrappers, wrapper)(env)

    # replay buffer:
    env_dict = create_env_dict(env)
    env_dict["rew"]["shape"] = env.n_agents
    rb = ReplayBuffer(cfg.buffer_size, env_dict)
    before_add = create_before_add_func(env)

    # DQN model:
    model = QNetwork(env.observation_space, env.action_space, cfg).to(cfg.model.device)

    # epsilon
    eps_sched = _epsilon_schedule(cfg.eps_start, cfg.eps_end, cfg.eps_decay)

    # training loop:
    start_time = time.process_time()
    obs = env.reset()
    for j in range(1, cfg.total_steps + 1):

        act = model.act(obs, epsilon=eps_sched(j))

        next_obs, rew, done, info = env.step(act)

        if cfg.use_proper_termination and done and info.get("TimeLimit.truncated", False):
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
            batch = {k: torch.from_numpy(v).to(cfg.model.device) for k, v in batch.items()}
            model.update(batch)

        if done:
            obs = env.reset()
        else:
            obs = next_obs

        if j % cfg.eval_freq == 0:
            end_time = time.process_time()
            logging.info(
                f"Completed: {100*j/cfg.total_steps}% - FPS: {cfg.eval_freq/(end_time - start_time):.1f}"
            )
            infos = _evaluate(env, model, cfg.eval_episodes, cfg.greedy_epsilon)
            infos = _squash_info(infos)

            logging.info(
                f"Evaluation ({cfg.eval_episodes} episodes): {infos['episode_reward']:.3f} mean reward"
            )

            infos.update({"epsilon": eps_sched(j)})
            logger.log_metrics(infos)
            start_time = time.process_time()

if __name__ == "__main__":
    main()