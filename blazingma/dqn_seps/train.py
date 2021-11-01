import math
import time
from collections import deque

import gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from cpprb import ReplayBuffer, create_before_add_func, create_env_dict
from blazingma.dqn_seps.model import QNetwork
from blazingma.utils.loggers import Logger
from blazingma.utils import wrappers
from tqdm import tqdm
from utils.video import record_episodes
from copy import deepcopy

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


def _evaluate(env, model, eval_episodes, seps_indices, greedy_epsilon):
    infos = []
    for j in range(eval_episodes):
        done = False
        obs = env.reset()
        while not done:
            with torch.no_grad():
                act = model.act(obs, seps_indices, greedy_epsilon)
            obs, _, done, info = env.step(act)

        infos.append(info)

    return infos


def main(env, logger, **cfg):
    cfg = DictConfig(cfg)

    # replay buffer:
    env_dict = create_env_dict(env)
    env_dict["rew"]["shape"] = env.n_agents
    rb = ReplayBuffer(cfg.buffer_size, env_dict)
    before_add = create_before_add_func(env)

    # DQN model:
    model = QNetwork(env.observation_space, env.action_space, cfg).to(cfg.model.device)
    logger.watch(model)

    # SePS setting
    if cfg.seps_setting == "nops":
        seps_indices = list(range(env.n_agents))
        seps_indices = torch.Tensor(seps_indices).type(torch.int64)
    elif cfg.seps_setting == "fups":
        seps_indices = [0 for _ in range(env.n_agents)]
        seps_indices = torch.Tensor(seps_indices).type(torch.int64)
    elif "[" in str(cfg.seps_setting):
        seps_indices = torch.Tensor(list(cfg.seps_setting)).type(torch.int64)
    else:
        raise ValueError(f'You provided a seps_setting of: {cfg.seps_setting}, which is not supported.')

    assert len(seps_indices) == env.n_agents, f'The given "seps_settings" does not match the number of agents, which is {env.n_agents}.'

    # epsilon
    eps_sched = _epsilon_schedule(cfg.eps_start, cfg.eps_end, cfg.eps_decay)

    # training loop:
    start_time = time.process_time()
    obs = env.reset()

    for j in tqdm(range(0, cfg.total_steps + 1)):

        act = model.act(obs, seps_indices, epsilon=eps_sched(j))

        next_obs, rew, done, info = env.step(act)

        if (
            cfg.use_proper_termination
            and done
            and info.get("TimeLimit.truncated", False)
        ):
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
            model.update(batch, seps_indices)

        if done:
            obs = env.reset()
        else:
            obs = next_obs

        if j % cfg.eval_interval == 0:
            end_time = time.process_time()
            logger.info(
                f"Completed: {100*j/cfg.total_steps}% - FPS: {cfg.eval_interval/(end_time - start_time):.1f}"
            )
            infos = _evaluate(env, model, cfg.eval_episodes, seps_indices, cfg.greedy_epsilon)
            mean_reward = sum(sum([ep["episode_reward"] for ep in infos]) / len(infos))
            logger.info(
                f"Evaluation ({cfg.eval_episodes} episodes): {mean_reward:.3f} mean reward"
            )
            logger.info(f"Epsilon value: {eps_sched(j):.3f}")

            logger.log_metrics(infos)
            start_time = time.process_time()

        if cfg.video_interval and j % cfg.video_interval == 0:
            record_episodes(
                deepcopy(env),
                lambda x: model.act(x, cfg.greedy_epsilon),
                cfg.video_frames,
                f"./videos/step-{j}.mp4",
            )


if __name__ == "__main__":
    main()
