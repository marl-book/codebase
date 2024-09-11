from collections import namedtuple
import math
from pathlib import Path

# from cpprb import ReplayBuffer
import hydra
import numpy as np
from omegaconf import DictConfig
import torch

from marlbase.utils.video import record_episodes


Batch = namedtuple("Batch", ["obss", "actions", "rewards", "dones", "filled"])


class ReplayBuffer:
    def __init__(
        self,
        buffer_size,
        n_agents,
        observation_space,
        max_episode_length,
        device,
    ):
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.max_episode_length = max_episode_length
        self.device = device

        self.pos = 0
        self.cur_pos = 0
        self.t = 0

        self.observations = [
            np.zeros(
                (max_episode_length + 1, buffer_size, *observation_space[i].shape),
                dtype=np.float32,
            )
            for i in range(n_agents)
        ]
        self.actions = np.zeros(
            (n_agents, max_episode_length, buffer_size), dtype=np.int64
        )
        self.rewards = np.zeros(
            (n_agents, max_episode_length, buffer_size), dtype=np.float32
        )
        self.dones = np.zeros((max_episode_length + 1, buffer_size), dtype=bool)
        self.filled = np.zeros((max_episode_length, buffer_size), dtype=bool)

    def __len__(self):
        return min(self.pos, self.buffer_size)

    def init_episode(self, obss):
        self.t = 0
        for i in range(self.n_agents):
            self.observations[i][0, self.cur_pos] = obss[i]

    def add(self, obss, acts, rews, done):
        assert self.t < self.max_episode_length, "Episode longer than given max length!"
        for i in range(self.n_agents):
            self.observations[i][self.t + 1, self.cur_pos] = obss[i]
        self.actions[:, self.t, self.cur_pos] = acts
        self.rewards[:, self.t, self.cur_pos] = rews
        self.dones[self.t + 1, self.cur_pos] = done
        self.filled[self.t, self.cur_pos] = True
        self.t += 1

        if done:
            self.pos += 1
            self.cur_pos = self.pos % self.buffer_size
            self.t = 0

    def can_sample(self, batch_size):
        return self.pos >= batch_size

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self), size=batch_size)
        obss = torch.stack(
            [
                torch.tensor(
                    self.observations[i][:, idx],
                    dtype=torch.float32,
                    device=self.device,
                )
                for i in range(self.n_agents)
            ]
        )
        actions = torch.tensor(
            self.actions[:, :, idx], dtype=torch.int64, device=self.device
        )
        rewards = torch.tensor(
            self.rewards[:, :, idx], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            self.dones[:, idx], dtype=torch.float32, device=self.device
        )
        filled = torch.tensor(
            self.filled[:, idx], dtype=torch.float32, device=self.device
        )
        return Batch(obss, actions, rewards, dones, filled)


def _epsilon_schedule(
    decay_style, decay_over, eps_start, eps_end, exp_decay_rate, total_steps
):
    """
    Exponential decay schedule for exploration epsilon.
    :param decay_style: style of epsilon schedule. One of "linear"/ "lin" or "exponential"/ "exp".
    :param decay_over: fraction of total steps over which to decay epsilon.
    :param eps_start: starting epsilon value.
    :param eps_end: ending epsilon value.
    :param exp_decay_rate: decay rate for exponential decay.
    :param total_steps: total number of steps to take.
    :return: Epsilon schedule function mapping step number to epsilon value.
    """
    assert decay_style in [
        "linear",
        "lin",
        "exponential",
        "exp",
    ], "decay_style must be one of 'linear' or 'exponential'"
    assert 0 <= eps_start <= 1 and 0 <= eps_end <= 1, "eps must be in [0, 1]"
    assert eps_start >= eps_end, "eps_start must be >= eps_end"
    assert 0 < decay_over <= 1, "decay_over must be in (0, 1]"
    assert total_steps > 0, "total_steps must be > 0"
    assert exp_decay_rate > 0, "eps_decay must be > 0"

    if decay_style in ["linear", "lin"]:

        def _thunk(steps_done):
            return max(
                eps_end
                + (eps_start - eps_end) * (1 - steps_done / (total_steps * decay_over)),
                eps_end,
            )

    elif decay_style in ["exponential", "exp"]:
        # decaying over all steps
        # eps_decay = (eps_start - eps_end) / total_steps * exp_decay_rate
        # decaying over decay_over fraction of steps
        eps_decay = (eps_start - eps_end) / (total_steps * decay_over) * exp_decay_rate

        def _thunk(steps_done):
            return max(
                eps_end + (eps_start - eps_end) * math.exp(-eps_decay * steps_done),
                eps_end,
            )
    else:
        raise ValueError("decay_style must be one of 'linear' or 'exponential'")
    return _thunk


def _evaluate(env, model, eval_episodes, eval_epsilon):
    infos = []
    while len(infos) < eval_episodes:
        obs, info = env.reset()
        hiddens = model.init_hiddens(1)
        done = False
        while not done:
            with torch.no_grad():
                actions, hiddens = model.act(obs, hiddens, eval_epsilon)
            obs, _, done, truncated, info = env.step(actions)
            done = done or truncated
        infos.append(info)
    return infos


def _collect_trajectory(env, model, rb, epsilon, use_proper_termination):
    obss, _ = env.reset()
    rb.init_episode(obss)
    hiddens = model.init_hiddens(1)
    done = False
    t = 0

    while not done:
        with torch.no_grad():
            actions, hiddens = model.act(obss, hiddens, epsilon=epsilon)
        next_obss, rews, done, truncated, info = env.step(actions)

        if use_proper_termination:
            # TODO: Previously was always False here?
            # also previously had other option "ignore"? Why was that separate from "ignore"?
            proper_done = done
        else:
            # here previously was always done?
            proper_done = done or truncated
        done = done or truncated

        rb.add(next_obss, actions, rews, proper_done)
        t += 1
        obss = next_obss

    return t, info


def main(env, eval_env, logger, time_limit, **cfg):
    cfg = DictConfig(cfg)

    model = hydra.utils.instantiate(
        cfg.model, env.observation_space, env.action_space, cfg
    )

    logger.watch(model)

    rb = ReplayBuffer(
        cfg.buffer_size,
        env.unwrapped.n_agents,
        env.observation_space,
        time_limit,
        cfg.model.device,
    )

    eps_sched = _epsilon_schedule(
        cfg.eps_decay_style,
        cfg.eps_decay_over,
        cfg.eps_start,
        cfg.eps_end,
        cfg.eps_exp_decay_rate,
        cfg.total_steps,
    )

    updates = 0
    step = 0
    last_eval = 0
    last_video = 0
    last_save = 0
    while step < cfg.total_steps + 1:
        t, _ = _collect_trajectory(
            env,
            model,
            rb,
            eps_sched(step),
            cfg.use_proper_termination,
        )
        step += t

        if step > cfg.training_start and rb.can_sample(cfg.batch_size):
            batch = rb.sample(cfg.batch_size)
            metrics = model.update(batch)
            updates += 1
        else:
            metrics = {}

        if cfg.eval_interval and (step - last_eval) >= cfg.eval_interval:
            infos = _evaluate(eval_env, model, cfg.eval_episodes, cfg.eps_evaluation)
            if metrics:
                infos.append(metrics)
            infos.append(
                {
                    "updates": updates,
                    "environment_steps": step,
                    "epsilon": eps_sched(step),
                }
            )
            logger.log_metrics(infos)
            last_eval = step

        if cfg.video_interval and (step - last_video) >= cfg.video_interval:
            record_episodes(
                eval_env,
                lambda x: model.act(x, cfg.greedy_epsilon),
                cfg.video_frames,
                f"./videos/step-{step}.mp4",
            )
            last_video = step

        if cfg.save_interval and (step - last_save) >= cfg.save_interval:
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/model_s{step}.pt")
            last_save = step

    env.close()
