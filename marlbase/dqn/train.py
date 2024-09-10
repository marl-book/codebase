import math
from pathlib import Path

from cpprb import ReplayBuffer, create_before_add_func, create_env_dict
import hydra
import numpy as np
from omegaconf import DictConfig
import torch

from marlbase.utils.video import record_episodes


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


def _evaluate(env, model, eval_episodes, greedy_epsilon):
    infos = []
    while len(infos) < eval_episodes:
        done = False
        obs, info = env.reset()
        hiddens = model.init_hiddens(1)
        while not done:
            with torch.no_grad():
                act, hiddens = model.act(obs, hiddens, greedy_epsilon)
            obs, _, done, truncated, info = env.step(act)
            done = done or truncated
        infos.append(info)
    return infos


def _collect_trajectory(
    env, model, epsilon, time_limit, n_agents, use_proper_termination
):
    obss = [
        np.zeros((time_limit + 1, *env.observation_space[i].shape))
        for i in range(n_agents)
    ]
    acts = [np.zeros((time_limit, 1), dtype=np.int64) for _ in range(n_agents)]
    rews = np.zeros((time_limit, n_agents), dtype=np.float32)
    dones = np.zeros(time_limit + 1, dtype=bool)
    filled = np.zeros(time_limit, dtype=bool)

    obs, _ = env.reset()
    hiddens = model.init_hiddens(1)
    done = False
    t = 0
    for i in range(n_agents):
        obss[i][0] = obs[i]

    while not done:
        with torch.no_grad():
            actions, hiddens = model.act(obs, hiddens, epsilon=epsilon)
        next_obs, rew, done, truncated, info = env.step(actions)

        if use_proper_termination:
            # TODO: Previously was always False here?
            # also previously had other option "ignore"? Why was that separate from "ignore"?
            proper_done = done
        else:
            # here previously was always done?
            proper_done = done or truncated
        done = done or truncated

        for i in range(n_agents):
            obss[i][t + 1] = next_obs[i]
            acts[i][t] = actions[i]
        rews[t] = rew
        dones[t + 1] = proper_done
        filled[t] = 1
        t += 1
        obs = next_obs

    ep_data = {
        "rew": rews,
        "done": dones,
        "filled": filled,
    }
    for i in range(n_agents):
        ep_data[f"obs{i}"] = obss[i]
        ep_data[f"act{i}"] = acts[i]
    return t, ep_data


def main(env, eval_env, logger, time_limit, **cfg):
    cfg = DictConfig(cfg)

    # episodic replay buffer:
    env_dict = {
        "rew": {"shape": (time_limit, env.unwrapped.n_agents), "dtype": np.float32},
        "done": {"shape": time_limit + 1, "dtype": bool},
        "filled": {"shape": time_limit, "dtype": bool},
    }
    for i in range(env.unwrapped.n_agents):
        env_dict[f"obs{i}"] = {
            "shape": (time_limit + 1, *env.observation_space[i].shape),
            "dtype": np.float32,
        }
        env_dict[f"act{i}"] = {"shape": (time_limit, 1), "dtype": np.int64}
    rb = ReplayBuffer(cfg.buffer_size, env_dict)

    model = hydra.utils.instantiate(
        cfg.model, env.observation_space, env.action_space, cfg
    )

    # Logging
    logger.watch(model)

    # epsilon
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
        t, ep_data = _collect_trajectory(
            env,
            model,
            eps_sched(step),
            time_limit,
            env.unwrapped.n_agents,
            cfg.use_proper_termination,
        )
        step += t
        rb.add(**ep_data)

        if step > cfg.training_start:
            batch = rb.sample(cfg.batch_size)
            batch = {
                k: torch.tensor(v, device=cfg.model.device) for k, v in batch.items()
            }
            metrics = model.update(batch)
            updates += 1
        else:
            metrics = {}

        if cfg.eval_interval and (step - last_eval) >= cfg.eval_interval:
            infos = _evaluate(eval_env, model, cfg.eval_episodes, cfg.greedy_epsilon)
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
