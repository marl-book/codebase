import math
from pathlib import Path

from cpprb import ReplayBuffer, create_before_add_func, create_env_dict
import hydra
from omegaconf import DictConfig
import torch

from marlbase.utils.video import record_episodes


def _epsilon_schedule(decay_style, eps_start, eps_end, eps_decay, total_steps):
    """
    Exponential decay schedule for exploration epsilon.
    :param decay_style: Style of epsilon schedule. One of "linear"/ "lin" or "exponential"/ "exp".
    :param eps_start: Starting epsilon value.
    :param eps_end: Ending epsilon value.
    :param eps_decay: Decay rate.
    :param total_steps: Total number of steps to take.
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
    assert total_steps > 0, "total_steps must be > 0"
    assert eps_decay > 0, "eps_decay must be > 0"

    if decay_style in ["linear", "lin"]:

        def _thunk(steps_done):
            return eps_end + (eps_start - eps_end) * (1 - steps_done / total_steps)
    elif decay_style in ["exponential", "exp"]:
        eps_decay = (eps_start - eps_end) / total_steps * eps_decay

        def _thunk(steps_done):
            return eps_end + (eps_start - eps_end) * math.exp(-eps_decay * steps_done)
    else:
        raise ValueError("decay_style must be one of 'linear' or 'exponential'")
    return _thunk


def _evaluate(env, model, eval_episodes, greedy_epsilon):
    infos = []
    for j in range(eval_episodes):
        done = False
        obs, info = env.reset()
        while not done:
            with torch.no_grad():
                act = model.act(obs, greedy_epsilon)
            obs, _, done, truncated, info = env.step(act)
            done = done or truncated

        infos.append(info)

    return infos


def main(env, eval_env, logger, **cfg):
    cfg = DictConfig(cfg)

    # replay buffer:
    env_dict = create_env_dict(env)
    env_dict["rew"]["shape"] = env.unwrapped.n_agents
    rb = ReplayBuffer(cfg.buffer_size, env_dict)
    before_add = create_before_add_func(env)

    model = hydra.utils.instantiate(
        cfg.model, env.observation_space, env.action_space, cfg
    )

    # Logging
    logger.watch(model)

    # epsilon
    eps_sched = _epsilon_schedule(
        cfg.eps_decay_style, cfg.eps_start, cfg.eps_end, cfg.eps_decay, cfg.total_steps
    )

    # training loop:
    obs, info = env.reset()

    updates = 0
    for step in range(cfg.total_steps + 1):
        if step % cfg.eval_interval == 0:
            infos = _evaluate(eval_env, model, cfg.eval_episodes, cfg.greedy_epsilon)
            infos.append(
                {
                    "updates": updates,
                    "environment_steps": step,
                    "epsilon": eps_sched(step),
                }
            )
            logger.log_metrics(infos)

        act = model.act(obs, epsilon=eps_sched(step))
        next_obs, rew, done, truncated, info = env.step(act)

        if cfg.use_proper_termination and done and truncated:
            proper_done = False
        elif cfg.use_proper_termination == "ignore":
            # TODO: Why completely ignore done here?
            proper_done = False
        else:
            proper_done = done

        rb.add(
            **before_add(obs=obs, act=act, next_obs=next_obs, rew=rew, done=proper_done)
        )

        if step > cfg.training_start:
            batch = rb.sample(cfg.batch_size)
            batch = {
                k: torch.from_numpy(v).to(cfg.model.device) for k, v in batch.items()
            }
            model.update(batch)
            updates += 1

        obs, info = env.reset() if done else (next_obs, info)

        if cfg.video_interval and step % cfg.video_interval == 0:
            record_episodes(
                eval_env,
                lambda x: model.act(x, cfg.greedy_epsilon),
                cfg.video_frames,
                f"./videos/step-{step}.mp4",
            )

        if cfg.save_interval and step % cfg.save_interval == 0:
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/model_s{step}.pt")
