from copy import deepcopy

from cpprb import ReplayBuffer, create_before_add_func, create_env_dict
import hydra
from omegaconf import DictConfig
import torch

from fastmarl.dqn.train import _evaluate, _epsilon_schedule
from fastmarl.utils import wrappers
from fastmarl.utils.video import record_episodes


def main(env, logger, **cfg):
    cfg = DictConfig(cfg)

    # replay buffer:
    env_dict = create_env_dict(env)
    
    force_coop = wrappers.is_wrapped_by(env, wrappers.CooperativeReward)
    if not force_coop: env_dict["rew"]["shape"] = env.n_agents
    rb = ReplayBuffer(cfg.buffer_size, env_dict)
    rb_policy = ReplayBuffer(cfg.policy_buffer_size, env_dict)
    before_add = create_before_add_func(env)

    model = hydra.utils.instantiate(cfg.model, env.observation_space, env.action_space, cfg)

    # Logging
    logger.watch(model)

    # epsilon
    eps_sched = _epsilon_schedule(cfg.eps_decay_style, cfg.eps_start, cfg.eps_end, cfg.eps_decay, cfg.total_steps)

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

        experience_sample = before_add(obs=obs, act=act, next_obs=next_obs, rew=rew, done=proper_done)
        rb.add(**experience_sample)
        rb_policy.add(**experience_sample)

        if j > cfg.training_start:
            batch = rb.sample(cfg.batch_size)
            batch = {
                k: torch.from_numpy(v).to(cfg.model.device) for k, v in batch.items()
            }
            critic_loss = model.compute_critic_loss(batch)

            policy_batch = rb_policy.sample(cfg.batch_size)
            policy_batch = {
                k: torch.from_numpy(v).to(cfg.model.device) for k, v in policy_batch.items()
            }
            agent_model_loss = model.compute_agent_model_loss(policy_batch)
            model.update([critic_loss, agent_model_loss])

        obs = env.reset() if done else next_obs

        if cfg.video_interval and j % cfg.video_interval == 0:
            record_episodes(
                deepcopy(env),
                lambda x: model.act(x, cfg.greedy_epsilon),
                cfg.video_frames,
                f"./videos/step-{j}.mp4",
            )


if __name__ == "__main__":
    main()
