import time
from collections import deque
from collections import defaultdict
from typing import DefaultDict
import hydra

from gym.spaces import flatdim
import numpy as np
import torch
from omegaconf import DictConfig

from blazingma.utils.loggers import FileSystemLogger
from blazingma.ac.model import Policy
from blazingma.utils.standarize_stream import RunningMeanStd
from blazingma.utils.envs import async_reset
from utils.video import record_episodes
from copy import deepcopy


@torch.jit.script
def _compute_returns(rewards, done, next_value, gamma: float):
    returns = [next_value]
    for i in range(len(rewards) - 1, -1, -1):
        ret = rewards[i] + gamma * returns[0] * (1 - done[i, :].unsqueeze(1))
        returns.insert(0, ret)
    return returns


def _log_progress(
    infos, prev_time, step, parallel_envs, n_steps, total_steps, log_interval, logger
):

    elapsed = time.time() - prev_time
    ups = log_interval / elapsed
    fps = ups * parallel_envs * n_steps
    env_steps = parallel_envs * n_steps * step
    mean_reward = sum(sum([ep["episode_reward"] for ep in infos]) / len(infos))

    infos.append(
        {'mean_reward': mean_reward, 'updates': step, 'environment_steps': env_steps}
    )

    logger.log_metrics(infos)

    logger.info(
        f"Updates {step}, Environment timesteps {env_steps}"
    )
    logger.info(
        f"UPS: {ups:.1f}, FPS: {fps:.1f}, ({100*step/total_steps:.2f}% completed)"
    )

    logger.info(f"Last {len(infos)} episodes with mean reward: {mean_reward:.3f}")
    logger.info("-------------------------------------------")

def _split_batch(splits):
    def thunk(batch):
        return torch.split(batch, splits, dim=-1)

    return thunk

def main(envs, logger, **cfg):
    cfg = DictConfig(cfg)

    # envs = _make_envs(cfg.env.name, cfg.parallel_envs, cfg.dummy_vecenv, cfg.env.wrappers, cfg.env.time_limit, cfg.seed)

    model = hydra.utils.instantiate(cfg.model, obs_space=envs.observation_space, action_space=envs.action_space)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    # Logging
    logger = FileSystemLogger('', cfg)
    logger.watch(model)

    # model.load_state_dict(torch.load("/home/almak/repos/blazing-ma/blazingma/ac/outputs/2021-03-06/21-37-57/model.s200000.pt"))
    # creates and initialises storage
    obs = async_reset(envs)
    parallel_envs = obs[0].shape[0]

    batch_obs = torch.zeros(cfg.n_steps + 1, parallel_envs, flatdim(envs.observation_space)) 
    batch_done = torch.zeros(cfg.n_steps + 1, parallel_envs)
    batch_act = torch.zeros(cfg.n_steps, parallel_envs, len(envs.action_space))
    batch_rew = torch.zeros(cfg.n_steps, parallel_envs, len(envs.observation_space))

    ret_ms = RunningMeanStd(shape=(len(envs.observation_space), ))

    batch_obs[0, :, :] = torch.cat([torch.from_numpy(o) for o in obs], dim=1)

    split_obs = _split_batch([flatdim(s) for s in envs.observation_space])
    split_act = _split_batch(len(envs.action_space) * [1])

    storage = defaultdict(lambda: deque(maxlen=cfg.n_steps))
    storage["info"] = deque(maxlen=20)

    start_time = time.time()

    for step in range(1, cfg.total_steps + 1):

        if cfg.video_interval and step % cfg.video_interval == 0:
            record_episodes(
                deepcopy(envs.envs[0]),
                lambda obs: [a.item() for a in model.act([torch.from_numpy(x) for x in obs])],
                cfg.video_frames,
                f"./videos/step-{step}.mp4",
            )

        if step % cfg.eval_interval == 0 and len(storage["info"]):
            _log_progress(list(storage["info"]), start_time, step, parallel_envs, cfg.n_steps, cfg.total_steps,
                          cfg.log_interval, logger)

            start_time = time.time()
            storage["info"].clear()
        
        if step % cfg.save_interval == 0:
            torch.save(model.state_dict(), f"model.s{step}.pt")

        for n in range(cfg.n_steps):
            with torch.no_grad():
                actions = model.act(split_obs(batch_obs[n, :, :]))

            obs, reward, done, info = envs.step([x.squeeze().tolist() for x in torch.cat(actions, dim=1).split(1, dim=0)])

            done = torch.tensor(done, dtype=torch.float32)
            if cfg.use_proper_termination:
                bad_done = torch.FloatTensor(
                    [1.0 if i.get("TimeLimit.truncated", False) else 0.0 for i in info]
                ).to(cfg.model.device)
                done = done - bad_done

            batch_obs[n + 1, :, :] = torch.cat([torch.from_numpy(o) for o in obs], dim=1)
            batch_act[n, :, :] = torch.cat(actions, dim=1)
            batch_done[n + 1, :] = done
            batch_rew[n, :] = torch.tensor(reward)
            storage["info"].extend([i for i in info if "episode_reward" in i])

        with torch.no_grad():
            next_value = model.get_target_value(split_obs(batch_obs[cfg.n_steps, :, :]))

        if cfg.standarize_returns:
            next_value = next_value * torch.sqrt(ret_ms.var) + ret_ms.mean

        returns = _compute_returns(batch_rew, batch_done, next_value, cfg.gamma)
        values, action_log_probs, entropy = model.evaluate_actions(split_obs(batch_obs[:-1]), split_act(batch_act))

        returns = torch.stack(returns)[:-1]

        if cfg.standarize_returns:
            ret_ms.update(returns)
            returns = (returns - ret_ms.mean) / torch.sqrt(ret_ms.var)

        advantage = returns - values

        actor_loss = (
            -(action_log_probs * advantage.detach()).sum(dim=2).mean()
            - cfg.entropy_coef * entropy
        )
        value_loss = (returns - values).pow(2).sum(dim=2).mean()

        loss = actor_loss + cfg.value_loss_coef * value_loss
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        model.soft_update(0.01)

        batch_obs[0, :, :] = batch_obs[-1, :, :]
        batch_done[0, :] = batch_done[-1, :]

    envs.close()


if __name__ == "__main__":
    main()
