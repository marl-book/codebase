import time
from collections import deque
from collections import defaultdict
from typing import DefaultDict

import numpy as np
import omegaconf
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
import hydra
from omegaconf import DictConfig

import blazingma.utils.wrappers
from blazingma.utils.wrappers import RecordEpisodeStatistics, SquashDones, TimeLimit
from blazingma.utils.loggers import Logger
from blazingma.ac.model import Policy


class Torcherize(VecEnvWrapper):
    def reset(self):
        device="cpu"
        obs = self.venv.reset()
        return [torch.from_numpy(o).to(device) for o in obs]

    def step_async(self, actions):
        actions = [a.squeeze().cpu().numpy() for a in actions]
        actions = list(zip(*actions))
        return self.venv.step_async(actions)

    def step_wait(self):
        device="cpu"
        obs, rew, done, info = self.venv.step_wait()
        return (
            [torch.from_numpy(o).float().to(device) for o in obs],
            torch.from_numpy(rew).float().to(device),
            torch.from_numpy(done).float().to(device),
            info,
        )

def _compute_returns(storage, next_value, gamma):
    returns = [next_value]
    for rew, done in zip(reversed(storage["rewards"]), reversed(storage["done"])):
        ret = returns[0] * gamma + rew * (1 - done.unsqueeze(1))
        returns.insert(0, ret)

    return returns

def _log_progress(
    infos, prev_time, step, parallel_envs, n_steps, total_steps, log_interval, logger
):

    elapsed = time.time() - prev_time
    ups = log_interval / elapsed
    fps = ups * parallel_envs * n_steps
    mean_reward = sum(sum([ep["episode_reward"] for ep in infos]) / len(infos))

    logger.info(f"Updates {step}, Environment timesteps {parallel_envs* n_steps * step}")
    logger.info(
        f"UPS: {ups:.1f}, FPS: {fps:.1f}, ({100*step/total_steps:.2f}% completed)"
    )

    logger.info(f"Last {len(infos)} episodes with mean reward: {mean_reward:.3f}")
    logger.info("-------------------------------------------")


def main(envs, logger, **cfg):
    cfg = DictConfig(cfg)

    # envs = _make_envs(cfg.env.name, cfg.parallel_envs, cfg.dummy_vecenv, cfg.env.wrappers, cfg.env.time_limit, cfg.seed)
    
    # make actor-critic model
    model = Policy(envs.observation_space, envs.action_space, cfg).to(cfg.model.device)
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr, eps=cfg.optim_eps)

    logger.watch(model)

    # model.load_state_dict(torch.load("/home/almak/repos/blazing-ma/blazingma/ac/outputs/2021-03-06/21-37-57/model.s200000.pt"))
    # creates and initialises storage
    envs = Torcherize(envs)
    obs = envs.reset()

    storage = defaultdict(lambda: deque(maxlen=cfg.n_steps))
    storage["obs"] = deque(maxlen=cfg.n_steps + 1)
    storage["done"] = deque(maxlen=cfg.n_steps + 1)
    storage["obs"].append(obs)
    storage["done"].append(torch.zeros(cfg.parallel_envs))
    storage["info"] = deque(maxlen=20)

    start_time = time.time()
    for step in range(1, cfg.total_steps + 1):

        if step % cfg.log_interval == 0 and len(storage["info"]):
            _log_progress(storage["info"], start_time, step, cfg.parallel_envs, cfg.n_steps, cfg.total_steps, cfg.log_interval, logger)
            start_time = time.time()
            storage["info"].clear()
        
        if step % cfg.save_interval == 0:
            torch.save(model.state_dict(), f"model.s{step}.pt")

        for n in range(cfg.n_steps):
            with torch.no_grad():
                actions = model.act(storage["obs"][-1])
            obs, reward, done, info = envs.step(actions)

            if cfg.use_proper_termination:
                bad_done = torch.FloatTensor(
                    [1.0 if i.get("TimeLimit.truncated", False) else 0.0 for i in info]
                ).to(cfg.model.device)
                done = done - bad_done

            storage["obs"].append(obs)
            storage["actions"].append(actions)
            storage["rewards"].append(reward)
            storage["done"].append(done)
            storage["info"].extend([i for i in info if "episode_reward" in i])

        with torch.no_grad():
            next_value = model.get_target_value(storage["obs"][-1])
        returns = _compute_returns(storage, next_value, cfg.gamma)

        input_obs = zip(*storage["obs"])
        input_obs = [torch.stack(o)[:-1] for o in input_obs]

        input_actions = zip(*storage["actions"])
        input_actions = [torch.stack(a) for a in input_actions]

        values, action_log_probs, entropy = model.evaluate_actions(
            input_obs, input_actions
        )

        returns = torch.stack(returns)[:-1]
        advantage = returns - values

        if cfg.normalize_advantages:
            mean = advantage.mean(0).mean(0)
            std = advantage.std(0).std(0)
            advantage = (advantage - mean) / (std + 1e-8)

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

    envs.close()


if __name__ == "__main__":
    main()