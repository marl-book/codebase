import time
from collections import deque
from collections import defaultdict
from itertools import product
from typing import DefaultDict

from gym.spaces import flatdim
import numpy as np
import torch
from torch.nn import functional as F
from omegaconf import DictConfig

from blazingma.ac_maxq.model import Policy
from blazingma.utils.standarize_stream import RunningMeanStd
from blazingma.utils.envs import async_reset

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
    mean_reward = sum(sum([ep["episode_reward"] for ep in infos]) / len(infos))

    logger.log_metrics(infos)

    logger.info(f"Updates {step}, Environment timesteps {parallel_envs* n_steps * step}")
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
    
    # make actor-critic model
    model = Policy(envs.observation_space, envs.action_space, cfg).to(cfg.model.device)
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr, eps=cfg.optim_eps)

    logger.watch(model)

    # model.load_state_dict(torch.load("/home/almak/repos/blazing-ma/blazingma/ac/outputs/2021-03-06/21-37-57/model.s200000.pt"))
    # creates and initialises storage
    obs = async_reset(envs)
    parallel_envs = obs[0].shape[0]

    agent_count = len(envs.action_space)

    batch_obs = torch.zeros(cfg.n_steps + 1, parallel_envs, flatdim(envs.observation_space)) 
    batch_done = torch.zeros(cfg.n_steps + 1, parallel_envs)
    batch_act = torch.zeros(cfg.n_steps, parallel_envs, agent_count)
    batch_rew = torch.zeros(cfg.n_steps, parallel_envs, agent_count)

    ret_ms = RunningMeanStd(shape=(agent_count, ))

    batch_obs[0, :, :] = torch.cat([torch.from_numpy(o) for o in obs], dim=1)

    split_obs = _split_batch([flatdim(s) for s in envs.observation_space])
    split_act = _split_batch(len(envs.action_space) * [1])

    storage = defaultdict(lambda: deque(maxlen=cfg.n_steps))
    storage["info"] = deque(maxlen=20)

    start_time = time.time()
    for step in range(1, cfg.total_steps + 1):

        if step % cfg.log_interval == 0 and len(storage["info"]):
            _log_progress(storage["info"], start_time, step, parallel_envs, cfg.n_steps, cfg.total_steps, cfg.log_interval, logger)
            start_time = time.time()
            storage["info"].clear()
        
        if step % cfg.save_interval == 0:
            torch.save(model.state_dict(), f"model.s{step}.pt")

        for n in range(cfg.n_steps):
            with torch.no_grad():
                actions = model.act(split_obs(batch_obs[n, :, :]))

            obs, reward, done, info = envs.step([x.squeeze().tolist() for x in torch.cat(actions, dim=1).split(1, dim=0)])
            # envs.envs[0].render()
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
            next_value = model.get_target_value(agent_count*[batch_obs[cfg.n_steps, :, :]])

        if cfg.standarize_returns:
            next_value = next_value * torch.sqrt(ret_ms.var) + ret_ms.mean

        q_input = []
        # hard part is to "remove" the actions that aren't the agents'
        one_hot_actions = split_act(batch_act.long())
        one_hot_actions = [F.one_hot(act.view(-1), num_classes=envs.action_space[i].n).view(cfg.n_steps, parallel_envs, -1) for i, act in enumerate(one_hot_actions)]
        q_input = [torch.cat([batch_obs[:-1]] + [act for i, act in enumerate(one_hot_actions) if i != j], dim=2) for j in range(agent_count)]

        q_out = model.critic(q_input)
        q_out = torch.cat([torch.gather(q, -1, a) for q, a in zip(q_out, split_act(batch_act.long()))], dim=-1)

        returns = _compute_returns(batch_rew, batch_done, next_value, cfg.gamma)
        values, action_log_probs, entropy = model.evaluate_actions(split_obs(batch_obs[:-1]), split_act(batch_act), state=agent_count*[batch_obs[:-1]])
        returns = torch.stack(returns)[:-1]

        if cfg.standarize_returns:
            ret_ms.update(returns)
            returns = (returns - ret_ms.mean) / torch.sqrt(ret_ms.var)

        # print q table in 3x3 matrix game
        # table = model.critic([torch.cat([torch.zeros(3,4), torch.eye(3)], dim=-1)]*2)
        # if step % 200 == 0:
        #     print("Agent 0:")
        #     print(table[0].T.int())
        #     print("Agent 1:")
        #     print(table[1].int())

        q_all_input = []
        for i in range(agent_count):
            other_acts = [a.n for j, a in enumerate(envs.action_space) if i != j]
            x = batch_obs[:-1].repeat(np.prod(other_acts), 1, 1, 1)

            y = [torch.cat(x) for x in product(*[torch.eye(j) for j in other_acts])]
            y = torch.stack(y)

            y = y.unsqueeze(1).unsqueeze(1).repeat(1, cfg.n_steps, parallel_envs, 1)
            z = torch.cat([x, y], dim=-1)
            q_all_input += [z]

        q_all = model.critic(q_all_input)

        q_all = [torch.gather(q, -1, a.repeat(q.shape[0], 1, 1, 1)) for q, a in zip(q_all, split_act(batch_act.long()))]

        # mixed = model.mixing([t.T.detach() for t in q_all])
        # mixed = torch.cat(mixed, dim=-1).squeeze()

        # q_all = mixed
        # q_all = torch.cat([torch.max(q, dim=0)[0] for q in q_all], dim=-1)
        
        # gate_action = torch.distributions.Categorical(torch.tensor([model.gate, 1-model.gate])).sample()
        storage["gate_rewards"].append(returns.mean())
        
        # print(q_all.shape)
        q_all = F.sigmoid(model.gate) * torch.cat([torch.max(q, dim=0)[0] for q in q_all], dim=-1) + \
                (1 - F.sigmoid(model.gate)) * torch.cat([torch.min(q, dim=0)[0] for q in q_all], dim=-1)

        if step % 500 == 0:
            print(F.sigmoid(model.gate))
        # advantage = returns - values
        advantage = q_all - values

        actor_loss = (
            -(action_log_probs * advantage.detach()).sum(dim=2).mean()
            - cfg.entropy_coef * entropy
        )
        value_loss = (returns - values).pow(2).sum(dim=2).mean()
        q_loss = (returns - q_out).pow(2).sum(dim=2).mean()

        gate_loss = -torch.mean(torch.log(F.sigmoid(model.gate)) * returns)
        loss = actor_loss + cfg.value_loss_coef * value_loss + q_loss + gate_loss
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