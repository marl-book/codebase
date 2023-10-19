from collections import deque
from collections import defaultdict
from copy import deepcopy

from gym.spaces import flatdim
import hydra
from omegaconf import DictConfig
import torch

from fastmarl.utils.envs import async_reset
from fastmarl.utils.video import record_episodes


def _log_progress(
    infos, step, parallel_envs, n_steps, total_steps, eval_interval, logger
):
    env_steps = parallel_envs * n_steps * step
    infos.append(
        {'updates': step, 'environment_steps': env_steps}
    )
    logger.log_metrics(infos)


def main(envs, logger, **cfg):
    cfg = DictConfig(cfg)

    model = hydra.utils.instantiate(cfg.model, envs.observation_space, envs.action_space, cfg)

    # Logging
    logger.watch(model)

    # creates and initialises storage
    obs = async_reset(envs)
    parallel_envs = obs[0].shape[0]

    batch_obs = torch.zeros(cfg.n_steps + 1, parallel_envs, flatdim(envs.observation_space), device=cfg.model.device) 
    batch_done = torch.zeros(cfg.n_steps + 1, parallel_envs, device=cfg.model.device)
    batch_act = torch.zeros(cfg.n_steps, parallel_envs, len(envs.action_space), device=cfg.model.device)
    batch_rew = torch.zeros(cfg.n_steps, parallel_envs, len(envs.observation_space), device=cfg.model.device)

    batch_obs[0, :, :] = torch.cat([torch.from_numpy(o) for o in obs], dim=1)

    storage = defaultdict(lambda: deque(maxlen=cfg.n_steps))
    storage["info"] = deque(maxlen=20)

    first_trigger = False

    for step in range(cfg.total_steps + 1):
        if cfg.video_interval and step % cfg.video_interval == 0:
            record_episodes(
                deepcopy(envs.envs[0]),
                lambda obs: [a.item() for a in model.act([torch.from_numpy(x) for x in obs])],
                cfg.video_frames,
                f"./videos/step-{step}.mp4",
            )

        if len(storage["info"]) > 1 and (step % cfg.eval_interval == 0 or not first_trigger):
            _log_progress(list(storage["info"]), step, parallel_envs, cfg.n_steps, cfg.total_steps,
                          cfg.eval_interval, logger)
            storage["info"].clear()
            first_trigger = True
        
        if step % cfg.save_interval == 0:
            torch.save(model.state_dict(), f"model_s{step}.pt")

        for n in range(cfg.n_steps):
            with torch.no_grad():
                actions = model.act(model.split_obs(batch_obs[n, :, :]))

            obs, reward, done, info = envs.step([x.squeeze().tolist() for x in torch.cat(actions, dim=1).split(1, dim=0)])

            done = torch.tensor(done, dtype=torch.float32)
            if cfg.use_proper_termination:
                bad_done = torch.FloatTensor(
                    [1.0 if i.get("TimeLimit.truncated", False) else 0.0 for i in info], device=cfg.model.device
                )
                done = done - bad_done

            batch_obs[n + 1, :, :] = torch.cat([torch.from_numpy(o) for o in obs], dim=1)
            batch_act[n, :, :] = torch.cat(actions, dim=1)
            batch_done[n + 1, :] = done
            batch_rew[n, :] = torch.tensor(reward)
            storage["info"].extend([i for i in info if "episode_returns" in i])

        model.update(batch_obs, batch_act, batch_rew, batch_done, step)
        
        batch_obs[0, :, :] = batch_obs[-1, :, :]
        batch_done[0, :] = batch_done[-1, :]

    envs.close()


if __name__ == "__main__":
    main()
