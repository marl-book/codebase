from collections import deque, defaultdict
from pathlib import Path

from gymnasium.spaces import flatdim
import hydra
from omegaconf import DictConfig
import torch

from marlbase.utils.video import record_episodes


def _log_progress(infos, step, updates, logger):
    infos.append({"updates": updates, "environment_steps": step})
    logger.log_metrics(infos)


def main(envs, eval_env, logger, **cfg):
    cfg = DictConfig(cfg)

    model = hydra.utils.instantiate(
        cfg.model, envs.single_observation_space, envs.single_action_space, cfg
    )

    # Logging
    logger.watch(model)

    # creates and initialises storage
    obs, info = envs.reset()
    n_agents = len(obs)
    parallel_envs = envs.observation_space[0].shape[0]
    obs_dim = flatdim(envs.single_observation_space)

    batch_obs = torch.zeros(
        cfg.n_steps + 1,
        parallel_envs,
        obs_dim,
        device=cfg.model.device,
    )
    batch_done = torch.zeros(cfg.n_steps + 1, parallel_envs, device=cfg.model.device)
    batch_act = torch.zeros(
        cfg.n_steps, parallel_envs, n_agents, device=cfg.model.device
    )
    batch_rew = torch.zeros(
        cfg.n_steps, parallel_envs, n_agents, device=cfg.model.device
    )

    batch_obs[0, :, :] = torch.cat([torch.from_numpy(o) for o in obs], dim=-1)

    storage = defaultdict(lambda: deque(maxlen=cfg.n_steps))
    storage["info"] = deque(maxlen=100)

    first_trigger = False

    updates = 0
    for step in range(0, cfg.total_steps + 1, cfg.n_steps * parallel_envs):
        if cfg.video_interval and step % cfg.video_interval == 0:
            record_episodes(
                eval_env,
                lambda obs: [
                    a.item() for a in model.act([torch.from_numpy(x) for x in obs])
                ],
                cfg.video_frames,
                f"./videos/step-{step}.mp4",
            )

        if len(storage["info"]) > 1 and (
            step % cfg.eval_interval == 0 or not first_trigger
        ):
            _log_progress(
                list(storage["info"]),
                step,
                updates,
                logger,
            )
            storage["info"].clear()
            first_trigger = True

        if cfg.save_interval and step % cfg.save_interval == 0:
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/model_s{step}.pt")

        for n in range(cfg.n_steps):
            with torch.no_grad():
                actions = model.act(model.split_obs(batch_obs[n, :, :]))

            obs, reward, done, truncated, info = envs.step(
                torch.stack(actions, dim=0).squeeze().tolist()
            )

            done = torch.tensor(done, dtype=torch.float32, device=cfg.model.device)
            truncated = torch.tensor(
                truncated, dtype=torch.float32, device=cfg.model.device
            )
            if cfg.use_proper_termination:
                # TODO: does this make sense?
                done = done - truncated

            batch_obs[n + 1, :, :] = torch.cat(
                [torch.from_numpy(o) for o in obs], dim=1
            )
            batch_act[n, :, :] = torch.cat(actions, dim=-1)
            batch_done[n + 1, :] = done
            batch_rew[n, :] = torch.tensor(reward)
            if "final_info" in info:
                storage["info"].extend(
                    [
                        i
                        for i in info["final_info"]
                        if i is not None and "episode_returns" in i
                    ]
                )

        model.update(batch_obs, batch_act, batch_rew, batch_done, step)
        updates += 1

        batch_obs[0, :, :] = batch_obs[-1, :, :]
        batch_done[0, :] = batch_done[-1, :]

    envs.close()
