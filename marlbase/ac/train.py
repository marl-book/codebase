from collections import namedtuple
from pathlib import Path

from einops import rearrange
import numpy as np
from gymnasium.spaces import flatdim
import hydra
from omegaconf import DictConfig
import torch

from marlbase.utils.video import VideoRecorder


Batch = namedtuple(
    "Batch", ["obss", "actions", "rewards", "dones", "filled", "action_masks"]
)


def _log_progress(infos, step, updates, logger):
    infos.append({"updates": updates, "environment_steps": step})
    logger.log_metrics(infos)


def _collect_trajectories(
    envs, model, max_ep_length, parallel_envs, n_agents, device, use_proper_termination
):
    running = torch.ones(parallel_envs, device=device, dtype=torch.bool)

    # creates and initialises storage
    obss, info = envs.reset()
    obss = [torch.tensor(o, device=device) for o in obss]
    parallel_envs = envs.observation_space[0].shape[0]
    obs_dim = flatdim(envs.single_observation_space)
    num_actions = max(action_space.n for action_space in envs.single_action_space)

    batch_obs = torch.zeros(
        max_ep_length + 1,
        parallel_envs,
        obs_dim,
        device=device,
    )
    batch_done = torch.zeros(
        max_ep_length + 1, parallel_envs, device=device, dtype=torch.bool
    )
    batch_act = torch.zeros(
        max_ep_length, parallel_envs, n_agents, device=device, dtype=torch.long
    )
    batch_rew = torch.zeros(max_ep_length, parallel_envs, n_agents, device=device)
    batch_filled = torch.zeros(max_ep_length, parallel_envs, device=device)
    t = 0
    infos = []

    # if the environment provides action masks, use them
    if "action_mask" in info:
        batch_action_masks = torch.ones(
            max_ep_length + 1, parallel_envs, n_agents, num_actions, device=device
        )
        mask = np.stack(info["action_mask"], dtype=np.float32)
        action_mask = torch.tensor(mask, dtype=torch.float32, device=device)
        batch_action_masks[0] = action_mask
        action_mask = action_mask.swapaxes(0, 1)
    else:
        batch_action_masks = None
        action_mask = None

    # set initial obs
    batch_obs[0] = torch.cat(obss, dim=-1)

    actor_hiddens = model.init_actor_hiddens(parallel_envs)

    while running.any():
        with torch.no_grad():
            actions, actor_hiddens = model.act(
                obss,
                actor_hiddens,
                action_mask=action_mask,
            )

        next_obss, rewards, done, truncated, info = envs.step(
            actions.squeeze().tolist()
        )
        next_obss = [torch.tensor(o, device=device) for o in next_obss]

        done = torch.tensor(done, dtype=torch.bool, device=device)
        truncated = torch.tensor(truncated, dtype=torch.bool, device=device)
        if not use_proper_termination:
            # TODO: does this make sense?
            done = torch.logical_or(done, truncated)

        batch_obs[t + 1, running, :] = torch.cat(next_obss, dim=1)[running]
        batch_act[t, running] = rearrange(actions, "N B 1 -> B N")[running]
        batch_done[t + 1, running] = done[running]
        batch_rew[t, running] = torch.tensor(rewards, dtype=torch.float32)[running]
        batch_filled[t, running] = 1
        if "action_mask" in info:
            mask = np.stack(info["action_mask"], dtype=np.float32)
            action_mask = torch.tensor(mask, dtype=torch.float32, device=device)
            batch_action_masks[t + 1, running] = action_mask[running]
            action_mask = action_mask.swapaxes(0, 1)

        if done.any():
            for i, d in enumerate(done):
                if d:
                    assert (
                        "final_info" in info
                        and info["final_info"][i] is not None
                        and "episode_returns" in info["final_info"][i]
                    ), "sanity check"
                    infos.append(info["final_info"][i])
                    running[i] = False

        t += 1
        obss = next_obss

    batch = Batch(
        batch_obs, batch_act, batch_rew, batch_done, batch_filled, batch_action_masks
    )

    return t, batch, infos


def record_episodes(env, model, n_timesteps, path, device):
    recorder = VideoRecorder()
    done = True

    for _ in range(n_timesteps):
        if done:
            obss, info = env.reset()
            hiddens = model.init_actor_hiddens(1)
            if "action_mask" in info:
                action_mask = torch.tensor(
                    info["action_mask"], dtype=torch.float32, device=device
                )
            else:
                action_mask = None
            done = False
        else:
            with torch.no_grad():
                obss = torch.tensor(obss, dtype=torch.float32, device=device).unsqueeze(
                    1
                )
                actions, hiddens = model.act(obss, hiddens, action_mask)
                obss, _, done, truncated, info = env.step([a.item() for a in actions])
                if "action_mask" in info:
                    action_mask = torch.tensor(
                        info["action_mask"], dtype=torch.float32, device=device
                    )
            done = done or truncated
        recorder.record_frame(env)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    recorder.save(path)


def main(envs, eval_env, logger, time_limit, **cfg):
    cfg = DictConfig(cfg)

    model = hydra.utils.instantiate(
        cfg.model, envs.single_observation_space, envs.single_action_space, cfg
    )
    logger.watch(model)

    parallel_envs = envs.observation_space[0].shape[0]

    step = 0
    updates = 0
    last_eval = 0
    last_save = 0
    last_video = 0
    while step < cfg.total_steps + 1:
        t, batch, infos = _collect_trajectories(
            envs,
            model,
            time_limit,
            parallel_envs,
            model.n_agents,
            cfg.model.device,
            cfg.use_proper_termination,
        )

        metrics = model.update(batch, step)
        infos.append(metrics)

        if (step - last_eval) >= cfg.eval_interval:
            _log_progress(infos, step, updates, logger)
            last_eval = step

        if cfg.save_interval and (step - last_save) >= cfg.save_interval:
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/model_s{step}.pt")
            last_save = step

        if cfg.video_interval and (step - last_video) >= cfg.video_interval:
            record_episodes(
                eval_env,
                model,
                cfg.video_frames,
                f"./videos/step-{step}.mp4",
                cfg.model.device,
            )
            last_video = step

        updates += 1
        step += t * parallel_envs

    envs.close()
