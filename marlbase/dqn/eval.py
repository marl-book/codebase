import hydra
from omegaconf import DictConfig
import torch

from marlbase.dqn.train import record_episodes


def main(env, ckpt_path, **cfg):
    cfg = DictConfig(cfg)

    model = hydra.utils.instantiate(
        cfg.model, env.observation_space, env.action_space, cfg
    )
    print(f"Loading model from {ckpt_path}")
    state_dict = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(state_dict)

    record_episodes(
        env,
        model,
        cfg.video_frames,
        "./eval.mp4",
        cfg.model.device,
        cfg.eps_evaluation,
    )

    env.close()
