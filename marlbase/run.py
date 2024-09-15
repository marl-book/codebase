import os

import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
import torch

OmegaConf.register_new_resolver(
    "random",
    lambda x: os.urandom(x).hex(),
)


@hydra.main(config_path="configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig):
    logger = hydra.utils.instantiate(cfg.logger, cfg=cfg.algorithm, _recursive_=False)

    env = hydra.utils.call(cfg.env, seed=cfg.seed)

    # Use singular env for evaluation/ recording
    if "parallel_envs" in cfg.env:
        del cfg.env.parallel_envs
    eval_env = hydra.utils.call(
        cfg.env,
        enable_video=True if cfg.algorithm.video_interval else False,
        seed=cfg.seed,
    )

    torch.set_num_threads(1)

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    else:
        logger.warning("No seed has been set.")

    hydra.utils.call(cfg.algorithm, env, eval_env, logger, _recursive_=False)

    return logger.get_state()


if __name__ == "__main__":
    main()
