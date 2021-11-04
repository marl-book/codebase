import os
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig

OmegaConf.register_new_resolver(
    "random",
    lambda x: os.urandom(x).hex(),
)

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):

    logger = hydra.utils.instantiate(cfg.logger, cfg=cfg.algorithm, _recursive_=False)

    env = hydra.utils.call(cfg.env, cfg.seed)

    torch.set_num_threads(1)

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    else:
        logger.warning("No seed has been set.")

    hydra.utils.call(cfg.algorithm, env, logger, _recursive_=False)

if __name__ == "__main__":

    main()
