import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
import torch


OmegaConf.register_new_resolver(
    "random",
    lambda x: os.urandom(x).hex(),
)


@hydra.main(config_path="configs", config_name="eval", version_base="1.3")
def main(cfg: DictConfig):
    path = Path(__file__).parent / cfg.path
    assert path.exists(), f"Path {path} does not exist."
    assert path.is_dir(), f"Path {path} is not a directory."

    config_path = path / "config.yaml"
    run_config = OmegaConf.load(config_path)

    assert (
        "name" in cfg.env and "time_limit" in cfg.env
    ), "Must specify env.name and env.time_limit!"
    env = hydra.utils.call(
        cfg.env,
        enable_video=True,
        seed=cfg.seed,
    )

    torch.set_num_threads(1)

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    run_config._target_ = run_config._target_.replace("train", "eval")

    if cfg.load_step is not None:
        load_step = cfg.load_step
    else:
        # Find the latest checkpoint
        load_step = max(
            [
                int(f.stem.split("_")[-1][1:])
                for f in (path / "checkpoints").glob("model_s*.pt")
            ]
        )
    ckpt_path = path / "checkpoints" / f"model_s{load_step}.pt"

    hydra.utils.call(
        run_config,
        env,
        ckpt_path,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
