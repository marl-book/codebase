import json
import logging
import time
from hashlib import sha256
from typing import Dict
from collections import deque, defaultdict

import numpy as np
from hydra.conf import HydraConf
from omegaconf import DictConfig, OmegaConf

def squash_info(info):
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")
    keys.discard("terminal_observation")
    for key in keys:
        mean = np.mean([np.array(d[key]).sum() for d in info if key in d])
        std = np.std([np.array(d[key]).sum() for d in info if key in d])

        split_key = key.rsplit("/", 1)
        mean_key = split_key[:]
        std_key = split_key[:]
        mean_key[-1] = "mean_" + mean_key[-1]
        std_key[-1] = "std_" + std_key[-1]
        
        new_info["/".join(mean_key)] = mean
        new_info["/".join(std_key)] = std
    return new_info


class Logger:
    def __init__(self, project_name, cfg: DictConfig) -> None:
        self.config_hash = sha256(
            json.dumps(
                {k: v for k, v in OmegaConf.to_container(cfg).items() if k != "seed"},
                sort_keys=True,
            ).encode("utf8")
        ).hexdigest()[-10:]

    def log_metrics(self, d: Dict):
        ...

    def watch(self, model):
        self.debug(model)

    def debug(self, *args, **kwargs):
        return logging.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        return logging.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        return logging.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        return logging.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        return logging.critical(*args, **kwargs)


class WandbLogger(Logger):
    def __init__(self, project_name, cfg: DictConfig) -> None:
        import wandb

        super().__init__(project_name, cfg)
        self._run = wandb.init(
            project=project_name,
            config=OmegaConf.to_container(cfg),
            monitor_gym=True,
            group=self.config_hash,
        )

    def log_metrics(self, d: Dict):
        if type(d) in (list, tuple, deque):
            self._run.log(squash_info(d))
        else:
            self._run.log(d)

    def watch(self, model):
        self.debug(model)
        self._run.watch(model)
