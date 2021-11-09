import json
import logging
import time
from hashlib import sha256
from typing import Dict
from collections import deque, defaultdict
import pandas as pd
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


class FileSystemLogger(Logger):
    def __init__(self, project_name, cfg):
        super().__init__(project_name, cfg)

        self.file_name = 'results.csv'

    def log_metrics(self, d):
        unrolled = self._unroll_metrics(d)
        df = pd.DataFrame.from_dict([unrolled])

        # Since we are appending, we only want to write the csv headers if the file does not already exist
        # the following codeblock handles this automatically
        with open(self.file_name, 'a') as f:
            df.to_csv(f, header=f.tell() == 0, index=False)

    def _unroll_metrics(self, metrics_dict):
        unrolled = {}

        for k in metrics_dict[-1].keys():
            unrolled[k] = metrics_dict[-1][k]

        unrolled['mean_eval_episode_time'] = np.mean(
            [x['episode_time'] for x in metrics_dict if 'episode_time' in x.keys()]
        )

        averaged_rewards_per_agent = np.mean(
            np.array([x['episode_reward'] for x in metrics_dict if 'episode_reward' in x.keys()]), axis=0
        )

        for enum, mean in enumerate(averaged_rewards_per_agent):
            unrolled[f'agent_{enum}_mean_reward'] = mean

        ordered_keys = sorted(list(unrolled.keys()))
        ordered_keys.pop(ordered_keys.index('environment_steps'))
        ordered_keys = ['environment_steps'] + ordered_keys

        unrolled_orderd = {}

        for k in ordered_keys:
            unrolled_orderd[k] = unrolled[k]

        return unrolled_orderd

    def watch(self, model):
        pass
