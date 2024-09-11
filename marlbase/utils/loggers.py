from datetime import timedelta
from hashlib import sha256
import json
import logging
import math
import time
from typing import Dict, List

import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd


def squash_info(info):
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")
    keys.discard("terminal_observation")
    for key in keys:
        values = [d[key] for d in info if key in d]
        if len(values) == 1:
            new_info[key] = values[0]
            continue

        mean = np.mean([np.array(v).sum() for v in values])
        std = np.std([np.array(v).sum() for v in values])

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

        self._total_steps = cfg.total_steps
        self._start_time = time.time()
        self._prev_time = None
        self._prev_steps = (0, 0)  # steps (updates) and env_samples

    def log_metrics(self, metrics: List[Dict]): ...

    def print_progress(self, updates, steps, mean_returns, episodes):
        self.info(f"Updates {updates}, Environment timesteps {steps}")

        time_now = time.time()

        elapsed_wallclock = time_now - self._prev_time[0] if self._prev_time else None
        elapsed_cpu = (
            time.process_time() - self._prev_time[1] if self._prev_time else None
        )
        elapsed_from_start = timedelta(seconds=math.ceil((time_now - self._start_time)))

        completed = steps / self._total_steps

        if elapsed_wallclock:
            ups = (updates - self._prev_steps[0]) / elapsed_wallclock
            fps = (steps - self._prev_steps[1]) / elapsed_wallclock
            self.info(f"UPS: {ups:.2f}, FPS: {fps:.2f} (wall time)")

            # ups = (updates - self._prev_steps[0]) / elapsed_cpu
            # fps = (steps - self._prev_steps[1]) / elapsed_cpu
            # self.info(f"UPS: {ups:.2f}, FPS: {fps:.2f} (cpu time)")

            eta = elapsed_from_start * (1 - completed) / completed
            eta = timedelta(seconds=math.ceil(eta.total_seconds()))
            self.info(f"Elapsed Time: {elapsed_from_start}")
            self.info(f"Estim. Time Left: {eta}")

        self.info(f"Completed: {100*completed:.2f}%")

        self._prev_steps = (updates, steps)
        self._prev_time = time.time(), time.process_time()

        self.info(f"Last {episodes} episodes with mean returns: {mean_returns:.3f}")
        self.info("-------------------------------------------")

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

    def get_state(self):
        return None


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

    def log_metrics(self, metrics: List[Dict]):
        d = squash_info(metrics)
        self._run.log(d)

        self.print_progress(
            d["updates"],
            d["environment_steps"],
            d["mean_episode_returns"],
            len(metrics) - 1,
        )

    def watch(self, model):
        self.debug(model)
        self._run.watch(model)


class FileSystemLogger(Logger):
    def __init__(self, project_name, cfg):
        super().__init__(project_name, cfg)

        self.results_path = "results.csv"
        self.config_path = "config.yaml"
        with open(self.config_path, "w") as f:
            OmegaConf.save(cfg, f)

    def log_metrics(self, metrics):
        d = squash_info(metrics)
        df = pd.DataFrame.from_dict([d])[
            ["environment_steps"]
            + sorted([k for k in d.keys() if k != "environment_steps"])
        ]
        # Since we are appending, we only want to write the csv headers if the file does not already exist
        # the following codeblock handles this automatically
        with open(self.results_path, "a") as f:
            df.to_csv(f, header=f.tell() == 0, index=False)

        self.print_progress(
            d["updates"],
            d["environment_steps"],
            d["mean_episode_returns"],
            len(metrics) - 1,
        )

    def get_state(self):
        df = pd.read_csv(self.results_path, index_col=0)
        return df
