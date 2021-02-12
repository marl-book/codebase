import wandb
import logging
from omegaconf import DictConfig, OmegaConf

from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format="(%(process)d) [%(levelname).1s] - (%(asctime)s) - %(name)s >> %(message)s",
    datefmt="%m/%d %H:%M:%S",
)

class Logger:
    def __init__(self, project_name, cfg) -> None:
        self._run = wandb.init(project=project_name, config=OmegaConf.to_container(cfg))

    def log_metrics(self, d: Dict):
        self._run.log(d)