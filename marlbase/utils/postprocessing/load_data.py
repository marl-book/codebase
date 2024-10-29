from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from omegaconf import OmegaConf
import pandas as pd


def _load_csv(path: Path) -> Dict[str, List[float]]:
    assert (
        path.exists() and path.is_file() and path.suffix == ".csv"
    ), f"{path} is not a valid csv file"
    return pd.read_csv(path).to_dict(orient="list")


def _load_config(path: Path) -> OmegaConf:
    assert (
        path.exists() and path.is_file() and path.suffix == ".yaml"
    ), f"{path} is not a valid config file"
    return OmegaConf.load(path)


class Run:
    def __init__(self, config: OmegaConf, data: Dict[str, List[float]], path: Path):
        self.data = data
        self.config = config
        self.path = path

    def __from_path__(path: Path) -> "Run":
        assert path.exists() and path.is_dir(), f"{path} is not a valid run directory"
        data = _load_csv(path / "results.csv")
        config = _load_config(path / "config.yaml")
        return Run(config, data, path)

    def __str__(self) -> str:
        return f"Run {self.path}"

    def get_config_name(self) -> str:
        return " ".join(
            [f"{key}={value}" for key, value in self.config.items() if key != "seed"]
        )


def _load_run(path: Path):
    return Run.__from_path__(path)


class Group:
    def __init__(self, name, runs: List[Run]):
        self.name = name
        self.config = runs[0].config
        self.config.pop("seed")
        self.runs = runs

    def __str__(self) -> str:
        return f"Group {self.name} ({len(self.runs)} runs)"

    def has_metric(self, key) -> bool:
        has_metrics = [key in run.data for run in self.runs]
        assert all(has_metrics) or not any(
            has_metrics
        ), f"Key {key} is present in some but not all runs"
        return has_metrics[0]

    def get_metric(self, key) -> np.ndarray:
        assert self.has_metric(key), f"Key {key} is not present in all runs"
        values = [run.data[key] for run in self.runs]
        assert all(
            len(value) == len(values[0]) for value in values
        ), f"Values for key {key} have different lengths"
        return np.array(values)


def _load_runs(path: Path) -> List[Run]:
    assert path.exists() and path.is_dir(), f"{path} is not a valid directory"
    runs = []
    for run in path.glob("**/results.csv"):
        run = _load_run(run.parent)
        runs.append(run)
    return runs


def _flatten_omegaconf(
    config: OmegaConf, base_name=None
) -> Dict[str, Union[str, float, int]]:
    flat_config = {}
    for key, value in config.items():
        key = f"{base_name}.{key}" if base_name else key
        if OmegaConf.is_config(value) and not OmegaConf.is_list(value):
            flat_config.update(_flatten_omegaconf(value, key))
        else:
            flat_config[key] = value
    return flat_config


def load_and_group_runs(path: Path, minimal_name: bool = True) -> List[Group]:
    """
    Load all runs in a directory and group them by unique configurations
    :param path: Path to directory containing runs
    :param minimal_name: Use minimal name for each group
    :return: List of Group objects
    """
    # group runs by unique confrigurations
    runs_by_config_name = defaultdict(list)
    for run in _load_runs(path):
        runs_by_config_name[run.get_config_name()].append(run)

    if minimal_name:
        # identify minimal hyperparameters that differentiate runs
        values_by_key = defaultdict(set)
        for config_name, runs in runs_by_config_name.items():
            group_config = _flatten_omegaconf(runs[0].config)
            for key, value in group_config.items():
                if (
                    key == "seed"
                    or key == "algorithm.name"
                    or "_target_" in key
                    or key == "hypergroup"
                    or "wrappers" in key
                ):
                    continue
                values_by_key[key].add(value)

        # distinguishing hyperparameters
        distinguishing_keys = [
            key for key, values in values_by_key.items() if len(values) > 1
        ]

        runs_by_minimal_config_name = {}
        for runs in runs_by_config_name.values():
            group_config = _flatten_omegaconf(runs[0].config)
            minimal_config_name = group_config["algorithm.name"].upper()
            config_name = " ".join(
                [
                    f"{key}={group_config[key]}"
                    for key in distinguishing_keys
                    if key in group_config
                ]
            )
            if config_name:
                minimal_config_name += f" ({config_name})"
            runs_by_minimal_config_name[minimal_config_name] = runs

        runs_by_config_name = runs_by_minimal_config_name

    return [Group(name, runs) for name, runs in runs_by_config_name.items()]
