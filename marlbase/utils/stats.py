import os

import numpy as np
import pandas as pd


def _load_data_from_subfolder(folder, metric, step=None, step_metric=None):
    """Helper function for pulling results data from logs

    Args:
        folder (str):
        metric (str):
        step (int):
        step_metric (str):

    Returns:
        list of performance values
    """
    # The given folder will contain several sub-folders with random hashes like "1a8fdsk3"
    # Within each sub-folder is the data we need
    results = []

    for subfolder in os.listdir(folder):
        data = pd.read_csv(f'{os.path.join(folder, subfolder, "results.csv")}')

        if step is not None and step_metric is not None:
            data = [data[data[step_metric] == step][metric].tolist()[0]]

        else:
            data = data[metric].tolist()

        results.append(data)

    return results


def make_agg_metrics_intervals(folders, algos, metric, step=None, step_metric=None):
    """Pulls results for the 'Aggregate metrics with 95% Stratified Bootstrap CIs' plot
    Can also be used for "Performance Profiles" plot

    Below is an example usage for this function:
        make_agg_metrics_intervals(
            folders=[folder, folder, folder, folder],
            algos=['ac', 'ac', 'dqn', 'dqn'],
            metric=['mean_reward', 'mean_reward', 'mean_reward', 'mean_reward'],
            step=[240, 240, 500, 500],
            step_metric=['environment_steps', 'environment_steps', 'updates', 'updates']
        )

    Shape of the output data is {'algo_1': (n_runs x n_envs), ..., 'algo_j': (n_runs x n_envs}

    Args:
        folders (List[str]):
        algos (List[str]):
        metric (List[str]):
        step (List[int]):
        step_metric (List[str]):

    Returns:
        Dict of performance matrices
    """
    # For the interval estimates plot, we need performance at a specific point during training/evaluation
    if step is None:
        raise ValueError('For interval plots, a specific step must be specified')
    if step_metric is None:
        raise ValueError('For interval plots, a specific step_metric must be specified')

    # Process for reading in the data
    results = {}

    for i in range(len(folders)):
        data = _load_data_from_subfolder(os.path.join(folders[i], algos[i]), metric[i], step[i], step_metric[i])

        if algos[i] not in results.keys():
            results[algos[i]] = []

        results[algos[i]].append(data)

    # Now we need to transpose the pulled results into results matrices. For specific shape, see function docstring
    results_T = {}

    for algo in results.keys():
        pulled_results = results[algo]
        results_T[algo] = np.array(pulled_results).T[0]

    return results_T


def make_agg_metrics_pxy(folders, algos, metric, step=None, step_metric=None):
    """Pulls results for the 'Probability of Improvement' plot

    Below is an example usage for this function:
        make_agg_metrics_pxy(
            folders=[folder, folder, folder, folder],
            algos=['ac', 'ac', 'dqn', 'dqn'],
            metric=['mean_reward', 'mean_reward', 'mean_reward', 'mean_reward'],
            step=[240, 240, 500, 500],
            step_metric=['environment_steps', 'environment_steps', 'updates', 'updates']
        )

    Shape of the output data is {'algo_1,algo_2': ((n_runs x n_envs), (n_runs x n_envs)), ...}

    Args:
        folders (List[str]):
        algos (List[str]):
        metric (List[str]):
        step (List[int]):
        step_metric (List[str]):

    Returns:
        Dicts of comparative performance matrices
    """
    # First pulling the metrics as we would for other single-value plots
    agg_metrics = make_agg_metrics_intervals(folders=folders, algos=algos, metric=metric,
                                             step=step, step_metric=step_metric)

    # Now building out the combinatorics dict
    results = {}

    for i in range(len(algos)):
        for j in range(len(algos)):
            if i == j:
                continue
            results[f'{algos[i]},{algos[j]}'] = (agg_metrics[algos[i]], agg_metrics[algos[j]])

    return results


def make_agg_metrics_efficiency(folders, algos, metric):
    """Pulls results for the 'Aggregate metrics with 95% Stratified Bootstrap CIs' plot
    Can also be used for "Performance Profiles" plot

    Below is an example usage for this function:
        make_agg_metrics_efficiency(
            folders=[folder, folder, folder, folder],
            algos=['ac', 'ac', 'dqn', 'dqn'],
            metric=['mean_reward', 'mean_reward', 'mean_reward', 'mean_reward'],
        )

    Shape of the output data is {'algo_1': (n_runs x n_envs x n_steps), ...,}

    Args:
        folders (List[str]):
        algos (List[str]):
        metric (List[str]):
        step (List[int]):
        step_metric (List[str]):

    Returns:
        Dict of performance matrices
    """
    step = [None for _ in range(len(algos))]
    step_metric = [None for _ in range(len(algos))]

    # Process for reading in the data
    results = {}

    for i in range(len(folders)):
        data = _load_data_from_subfolder(os.path.join(folders[i], algos[i]), metric[i], step[i], step_metric[i])

        if algos[i] not in results.keys():
            results[algos[i]] = []

        results[algos[i]].append(data)

    results_T = {}

    for algo in results.keys():
        pulled_results = results[algo]

        n_envs = len(pulled_results)
        n_runs = len(pulled_results[0])
        n_steps = len(pulled_results[0][0])


        results_T[algo] = np.array(pulled_results).reshape((n_runs, n_envs, n_steps))

    return results_T
