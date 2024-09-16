from pathlib import Path

import click
import matplotlib.pyplot as plt
import seaborn as sns

from marlbase.utils.postprocessing.load_data import load_and_group_runs


DEFAULT_METRIC = "mean_episode_returns"


@click.command()
@click.option("--source", type=click.Path(dir_okay=True, writable=False), required=True)
@click.option("--minimal-name", type=bool, default=True)
@click.option("--metric", type=str, default=DEFAULT_METRIC)
@click.option("--save_path", type=click.Path(dir_okay=True, writable=True))
def run(source, minimal_name, metric, save_path):
    groups = load_and_group_runs(Path(source), minimal_name)
    assert len(groups) > 0, "No groups found"

    click.echo(f"Loaded {len(groups)} groups:")
    for group in groups:
        click.echo(f"\t{group.name} with {len(group.runs)} runs")

    assert all(
        group.has_metric(metric) for group in groups
    ), f"Metric {metric} not found in all groups"

    envs = set([group.config.env.name for group in groups])

    for env in envs:
        env_groups = [group for group in groups if group.config.env.name == env]

        sns.set_style("whitegrid")
        plt.figure()
        for group in env_groups:
            steps = group.get_metric("environment_steps").mean(axis=0)
            values = group.get_metric(metric)
            means = values.mean(axis=0)
            stds = values.std(axis=0)
            plt.plot(steps, means, label=group.name)
            plt.fill_between(
                steps,
                means - stds,
                means + stds,
                alpha=0.3,
            )
        plt.legend()
        plt.xlabel("Environment steps")
        plt.ylabel(metric)
        plt.title(env)
        if save_path:
            path = Path(save_path) / f"{env.replace('/', ':')}_{metric}.pdf"
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path)
        plt.show()


if __name__ == "__main__":
    run()
