from pathlib import Path

import click
from omegaconf import OmegaConf

from marlbase.utils.postprocessing.load_data import load_and_group_runs


DEFAULT_METRIC = "mean_episode_returns"


@click.command()
@click.option("--source", type=click.Path(dir_okay=True, writable=False), required=True)
@click.option("--metric", type=str, default=DEFAULT_METRIC)
def run(source, metric):
    groups = load_and_group_runs(Path(source))
    assert len(groups) > 0, "No groups found"

    assert all(
        group.has_metric(metric) for group in groups
    ), f"Metric {metric} not found in all groups"

    envs = set([group.config.env.name for group in groups])

    for env in envs:
        env_groups = [group for group in groups if group.config.env.name == env]

        best_group = None
        best_value = -float("inf")

        for group in env_groups:
            values = group.get_metric(metric)
            mean = values.mean()
            if mean > best_value:
                best_group = group
                best_value = mean

        click.echo(
            "Best group for "
            + click.style(env, fg="red", bold=True)
            + " according to "
            + click.style(metric, fg="red", bold=True)
            + ": "
            + click.style(best_group.name, fg="red", bold=True)
        )

        click.echo(OmegaConf.to_yaml(best_group.config))

        click.echo(85 * "-" + "\n")


if __name__ == "__main__":
    run()
