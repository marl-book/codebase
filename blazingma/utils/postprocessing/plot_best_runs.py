from pathlib import Path

import pandas as pd
import click
import yaml
from munch import munchify
from hashlib import sha256
from collections import defaultdict
import json
import matplotlib.pyplot as plt


def _plot_best_runs(df):
    # calculate mean and std
    # transposing because agg doesn't work on axis=1 for some reason
    df = df.T.groupby(axis=0, level=[0, 1]).agg(["mean", "std"]).T

    mean = df.loc[(slice(None), "mean"), :].droplevel(1)
    shade_min = mean - df.loc[(slice(None), "std"), :].droplevel(1)
    shade_max = mean + df.loc[(slice(None), "std"), :].droplevel(1)

    for env in set(df.columns.get_level_values(0)):
        for alg in set(df.columns.get_level_values(1)):
            idx = (env, alg)
            plt.plot(mean[idx], label=alg.upper())
            plt.fill_between(mean[idx].index, shade_min[idx], shade_max[idx], alpha=0.3)
        plt.legend()
        plt.title(env)
        plt.show()


@click.command()
@click.option("--exported-file", type=click.Path(dir_okay=False, writable=False))
@click.pass_context
def run(ctx, exported_file):

    df = pd.read_hdf(exported_file, "df")
    configs = pd.read_hdf(exported_file, "configs")

    best_hash = (
        df.groupby(axis=1, level=[0, 1, 2]).mean().max().groupby(level=[0, 1]).idxmax()
    )
    df = pd.concat(
        [df.xs(h, level=[0, 1, 2], axis=1, drop_level=False) for h in best_hash.values],
        axis=1,
    ).droplevel(2, axis=1)

    _plot_best_runs(df)


if __name__ == "__main__":
    run()
