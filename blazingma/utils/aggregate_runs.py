from pathlib import Path

import pandas as pd
import click
import yaml
from munch import munchify
from hashlib import sha256
from collections import defaultdict
import json
import matplotlib.pyplot as plt


def _load_data(folder):
    path = Path(folder)
    config_files = path.glob("**/**/.hydra/config.yaml")

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seed_data = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    algos = set()
    envs = set()

    hash_to_config = defaultdict(list)

    for config_path in config_files:
        # hash = config_path.parent.parent.name

        with config_path.open() as fp:
            config = munchify(yaml.safe_load(fp))

        env = config.env.name
        try:
            env = env.split(":")[1] # remove the library
        except IndexError:
            pass

        algo = config.algorithm.name

        algos.add(algo)
        envs.add(env)


        seed = config.seed
        del config.seed

        raw_data = json.dumps(config, sort_keys=True).encode("utf8")
        hash = sha256(raw_data).hexdigest()[:12]

        hash_to_config[hash].append(config_path)

        df = pd.read_csv(config_path.parent.parent / "results.csv", index_col=0)["mean_reward"]
        data[env][algo][hash].append(df.rename(f"seed={seed}"))
        assert seed not in seed_data[env][algo][hash], "Duplicate seed"
        seed_data[env][algo][hash].add(seed)

    env_df_list = []
    for env in data.keys():
        algo_df_list = []
        for algo in data[env].keys():
            lst = []
            for hash in data[env][algo].keys():
                lst.append(pd.concat(data[env][algo][hash], axis=1))
            df = pd.concat(lst, axis=1, keys=[h for h in data[env][algo].keys()])
            algo_df_list.append(df)
        df = pd.concat(algo_df_list, axis=1, keys=data[env].keys())
        env_df_list.append(df)
    df = pd.concat(env_df_list, axis=1, keys=data.keys())


    return hash_to_config, df


def _print_best_config(hash_to_config, best_hash):
    best_hash = best_hash.values

    for (env, alg, hash) in best_hash:
        print(f"{env}: {alg} best config is:")
        with hash_to_config[hash][0].open() as fp:
            config = munchify(yaml.safe_load(fp))
            print(config)
            print("---")
        
def _plot_best_runs(df):
    # calculate mean and std
    # transposing because agg doesn't work on axis=1 for some reason
    df = df.T.groupby(axis=0, level=[0,1]).agg(["mean", "std"]).T

    mean = df.loc[(slice(None), "mean"), :].droplevel(1)
    shade_min = mean - df.loc[(slice(None), "std"), :].droplevel(1)
    shade_max = mean + df.loc[(slice(None), "std"), :].droplevel(1)

    for env in set(df.columns.get_level_values(0)):
        for alg in set(df.columns.get_level_values(1)):
            print(env, alg)
            idx = (env, alg)
            plt.plot(mean[idx], label=alg.upper())
            plt.fill_between(mean[idx].index, shade_min[idx], shade_max[idx], alpha=0.3)
        plt.legend()
        plt.title(env)
        plt.show()

@click.command()
@click.option("--folder", type=click.Path(exists=True), default="outputs/")
@click.pass_context
def run(ctx, folder):

    hash_to_config, df = _load_data(folder)

    # select "best" configuration (hash)
    best_hash = df.groupby(axis=1, level=[0,1,2]).mean().max().groupby(level=[0,1]).idxmax()
    df = pd.concat([df.xs(h, level=[0,1,2], axis=1, drop_level=False) for h in best_hash.values], axis=1).droplevel(2, axis=1)

    _print_best_config(hash_to_config, best_hash)

    _plot_best_runs(df)

if __name__ == "__main__":
    run()
