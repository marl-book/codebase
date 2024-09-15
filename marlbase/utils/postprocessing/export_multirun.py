from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import click
import json
from munch import munchify
import pandas as pd
import yaml


def _load_data(folder):
    path = Path(folder)
    config_files = path.glob("**/**/.hydra/config.yaml")

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seed_data = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    algos = set()
    envs = set()

    hash_to_config = defaultdict()

    for config_path in config_files:
        # hash = config_path.parent.parent.name

        with config_path.open() as fp:
            config = munchify(yaml.safe_load(fp))

        env = config.env.name
        try:
            env = env.split(":")[1]  # remove the library
        except IndexError:
            pass

        algo = config.algorithm.name

        algos.add(algo)
        envs.add(env)

        seed = config.seed
        del config.seed

        raw_data = json.dumps(config, sort_keys=True).encode("utf8")
        hash = sha256(raw_data).hexdigest()[:12]

        hash_to_config[hash] = pd.json_normalize(config)

        df = pd.read_csv(config_path.parent.parent / "results.csv", index_col=0)[
            "mean_episode_returns"
        ]
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

    return pd.concat(hash_to_config).droplevel(1), df


@click.command()
@click.option("--folder", type=click.Path(exists=True), default="outputs/")
@click.option("--export-file", type=click.Path(dir_okay=False, writable=True))
@click.pass_context
def run(ctx, folder, export_file):

    hash_to_config, df = _load_data(folder)

    df.to_hdf(export_file, key="df", mode="w", complevel=9)
    hash_to_config.to_hdf(export_file, key="configs")


if __name__ == "__main__":
    run()
