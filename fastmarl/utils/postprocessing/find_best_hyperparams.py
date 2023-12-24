import click
import pandas as pd


def _print_best_config(configs, best_hash):
    best_hash = best_hash.values

    for (env, alg, hash) in best_hash:
        click.echo(
            "For "
            + click.style(env, fg="red", bold=True)
            + " the best "
            + click.style(alg.upper(), fg="red", bold=True)
            + " config is: "
        )
        print(85 * "=")
        print(configs.loc[hash])
        print(85 * "-" + "\n")


@click.command()
@click.option("--exported-file", type=click.Path(dir_okay=False, writable=False))
@click.pass_context
def run(ctx, exported_file):

    df = pd.read_hdf(exported_file, "df")
    configs = pd.read_hdf(exported_file, "configs")

    best_hash = (
        df.groupby(axis=1, level=[0, 1, 2]).mean().max().groupby(level=[0, 1]).idxmax()
    )

    _print_best_config(configs, best_hash)


if __name__ == "__main__":
    run()
