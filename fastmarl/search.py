from copy import deepcopy
from itertools import product
import multiprocessing
import random
import subprocess

import click
import yaml

_CPU_COUNT = multiprocessing.cpu_count() - 1


def _flatten_lists(object):
    for item in object:
        if isinstance(item, (list, tuple, set)):
            yield from _flatten_lists(item)
        else:
            yield item


def _seed_and_shuffle(configs, shuffle, seeds):

    configs = [[f"+hypergroup=hp_grp_{i}"] + c for i, c in enumerate(configs)]
    configs = list(product(configs, [f"seed={i}" for i in range(seeds)]))
    configs = [list(_flatten_lists(c)) for c in configs]

    if shuffle:
        random.Random(1337).shuffle(configs)

    return configs


def _load_config(filename):
    config = yaml.load(filename)
    return config

def _gen_combos(config, built_config):
    built_config = deepcopy(built_config)
    if not config:
        return [[f"{k}={v}" for k,v in built_config.items()]]
    
    k, v = list(config.items())[0]

    configs = []
    if type(v) is list:
        for item in v:
            new_config = deepcopy(config)
            del new_config[k]
            new_config[k] = item
            configs += _gen_combos(new_config, built_config)
    elif type(v) is tuple:
        new_config = deepcopy(config)
        del new_config[k]
        for item in v:
            new_config.update(item)
        
        configs += _gen_combos(new_config, built_config)
    else:
        new_config = deepcopy(config)
        del new_config[k]
        built_config[k] = v
        configs += _gen_combos(new_config, built_config)
    return configs

def work(cmd):
    cmd = cmd.split(" ")
    return subprocess.call(cmd, shell=False)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("output", type=click.Path(exists=False, dir_okay=False, writable=True))
def write(output):
    raise NotImplemented


@cli.group()
@click.option("--config", type=click.File(), default="config.yaml")
@click.option("--shuffle/--no-shuffle", default=True)
@click.option("--seeds", default=3, show_default=True, help="How many seeds to run")
@click.pass_context
def run(ctx, config, shuffle, seeds):
    config = _load_config(config)
    configs = _gen_combos(config, {})
    
    configs = _seed_and_shuffle(configs, shuffle, seeds)
    if len(configs) == 0:
        click.echo("No valid combinations. Aborted!")
        exit(1)
    ctx.obj = configs


@run.command()
@click.option(
    "--cpus",
    default=_CPU_COUNT,
    show_default=True,
    help="How many processes to run in parallel",
)
@click.pass_obj
def locally(combos, cpus):
    configs = ["python run.py " + "-m " + " ".join([c for c in combo]) for combo in combos]

    click.confirm(
        f"There are {click.style(str(len(combos)), fg='red')} combinations of configurations. Up to {cpus} will run in parallel. Continue?",
        abort=True,
    )


    pool = multiprocessing.Pool(processes=cpus)
    print(pool.map(work, configs))

@run.command()
@click.pass_obj
def dry_run(combos):
    configs = [" ".join([c for c in combo]) for combo in combos]
    click.echo(f"There are {click.style(str(len(combos)), fg='red')} configurations as shown below:")
    for c in configs:
        click.echo(c)


@run.command()
@click.argument(
    "index", type=int,
)
@click.pass_obj
def single(combos, index):
    """Runs a single hyperparameter combination
    INDEX is the index of the combination to run in the generated combination list
    """

    config = combos[index]
    cmd = "python run.py " + " ".join([c for c in config])
    print(cmd)
    work(cmd)


if __name__ == "__main__":
    cli()