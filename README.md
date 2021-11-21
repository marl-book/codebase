<h1><b>Fast</b> iteration of <b>MARL</b> research ideas: A starting point for Multi-Agent Reinforcement Learning</h1>

Algorithm implementations with emphasis on ***FAST*** iteration of ***MARL*** research ideas.
The algorithms are self-contained and the implementations are focusing on simplicity and speed.

<h1>Table of Contents</h1>

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Running an algorithm](#running-an-algorithm)
    - [(Optional) Use Hydra's tab completion](#optional-use-hydras-tab-completion)
  - [Running a hyperparameter search](#running-a-hyperparameter-search)
    - [An advanced hyperparameter search using `search.py`](#an-advanced-hyperparameter-search-using-searchpy)
  - [Logging](#logging)
    - [File System Logger:](#file-system-logger)
    - [WandB Logger:](#wandb-logger)
- [Implementing your own algorithm/ideas](#implementing-your-own-algorithmideas)
- [Interpreting your results](#interpreting-your-results)
- [Implemented Algorithms](#implemented-algorithms)
  - [Parameter Sharing](#parameter-sharing)
  - [Value Decomposition](#value-decomposition)
- [Contact](#contact)



# Getting Started

## Installation

We *strongly* suggest you use a virtual environment for the instructions below. A good starting point is [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Then, clone and install the repository using: 

```sh
git clone https://github.com/semitable/blazing-ma.git
cd blazing-ma
pip install -r requirements.txt
pip install -e .
```

## Running an algorithm
This project uses [Hydra](https://hydra.cc/) to structure its configuration. Algorithm implementations can be found under `blazingma/`. The respective configs are found in `blazingma/configs/algorithms/`.

You would first need an environment that is registered in OpenAI's Gym. This repository uses the Gym API (with the only difference being that the rewards are a tuple - one for each agent). 

A good starting point would be [Level-based Foraging](https://github.com/semitable/lb-foraging) and [RWARE](https://github.com/semitable/robotic-warehouse). You can install both using:
```sh
pip install -U lbforaging rware
```

Then, running an algorithm (e.g. A2C) looks like:

```sh
cd blazingma
python run.py +algorithm=ac env.name="lbforaging:Foraging-8x8-2p-3f-v2" env.time_limit=25
```

Similarly, running DQN can be done using:
```sh
python run.py +algorithm=dqn env.name="lbforaging:Foraging-8x8-2p-3f-v2" env.time_limit=25
```

Overriding hyperparameters is easy and can be done in the command line. An example of overriding the `batch_size` in DQN:
```sh
python run.py +algorithm=dqn env.name="lbforaging:Foraging-8x8-2p-3f-v2" env.time_limit=25 algorithm.batch_size=256
```

Find other hyperparameters in the files under `blazingma/configs/algorithm`.

### (Optional) Use Hydra's tab completion
Hydra also supports tab completion for filling in the hyperparameters. Install it using or see [here](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion) for other shells (`zsh` or `fish`).
```sh
eval "$(python run.py -sc install=bash)"
```
## Running a hyperparameter search

Can be easily done using [Hydra's multirun](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run) option. An example of sweeping over batch sizes is:

```sh
python run.py -m +algorithm=dqn env.name="lbforaging:Foraging-8x8-2p-3f-v2" env.time_limit=25 algorithm.batch_size=64,128,256
```

### An advanced hyperparameter search using `search.py`
*This section might get deprecated in the future if Hydra implements this feature.*

We include a script named `search.py` which reads a search configuration file (e.g. the included `configs/sweeps/dqn.lbf.yaml`) and runs a hyperparameter search in one or more tasks. The script can be run using
```sh
python search.py run --config configs/sweeps/dqn.lbf.yaml --seeds 5 locally
```
In a cluster environment where one run should go to a single process, it can also be called in a batch script like:
```sh
python search.py run --config configs/sweeps/dqn.lbf.yaml --seeds 5 single $TASK_ID
```
Where `$TASK_ID` is an index for the experiment (i.e. 1...#number of experiments).

## Logging
We implement two loggers: FileSystem Logger and WandB Logger.

### File System Logger:
The default logger is the FileSystemLogger which saves experiment results in a `results.csv` file. You can find that file, the configuration that has been used & more under `outputs/{env_name}/{alg_name}/{random_hash}` or `multirun/{date}/{time}/{experiment_id}` for multiruns.
### WandB Logger:
By appending `+logger=wandb` in the command line you can get support for WandB. Do not forget to `wandb login` first.

Example:

```sh
python run.py +algorithm=dqn env.name="lbforaging:Foraging-8x8-2p-3f-v2" env.time_limit=25 logger=wandb
```
You can override the project name using:

```sh
python run.py +algorithm=dqn env.name="lbforaging:Foraging-8x8-2p-3f-v2" env.time_limit=25 logger=wandb logger.project_name="my-project-name"
```

# Implementing your own algorithm/ideas

The fastest way would be to create a new folder starting from the algorithm of your choice e.g.
```sh
cp -R ac ac_new_idea
```
and create a new configuration file:
```sh
cp configs/algorithm/ac.yaml configs/algorithm/ac_new_idea.yaml
```

with the editor of your choice, open `ac_new_idea.yaml` and change
```yaml
...
algorithm:
  _target_: ac.train.main
  name: "ac"
  model:
    _target_: ac.model.Policy
...
```
to 
```yaml
...
algorithm:
  _target_: ac_new_idea.train.main
  name: "ac_new_idea"
  model:
    _target_: ac_new_idea.model.Policy
...
```
Make any changes you want to the files under `ac_new_idea/` and run it using:

```sh
python run.py +algorithm=ac_new_idea env.name="lbforaging:Foraging-8x8-2p-3f-v2" env.time_limit=25
```
You can now add new hyperparameters, change the training procedure, or anything else you want and keep the old implementations for easy comparison. We hope that the way we have implemented these algorithms makes it easy to change any part of the algorithm without the hustle of reading through large code-bases and huge unnecessary layers of abstraction. RL research benefits from iterating over ideas quickly to see how they perform!

# Interpreting your results

# Implemented Algorithms

|                             | A2C                | DQN (Double Q)     |
|-----------------------------|--------------------|--------------------|
| Parameter Sharing           | :heavy_check_mark: | :heavy_check_mark: |
| Selective Parameter Sharing | :heavy_check_mark: | :heavy_check_mark: |
| Centralized Critic          | :heavy_check_mark: | :x:                |
| Value Decomposition         | :x:                | :heavy_check_mark: |
| Return Standarization       | :heavy_check_mark: | :heavy_check_mark: |
| Target Networks             | :heavy_check_mark: | :heavy_check_mark: |


## Parameter Sharing
## Value Decomposition

# Contact
Filippos Christianos - f.christianos {at} ed {dot} ac {dot} uk

Project Link: https://github.com/semitable/blazing-ma

