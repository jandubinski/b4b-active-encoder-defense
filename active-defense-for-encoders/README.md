# active-ssl

## Setup
* Create environment with python 3.9
* Install requirements from requirements.txt
* Install and [setup Wandb](https://docs.wandb.ai/quickstart) (pipreqs does not add it to requirements)
```bash
pip install wandb
```
* Make sure you have access to project given by `<wandb_entity>/<wandb_project>` from [config](config)
* Set [src](src) directory as PYTHONPATH
```bash
export PYTHONPATH=${PYTHONPATH}:./src
```

## Running experiments sybil
1. Before you run experiments you should use scripts in [src/data/embeddings](src/data/embeddings)
to save train and test embeddings from DINO in .pt format 
(paths to those files are input parameters for sybil_defense.py script)

2. Main script to run experiment is [src/sybil_defense.py](src/sybil_defense.py).

We use hydra to manage experiments. Experiment parameters are stored in [config](config) directory. 

* To run training with default config, run the command below. 
Experiment parameters will be loaded from [config/config.yaml](config/config.yaml).
```shell
python src/sybil_defense.py
```
* To override experiment parameters, use dot notation.
```shell
python src/sybil_defense.py \
  seed=0 \
  benchmark.hparams.train_epochs_reference_classifier=10 \
  benchmark.hparams.lr_mapper = 0.0001
```
* To use load partial configs for benchmarks etc. use the notation from below.
Here *benchmark* is the name of the [directory](config/benchmark) in [config](config) and 
*mnist_embeddings* is the name of file in this directory (without .yaml extension).
```shell
python src/sybil_defense.py \
  benchmark=benchmark.mnist_embeddings
```

## Running sweeps (TODO)

[comment]: <> (Use wandb with sweeps from `sweeps` directory to run hyperparameter search:)

[comment]: <> (```shell)

[comment]: <> (wandb sweep sweeps/experiment.yaml)

[comment]: <> (```)

```