# XAI Interventional Robustness

Experiments for XAI Interventional Robustness paper, Summer 2021.

## Setup

To make everything reproducible, use Python 3.7.4 and install packages according to the requirements:

```bash
pip install -r requirements.txt
pip install -e .
```

## Running

Everything is written to be run with via python with the `-m` flag from the root directory.

e.g.

```bash
python -m runners.src.evaluate_performance
```

Typically results will be found in `storage/results`.

## Formatting

Please use the `black` formatting so we don't have big diffs due to people using different autoformatters.

## Gypsum

Running jobs GPU compatible _interactive_ nodes, use
```bash
srun --pty --gres=gpu:1 bash
```
and execute whatever code you want to.

Running batch jobs, use
```bash
sbatch runners/scripts/intervention_experiment.sh
```
or for training, run
```bash
python -m runners.src.run_experiment
```

Removing `events*` files that take up a lot of storage.

```bash
rm $(find -name events*)
```