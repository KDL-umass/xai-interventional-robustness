# XAI Interventional Robustness

Experiments for XAI Interventional Robustness paper, Summer 2021.

## Setup

To make everything reproducible, use Python 3.7.4 and install packages according to the requirements:
Use a virtualenv or conda environment to encapsulate these installations.

Key packages are `torch==1.8.1`, `ctoybox==0.4.2`,
`autonomous-learning-library` (local version), and `toyxbox`.

```bash
pip install -r requirements.txt
pip install -e .
```

Clone the autonomous learning library from KDL's repository:
```
cd ..
git clone git@github.com:KDL-umass/autonomous-learning-library.git
cd autonomous-learning-library
pip install -e .
```

## Running

Everything is written to be run with via python with the `-m` flag from the root directory.

e.g.

```bash
python -m runners.src.evaluate_performance
```

Typically results will be found in `storage/results`.

## Training Agents

To see available commands, run
```bash
python -m runners.src.run_experiment --help
```

Example: To train a single a2c agent on SpaceInvaders, run the following command.

```bash
python -m runners.src.run_experiment --env SpaceInvaders --family a2c
```

When this is done, make sure that the `out/*.err` file created for that Slurm job did not error with CUDA error.
Error: `RuntimeError: CUDA driver initialization failed, you might not have a CUDA gpu.`
Repeat as necessary until 11 agents are successfuly training. 

Don't worry if there are `rm: missing operand. Try 'rm --help' for more information.` errors in the `*.err` files, this is expected and not a problem.

Sometimes there are CUDA errors unless you run this command from an interactive node:
```bash
srun --pty bash
```

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
