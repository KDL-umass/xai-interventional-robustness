# Source Code: Measuring the Interventional Robustness of Reinforcement Learning Agents

Experiments IJCAI '22 submission

## Summary

Running everything for this repo sequentially involves these bash scripts

First make them all executable so they run as bash and not alternative shells
```bash
chmod +x scripts/*
```

Training requires conda and a cluster with SLURM, but can be modified if SLURM is not available

```bash
./scripts/reproEnv.sh
./scripts/trainAgents.sh
```

Then agents must be organized from the `runs/` directory into the `storage/models/{env}/{family}` folders accordingly. Then interventions and performance estimates can be run:

```bash
./scripts/runInterventionExperiments.sh
./scripts/samplePerformance.sh
```

And finally plots and tables found in the paper and supplementary materials can be generated:

```bash
./scripts/generateArtifacts.sh
```

This will produce plots in `storage/plots/sampled_jsdivmat/` and `storage/plots/performance/` for the environments and agents used.
It will also print the tables found in the Appendix of the interventional robustness measure values.

## Prerequisites

Our reproduction relies on an installation of `Anaconda` or `Miniconda` to create virtualenvs and install appropriate packages in an isolated fashion.

Training the agents and conducting the intervention experiments requires a CUDA capable GPU with `pytorch` compatibility.

## Setup

### Script

We provide a reproduction script to reproduce the environment as follows:

```
./scripts/reproEnv.sh
```

### Steps in `reproEnv.sh`

All experiments were conducted on linux systems (CentOS and Ubuntu) using CUDA capable GPUs. Using one of these systems is recommended for reproducibility.

To make everything reproducible, use Python 3.7.4 and install packages according to the requirements:
Use a virtualenv or conda environment to encapsulate these installations.

Key packages are `torch==1.8.1`, `ctoybox==0.4.2`,
`autonomous-learning-library` (local version), and `toyxbox`.

```bash
pip install -r requirements.txt
pip install -e .
```

Install the local ALL

```
cd ..
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

Once setup is complete, you can train the agents using:

```bash
./scripts/trainAgents.sh
```

This presumes that you are running this on a SLURM system, and attempts to create SLURM batch jobs to parallelize compute.

### Environment Setup

To set up the environments, we have to initialize the start states as follows:

```bash
python -m envs.wrappers.space_invaders.interventions.interventions
python -m envs.wrappers.amidar.interventions.interventions
python -m envs.wrappers.breakout.interventions.interventions
```

Now we can begin training.

### Training

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

Don't worry if there are `rm: missing operand. Try 'rm --help' for more information.` errors in the `*.err` files, this is expected and not a problem. It's clearing out unused data so we don't run out of storage.

Sometimes there are CUDA errors unless you run this command from a slurm interactive node (if using slurm):

```bash
srun --pty bash
```

## Slurm GPU compute cluster instructions (if applicable)

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

Removing `events*` files produced by ALL that take up a lot of storage:

```bash
rm $(find -name events*)
```

this is found in the `scripts/removeEvents.sh` script.
