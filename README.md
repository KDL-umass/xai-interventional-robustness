# XAI Interventional Robustness

Experiments for XAI Interventional Robustness paper, Summer 2021.

## Setup

To make everything reproducible and compatible with gypsum's CentOS, run the following shell script to create the venv and install the appropriate python packages.

```bash
sh setup.sh
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
