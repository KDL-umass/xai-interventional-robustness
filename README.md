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
