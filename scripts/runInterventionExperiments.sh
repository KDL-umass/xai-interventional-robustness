#!/bin/bash

conda activate repro

# conduct interventions and evaluate IR
python -m runners.src.run_intervention_eval -g
