#!/bin/bash

conda activate repro

# conduct interventions and evaluate OR
python -m runners.src.run_intervention_eval -g
