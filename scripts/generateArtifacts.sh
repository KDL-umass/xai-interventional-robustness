#!/bin/bash

# generate primary plots show in paper
for env in Amidar Breakout SpaceInvaders
do
    python -m analysis.plotting.sampled_jsdiv $env &
    python -m analysis.plotting.sampled_jsdiv $env norm &
done

# generate individual plots
python -m analysis.plotting.sampled_jsdiv &

# generate performance plots
python -m analysis.plotting.performance

# generate tables
python -m analysis.plotting.tables
