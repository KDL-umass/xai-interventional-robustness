#!/bin/bash

for env in Amidar Breakout SpaceInvaders
do
    for fam in a2c
    do
        for n in {0} # create 11 agents
        do
            python -m runners.src.run_experiment --env $env --family $fam;
        done
    done
done
