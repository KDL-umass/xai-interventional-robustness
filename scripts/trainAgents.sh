#!/bin/bash

conda activate repro

for env in Amidar Breakout SpaceInvaders
do
    for fam in a2c c51 dqn ddqn ppo rainbow vqn vsarsa
    do
        for n in {0..10} # create 11 agents
        do
            python -m runners.src.run_experiment --env $env --family $fam;
        done
    done
done
