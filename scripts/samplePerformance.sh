#!/bin/bash

conda activate repro

for env in Amidar Breakout SpaceInvaders
do 
    for fam in a2c c51 dqn ddqn ppo rainbow vqn vsarsa
    do
        python -m runners.src.run_performance --env $env --family $family;    
    done
done
