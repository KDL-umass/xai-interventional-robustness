#!/bin/bash

for env in Amidar 
do 
    for fam in a2c c51 dqn ddqn ppo rainbow vqn vsarsa
    do
        for intv in {0..69}
        do
            echo $env $fam $intv; 
            sbatch scripts/unityInterventionPerf.sh $env $fam $intv;
        done
    done
done


for env in SpaceInvaders 
do 
    for fam in a2c c51 dqn ddqn ppo rainbow vqn vsarsa
    do
        for intv in {0..87}
        do
            echo $env $fam $intv; 
            sbatch scripts/unityInterventionPerf.sh $env $fam $intv;   
        done
    done
done

for env in Breakout 
do 
    for fam in a2c c51 dqn ddqn ppo rainbow vqn vsarsa
    do
        for intv in {0..38}
        do
            echo $env $fam $intv; 
            sbatch scripts/unityInterventionPerf.sh $env $fam $intv;
        done
    done
done
