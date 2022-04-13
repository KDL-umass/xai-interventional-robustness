#!/bin/bash

for env in Amidar 
do 
    for fam in a2c c51 dqn ddqn ppo rainbow vqn vsarsa
    do
        for intv in {0..68}
        do
            echo $env $fam $intv; 
            sbatch scripts/unityInterventionPerf.sh $env $fam $intv;
        done
    done
done

# Check whether the file has completed or not. 
# for env in Amidar 
# do 
#     for fam in a2c c51 dqn ddqn ppo rainbow vqn vsarsa
#     do
#         for intv in {0..68}
#         do
#             python -m runners.src.check_intervention_perf --env $env --family $fam --intervention $intv; 
#         done
#     done
# done


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
        for intv in {0..37}
        do
            echo $env $fam $intv; 
            sbatch scripts/unityInterventionPerf.sh $env $fam $intv;
        done
    done
done