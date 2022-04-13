#!/bin/bash
#SBATCH --mem=4196  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 04:00:00  # Job time limit
#SBATCH -o storage/logs/Breakout_ivpf/slurm-%j.out  # %j = job ID, change location according to the experiment
#SBATCH -e storage/logs/Breakout_ivpf/slurm-%j.err 

cd 
cd /gypsum/work1/jensen/pboddavarama/xai-interventional-robustness/
python -m runners.src.run_intervention_perf --env $1 --family $2 --intervention $3 