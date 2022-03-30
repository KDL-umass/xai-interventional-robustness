#!/bin/bash
#SBATCH --mem=4196  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 04:00:00  # Job time limit
#SBATCH -o storage/logs/slurm-%j.out  # %j = job ID
#SBATCH -e storage/logs/slurm-%j.err 

cd 
cd /gypsum/work1/jensen/pboddavarama/xai-interventional-robustness/
python -m runners.src.run_intervention_perf --env $1 --family $2 --intervention $3 