#!/bin/bash
#SBATCH --mem=4196  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 04:00:00  # Job time limit
#SBATCH -o storage/logs/slurm-%j.out  # %j = job ID
#SBATCH -e storage/logs/slurm-%j.err 

# conduct interventions and evaluate OR
python -m runners.src.run_intervention_eval -g
