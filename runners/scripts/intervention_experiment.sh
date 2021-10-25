#!/bin/sh

#SBATCH --job-name=eval_ir
#SBATCH --output=storage/logs/xai_%A_%a.out
#SBATCH --error=storage/logs/xai_%A_%a.err
#SBATCH --array=0-0
#SBATCH --partition=1080ti-long
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:1

cd /home/jnkenney/xai-interventional-robustness/
python -m runners.src.run_intervention_eval --gpu
