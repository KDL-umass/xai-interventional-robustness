#!/bin/sh

#SBATCH --job-name=xai_interventional_robustness
#SBATCH --output=out/xai_%A_%a.out
#SBATCH --error=out/xai_%A_%a.err
#SBATCH --array=0-0
#SBATCH --partition=1080ti-long
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:1

cd /home/ppruthi/xai-interventional-robustness/
python /home/ppruthi/xai-interventional-robustness/runners/src/run_intervention_eval.py --gpu
