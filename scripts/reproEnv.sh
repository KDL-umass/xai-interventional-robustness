#!/bin/bash

# srun --pty bash
# create conda env
conda create -n repro python=3.7.4
conda activate repro
# install dependencies
cd submission-code/
pip install --upgrade pip
conda install -c anaconda swig #==4.0.2
pip install --no-cache-dir torch==1.8.1
pip install -r requirements.txt
pip install -e .
cd ..
cd autonomous-learning-library/
pip install -e .
cd ../submission-code/
# intialize states
python -m envs.wrappers.space_invaders.interventions.interventions
python -m envs.wrappers.breakout.interventions.interventions
python -m envs.wrappers.amidar.interventions.interventions
