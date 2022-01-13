#!/bin/bash

mkdir -p sandbox/IJCAI22_SUBMISSION/submission

cp -r analysis sandbox/IJCAI22_SUBMISSION/submission
rm -r sandbox/IJCAI22_SUBMISSION/submission/analysis/notebooks

cp -r envs sandbox/IJCAI22_SUBMISSION/submission
cp -r models sandbox/IJCAI22_SUBMISSION/submission

cp -r runners sandbox/IJCAI22_SUBMISSION/submission
rm -r sandbox/IJCAI22_SUBMISSION/submission/runners/notebooks

cp -r scripts sandbox/IJCAI22_SUBMISSION/submission

mkdir -p storage/models
mkdir -p storage/plots
mkdir -p storage/results
mkdir -p storage/states

cp __init__.py sandbox/IJCAI22_SUBMISSION/submission
cp README.md sandbox/IJCAI22_SUBMISSION/submission
cp requirements.txt sandbox/IJCAI22_SUBMISSION/submission
cp setup.py sandbox/IJCAI22_SUBMISSION/submission
cp .gitignore sandbox/IJCAI22_SUBMISSION/submission

rm -r sandbox/IJCAI22_SUBMISSION/submission/**/__pycache__/
rm -r sandbox/IJCAI22_SUBMISSION/submission/**/*/__pycache__/
rm -r sandbox/IJCAI22_SUBMISSION/submission/**/.DS_Store

rm sandbox/IJCAI22_SUBMISSION/submission/scripts/composeSubmission.sh

# package
cd sandbox/IJCAI22_SUBMISSION
zip -r submission.zip *
