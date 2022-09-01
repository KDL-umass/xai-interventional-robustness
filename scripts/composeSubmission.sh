#!/bin/bash

mkdir -p sandbox/SUBMISSION_PKG/submission

cp -r analysis sandbox/SUBMISSION_PKG/submission
rm -r sandbox/SUBMISSION_PKG/submission/analysis/notebooks

cp -r envs sandbox/SUBMISSION_PKG/submission
cp -r models sandbox/SUBMISSION_PKG/submission

cp -r runners sandbox/SUBMISSION_PKG/submission
rm -r sandbox/SUBMISSION_PKG/submission/runners/notebooks

cp -r scripts sandbox/SUBMISSION_PKG/submission

mkdir -p storage/models
mkdir -p storage/plots
mkdir -p storage/results
mkdir -p storage/states

cp __init__.py sandbox/SUBMISSION_PKG/submission
cp README.md sandbox/SUBMISSION_PKG/submission
cp requirements.txt sandbox/SUBMISSION_PKG/submission
cp setup.py sandbox/SUBMISSION_PKG/submission
cp .gitignore sandbox/SUBMISSION_PKG/submission

rm -r sandbox/SUBMISSION_PKG/submission/**/__pycache__/
rm -r sandbox/SUBMISSION_PKG/submission/**/*/__pycache__/
rm -r sandbox/SUBMISSION_PKG/submission/**/.DS_Store

rm sandbox/SUBMISSION_PKG/submission/scripts/composeSubmission.sh

# package
cd sandbox/SUBMISSION_PKG
rm supplementary_materials.zip
zip -r supplementary_materials.zip *
