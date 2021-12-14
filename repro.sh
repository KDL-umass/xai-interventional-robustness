srun --pty bash
conda create -n repro python=3.7.4
conda activate repro
cd xai-interventional-robustness/
pip install requirements.txt
# pip install -e git+git://github.com/toybox-rs/Toybox.git@7b81f0802d9826b78f0af2e7f90289b579f3103f#egg=toybox
pip install --upgrade pip
pip install -e .
cd ..
cd autonomous-learning-library/
pip install -e .
cd ../xai-interventional-robustness/
python -m envs.wrappers.space_invaders.interventions.interventions
python -m envs.wrappers.breakout.interventions.interventions
python -m envs.wrappers.amidar.interventions.interventions
