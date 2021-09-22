from all.experiments import SlurmExperiment, run_experiment
from all.environments import AtariEnvironment
from all.presets import atari
import argparse
from all.presets.atari import c51, rainbow, a2c, dqn, vac, vpg, vsarsa, vqn, ppo, ddqn

from envs.wrappers.space_invaders.all_toybox_wrapper import ToyboxEnvironment
import numpy as np

env_name = "SpaceInvaders"
device = "cuda"
frames = 10e10
render = False
logdir = "runs"
writer = "tensorboard"
toybox = True
agent_replicate_num = 12
test_episodes = 100

def main():
    if toybox:
        env = ToyboxEnvironment("SpaceInvadersToybox", device=device)
    else:
        env = AtariEnvironment(env_name, device=device)

    agents = [
        #a2c.device(device),
        vqn.device(device),
        # vac.device(device),
        # vpg.device(device),
        # vsarsa.device(device),
        # vqn.device(device)
    ]

    agents = list(np.repeat(agents, agent_replicate_num))

    if device == "cuda":
        SlurmExperiment(
            agents,
            env,
            frames,
            test_episodes=test_episodes,
            logdir=logdir,
            write_loss=True,
            sbatch_args={"partition": "1080ti-long"},
        )
    else:
        run_experiment(
            agents,
            env,
            frames,
            render=render,
            logdir=logdir,
            writer=writer,
            test_episodes=test_episodes,
        )


if __name__ == "__main__":
    main()