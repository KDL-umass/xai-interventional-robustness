from all.experiments import SlurmExperiment, run_experiment
from all.environments import AtariEnvironment
from all.presets import atari
import argparse
from all.presets.atari import c51, rainbow, a2c, dqn, vac, vpg, vsarsa, vqn, ppo, ddqn

from envs.wrappers.all_toybox_wrapper import (
    ToyboxEnvironment,
    customAmidarResetWrapper,
    customBreakoutResetWrapper,
    customSpaceInvadersResetWrapper,
)
import numpy as np

env_name = "Amidar"
device = "cuda"
frames = 1e7 + 1
render = False
logdir = "runs"
writer = "tensorboard"
toybox = True
agent_replicate_num = 12
test_episodes = 100


def main():
    if toybox:
        env = ToyboxEnvironment(
            env_name + "Toybox",
            device=device,
            custom_wrapper=customAmidarResetWrapper(
                state_num=0, intv=-1, lives=3, use_trajectory_starts=False
            ),
        )
    else:
        env = AtariEnvironment(env_name, device=device)

    agents = [
        # a2c.device(device),
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
            # loadfile=""
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
