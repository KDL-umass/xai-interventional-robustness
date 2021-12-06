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
from glob import glob

env_name = "SpaceInvaders"
device = "cuda"
frames = 12e6 + 1
render = False
logdir = "runs"
writer = "tensorboard"
toybox = True
agent_replicate_num = 12
test_episodes = 150
nodelist = ""
loadfile = False  # replace with specific path if continuing from checkpoint
# e.g. loadfile = "/mnt/nfs/scratch1/kavery/si_ppo_snapshots"


if env_name == "SpaceInvaders":
    custom_wrapper = customSpaceInvadersResetWrapper(0, -1, 3, False)
elif env_name == "Amidar":
    custom_wrapper = customAmidarResetWrapper(0, -1, 3, False)
elif env_name == "Breakout":
    custom_wrapper = customBreakoutResetWrapper(0, -1, 3, False)


def main():
    if toybox:
        env = ToyboxEnvironment(
            env_name + "Toybox", device=device, custom_wrapper=custom_wrapper
        )
    else:
        env = AtariEnvironment(env_name, device=device)

    agents = [
        # a2c.device(device),
        # dqn.device(device),
        vsarsa.device(device),
        # vqn.device(device),
        # dqn.device(device),
        # ppo.device(device),
        # vsarsa.device(device),
        # vqn.device(device),
        # ppo.device(device),
        # rainbow.device(device),
        # c51.device(device),
        # ddqn.device(device),
    ]

    agents = list(np.repeat(agents, agent_replicate_num))

    if loadfile:
        loadfiles = glob(loadfile + "/*/")
        if len(loadfiles) == 1:
            loadfiles = [loadfile]
    else:
        loadfiles = [""]

    for load in loadfiles:
        if device == "cuda":
            if load != "":
                print(load + "preset10000000.pt")
            SlurmExperiment(
                agents,
                env,
                frames,
                test_episodes=test_episodes,
                logdir=logdir,
                write_loss=True,
                loadfile="" if load == "" else load + "preset10000000.pt",
                sbatch_args={"partition": "1080ti-long"},
                nodelist=nodelist,
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
