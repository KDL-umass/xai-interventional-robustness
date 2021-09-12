import numpy as np
from all.experiments import SlurmExperiment, run_experiment
from all.environments import AtariEnvironment
from all.presets.atari import a2c, dqn, vac, vpg, vsarsa, vqn, ppo

from envs.wrappers.space_invaders.all_toybox_wrapper import ToyboxEnvironment

render = False
logdir = "runs"
writer = "tensorboard"
agent_replicate_num = 2


def run_xai_experiment(agents, envs, device, frames, test_episodes):
    if device == "cuda":
        SlurmExperiment(
            agents,
            envs,
            frames,
            test_episodes=test_episodes,
            logdir=logdir,
            write_loss=True,
            sbatch_args={"partition": "1080ti-long"},
        )
    else:
        run_experiment(
            agents,
            envs,
            frames,
            render=render,
            logdir=logdir,
            writer=writer,
            test_episodes=test_episodes,
        )


def main():  # agent training
    device = "cpu"
    toybox = True
    test_episodes = 2
    frames = 1e2

    if toybox:
        env = ToyboxEnvironment("SpaceInvadersToybox", device=device)

    else:
        env = AtariEnvironment("SpaceInvaders", device=device)

    agents = [
        a2c.device(device),
        # dqn.device(device),
        # vac.device(device),
        # vpg.device(device),
        # vsarsa.device(device),
        # vqn.device(device)
    ]

    agents = list(np.repeat(agents, agent_replicate_num))

    run_xai_experiment(agents, env, device, frames, test_episodes)


if __name__ == "__main__":
    main()
