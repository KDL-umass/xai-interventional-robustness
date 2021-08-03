from envs.wrappers.space_invaders.interventions.start_states import sample_start_states
from envs.wrappers.space_invaders.interventions.interventions import (
    create_intervention_states,
)
import os
from envs.wrappers.space_invaders.interventions.reset_wrapper import (
    SpaceInvadersResetWrapper,
)
from all.experiments import SlurmExperiment, run_experiment
from all.environments import AtariEnvironment
from all.presets import atari
import argparse
from all.presets.atari import a2c, dqn, vac, vpg, vsarsa, vqn, ppo

from envs.wrappers.space_invaders.all_toybox_wrapper import (
    ToyboxEnvironment,
    customSpaceInvadersResetWrapper,
)
import numpy as np

env_name = "SpaceInvaders"
device = "cpu"
frames = 1e2
render = False
logdir = "runs"
writer = "tensorboard"
agent_replicate_num = 2
test_episodes = 2
toybox = True
interventions = True


if interventions:
    num_states_to_intervene_on = 2  # only default starting state
    start_horizon = 100  # sample from t=100
    sample_start_states(num_states_to_intervene_on, start_horizon)
    num_interventions = create_intervention_states(num_states_to_intervene_on)


def main():
    if toybox:
        if interventions:
            env = [
                ToyboxEnvironment(
                    "SpaceInvadersToybox",
                    device=device,
                    custom_wrapper=customSpaceInvadersResetWrapper(
                        state_num=state_num, intv=intv, lives=3
                    ),
                )
                for state_num in range(num_states_to_intervene_on)
                for intv in range(num_interventions)
            ]
        else:
            env = ToyboxEnvironment("SpaceInvadersToybox", device=device)

    else:
        env = AtariEnvironment(env_name, device=device)

    agents = [
        a2c.device(device),
        dqn.device(device),
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
