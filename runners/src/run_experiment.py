from all.experiments import SlurmExperiment, run_experiment
from all.presets.atari import (
    a2c,
    dqn,
    vac,
    vpg,
    vsarsa,
    vqn
)

from envs.wrappers.space_invaders_wrapper.all_toybox_wrapper import ToyboxEnvironment

def main():
    device = 'cpu' #'cuda'
    env = ToyboxEnvironment('SpaceInvadersToybox', device=device)
    agents = [
        a2c.device(device),
        # dqn.device(device),
    ]
    # SlurmExperiment(agents, env, 10e6, sbatch_args={
    #     'partition': '1080ti-long'
    # })
    run_experiment(agents, env, 10e6)

if __name__ == "__main__":
    #test_env_loading()
    main()
