from glob import glob
import argparse
import os
from subprocess import call

from all.experiments.run_experiment import get_experiment_type
import numpy as np

from analysis.checkpoints import all_checkpoints, checkpoints as paper_checkpoints
import torch

from envs.wrappers.all_toybox_wrapper import (
    ToyboxEnvironment,
    customAmidarResetWrapper,
    customBreakoutResetWrapper,
    customSpaceInvadersResetWrapper,
)

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


num_episodes = 5


def main(env_name, fam, intervention=-1, checkpoint=None):
    if env_name == "SpaceInvaders":
        custom_wrapper = customSpaceInvadersResetWrapper(0, intervention, 3)
    elif env_name == "Amidar":
        custom_wrapper = customAmidarResetWrapper(0, intervention, 3)
    elif env_name == "Breakout":
        custom_wrapper = customBreakoutResetWrapper(0, intervention, 3)
    else:
        raise ValueError(f"Unrecognized env_name: {env_name}")

    env = ToyboxEnvironment(
        env_name + "Toybox", device=device, custom_wrapper=custom_wrapper
    )

    dir = f"storage/results/intervention_performance/{env_name}/{fam}"
    os.makedirs(dir, exist_ok=True)

    # If returns file does not exist, then write the first line, else open in a+ mode.
    # Parallelize wrt intervention
    returns_file = dir + "/returns_intv_" + str(intervention) + ".txt"
    if os.path.exists(returns_file):
        # Check if all the experimental results are complete, then return() (very hacky way)
        if len(open(returns_file, "r").readlines()) == 56:
            print("Experiment is already done!")
            return ()
        # If not, continue
        else:
            f = open(returns_file, "a+")
    else:
        f = open(returns_file, "w")
        f.write("env,family,agent,intervention,frame,mean,std\n")
    print("Created returns file")

    if checkpoint is not None:
        assert checkpoint in all_checkpoints, "checkpoint not available"
        checkpoints = [checkpoint]
    else:
        checkpoints = paper_checkpoints

    modelPath = f"storage/models/{env_name}/{fam}"

    for check in checkpoints:
        loadfiles = glob(modelPath + f"/*/preset{check}.pt")

        agents = [
            torch.load(loadfile, map_location=torch.device(device))
            for loadfile in loadfiles
        ]
        print("Agents: ", agents)

        results = np.zeros((len(agents), num_episodes))
        for p, preset in enumerate(agents):
            make_experiment = get_experiment_type(preset)
            experiment = make_experiment(
                preset,
                env,
                train_steps=0,
                logdir="runs",
                quiet=False,
                write_loss=False,
            )

            # test_returns = experiment.test(episodes=num_episodes, log=False)
            test_returns = experiment.test(episodes=num_episodes)
            experiment.close()
            results[p, :] = test_returns

            # Write result for each agent
            print(results[p, :])

            mean = np.mean(results[p, :])
            std = np.std(results[p, :])

            f.write(f"{env_name},{fam},{p+1},{intervention},{check},{mean},{std}\n")
            f.flush()
            print(f"{env_name}, {fam}, {check} written")

    f.close()
    # call(["rm", "-rf", "runs/*"]) # Since we are parallelizing might delete other experimental results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train agent of specified type on environment."
    )
    parser.add_argument(
        "--env",
        nargs=1,
        type=str,
        help="environment name: SpaceInvaders, Amidar, or Breakout",
    )
    parser.add_argument(
        "--family",
        nargs=1,
        type=str,
        help="Agent family:  a2c,c51, dqn, ddqn, ppo, rainbow, vsarsa, vqn",
    )

    parser.add_argument(
        "--intervention",
        nargs=1,
        type=int,
        help="intervention number for each environment and family",
    )

    # Not using checkpoints, automatically run all experiments for all the checkpoints mentioned in the main() function
    parser.add_argument(
        "--checkpoint",
        nargs=1,
        type=int,
        help="Checkpoint to load",
    )

    parser.add_argument("--experiment_id", type=int)

    args = parser.parse_args()

    if args.intervention is not None:
        main(args.env[0], args.family[0], args.intervention[0])
    else:
        main(args.env[0], args.family[0])

    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
