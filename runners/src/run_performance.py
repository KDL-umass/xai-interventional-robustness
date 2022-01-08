from glob import glob
import argparse
import os
from subprocess import call

# from all.experiments import SlurmExperiment
from all.experiments.run_experiment import get_experiment_type
import numpy as np

from runners.src.run_intervention_eval import model_names

# from all.environments import AtariEnvironment
# from all.presets import atari
# from all.presets.atari import c51, rainbow, a2c, dqn, vac, vpg, vsarsa, vqn, ppo, ddqn
import torch

from envs.wrappers.all_toybox_wrapper import (
    ToyboxEnvironment,
    customAmidarResetWrapper,
    customBreakoutResetWrapper,
    customSpaceInvadersResetWrapper,
)

device = "cuda"
print("CUDA:", torch.cuda.is_available())


checkpoints = list(range(0, 100000, 10000))
checkpoints.extend(list(range(100000, 1000000, 100000)))
checkpoints.extend(list(range(1000000, 11000000, 1000000)))
num_episodes = 3


def main(env_name, fam=None):
    if env_name == "SpaceInvaders":
        custom_wrapper = customSpaceInvadersResetWrapper(0, -1, 3, False)
    elif env_name == "Amidar":
        custom_wrapper = customAmidarResetWrapper(0, -1, 3, False)
    elif env_name == "Breakout":
        custom_wrapper = customBreakoutResetWrapper(0, -1, 3, False)
    else:
        raise ValueError(f"Unrecognized env_name: {env_name}")

    env = ToyboxEnvironment(
        env_name + "Toybox", device=device, custom_wrapper=custom_wrapper
    )

    if fam is None:
        families = model_names
    else:
        families = [fam]

    for fam in families:
        dir = f"storage/results/performance/{env_name}/{fam}"
        os.makedirs(dir, exist_ok=True)

        f = open(dir + "/returns.txt", "w")
        f.write("frame,mean,std\n")

        modelPath = f"storage/models/{env_name}/{fam}"
        for check in checkpoints:
            loadfiles = glob(modelPath + f"/*/preset{check}.pt")
            print(check, loadfiles)

            agents = [torch.load(loadfile) for loadfile in loadfiles]

            results = np.zeros((len(agents), num_episodes))
            for p, preset in enumerate(agents):
                make_experiment = get_experiment_type(preset)
                experiment = make_experiment(
                    preset,
                    env,
                    train_steps=0,
                    logdir="runs",
                    quiet=True,
                    write_loss=False,
                )

                test_returns = experiment.test(episodes=num_episodes, log=False)
                results[p, :] = test_returns

            mean = np.mean(results)
            std = np.std(results)

            f.write(f"{check},{mean},{std}\n")

        f.close()
        call(["rm", "-rf", "runs/*"])


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
    parser.add_argument("--experiment_id", type=int)

    args = parser.parse_args()

    if args.family:
        main(args.env[0], args.family[0])
    else:
        main(args.env[0])

    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
