from glob import glob
import argparse
import os
from subprocess import call

from all.experiments.run_experiment import get_experiment_type
import numpy as np

from analysis.checkpoints import all_checkpoints
import torch

from envs.wrappers.all_toybox_wrapper import (
    ToyboxEnvironment,
    customAmidarResetWrapper,
    customBreakoutResetWrapper,
    customSpaceInvadersResetWrapper,
)

device = "cuda"
print("CUDA:", torch.cuda.is_available())


num_episodes = 3


def main(env_name, fam, checkpoint=None):
    checkpoints = all_checkpoints

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

    dir = f"storage/results/performance/{env_name}/{fam}"
    os.makedirs(dir, exist_ok=True)

    if checkpoint is not None:
        assert checkpoint in checkpoints, "checkpoint not available"
        checkpoints = [checkpoint]
        f = open(dir + "/returns.txt", "a+")
    else:
        f = open(dir + "/returns.txt", "w")
        f.write("frame,mean,std\n")

    modelPath = f"storage/models/{env_name}/{fam}"

    for check in checkpoints:
        loadfiles = glob(modelPath + f"/*/preset{check}.pt")

        agents = [torch.load(loadfile) for loadfile in loadfiles]

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

            test_returns = experiment.test(episodes=num_episodes, log=False)
            experiment.close()
            results[p, :] = test_returns

        mean = np.mean(results)
        std = np.std(results)

        f.write(f"{check},{mean},{std}\n")
        f.flush()
        print(f"{env_name}, {fam}, {check} written")

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

    parser.add_argument(
        "--checkpoint",
        nargs=1,
        type=int,
        help="Checkpoint to load",
    )
    parser.add_argument("--experiment_id", type=int)

    args = parser.parse_args()

    if args.checkpoint is not None:
        main(args.env[0], args.family[0], args.checkpoint[0])
    else:
        main(args.env[0], args.family[0])

    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
