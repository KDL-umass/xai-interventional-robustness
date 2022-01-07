import numpy as np
import matplotlib.pyplot as plt
import argparse


def getDir(env, fam, check):
    return f"storage/results/intervention_js_div/{env}/{fam}/11_agents/30_states/trajectory/check_{check}"


def create_histograms(env, fam, check, state, intv):
    dir = getDir(env, fam, check)
    data = np.loadtxt(dir + f"/actions.csv", delimiter=",")

    srows = (data[:, 0] == state).astype(bool)
    irows = (data[:, 1] == intv).astype(bool)
    rows = srows * irows

    d = data[rows, 2:]
    print(d)
    actions = np.mean(d, axis=0)
    print(actions)

    if env == "Amidar":
        my_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif env == "Breakout":
        my_bins = [0, 1, 2, 3]
    elif env == "SpaceInvaders":
        my_bins = [0, 1, 2, 3, 4, 5]

    h, _ = np.histogram(actions, bins=my_bins)
    print(h)
    plt.bar(range(len(my_bins) - 1), h, width=1, edgecolor="k")
    plt.xlabel("Action")
    plt.ylabel("Frequency of Selection Among Trained Agents")
    # plt.axis([-0.5, 5.5, 0, 10])
    plt.title(
        f"Actions selected by {fam} for {env}\nin state {state} under intv {intv}"
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train agent of specified type on environment."
    )
    parser.add_argument(
        "-e",
        "--env",
        nargs=1,
        type=str,
        help="Environment name: SpaceInvaders, Amidar, or Breakout",
    )
    parser.add_argument(
        "-f",
        "--family",
        nargs=1,
        type=str,
        help="Agent family:  a2c,c51, dqn, ddqn, ppo, rainbow, vsarsa, vqn",
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        nargs=1,
        type=str,
        help="Checkpoint [50000,100000,...,10000000]",
    )

    parser.add_argument(
        "-s",
        "--state",
        nargs=1,
        type=int,
        help="State:  [0,...,30]",
    )

    parser.add_argument(
        "-i",
        "--intv",
        nargs=1,
        type=int,
        help="Intervention:  [0,...,M]",
    )

    args = parser.parse_args()

    create_histograms(
        args.env[0], args.family[0], args.checkpoint[0], args.state[0], args.intv[0]
    )
