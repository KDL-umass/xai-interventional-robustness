import os

import matplotlib

font = {"size": 14}

matplotlib.rc("font", **font)

import matplotlib.pyplot as plt
import numpy as np


def plot_js_divergence_matrix(data, vanilla, title, normalize, fname=None):
    state = data[:, 1]  # 0 indexed
    intv = data[:, 2]  # 0 indexed
    samples = data[:, 3]

    nstates = np.max(state).astype(int) + 1
    assert nstates == len(vanilla)

    nintv = np.max(intv).astype(int) + 1
    print(f"nstates {nstates} nintv {nintv}")

    intv_mat = np.zeros((nstates, nintv))
    for s in range(nstates):
        for i in range(nintv):
            intv_mat[s, i] = samples[s * nintv + i] / np.log2(10)

    van_mat = vanilla[:, 3] / np.log2(10)
    if normalize:
        intv_mat /= van_mat.reshape(-1, 1)

    mat = np.zeros((nstates, nintv + 1))
    mat[:, 0] = van_mat
    mat[:, 1:] = intv_mat

    im = plt.matshow(mat, interpolation="none")
    cbar = plt.colorbar(im)
    cbar.set_label("JS Divergence of Action Distributions")
    if normalize:
        plt.clim(-1.0, 1.0)
    else:
        plt.clim(0, 1.0)
    ax = plt.gca()
    ax.tick_params(axis="x", top=False, bottom=True, labelbottom=True, labeltop=False)
    plt.title(title)
    plt.xlabel("Intervention Number")
    plt.ylabel("State of Interest")

    os.makedirs("storage/plots/sampled_jsdivmat", exist_ok=True)
    if fname is not None:
        plt.savefig(f"storage/plots/sampled_jsdivmat/{fname}.png", bbox_inches="tight")
    else:
        plt.savefig(f"storage/plots/sampled_jsdivmat/{title}.png", bbox_inches="tight")

    return van_mat.mean(), intv_mat.mean()


def print_image_name_table(families):
    print()
    print("\\begin{tabular}{ccc}")
    print("Family & Unnormalized & Normalized \\\\")
    for fam in families:
        print(
            fam
            + " & \\includegraphics[width=0.4\\textwidth]{plots/jsdiv_"
            + fam
            + ".png} & \includegraphics[width=0.4\\textwidth]{plots/jsdiv_"
            + fam
            + "_normalized.png}\\\\"
        )
    print("\\end{tabular}")


def print_values_table(
    families, checkpoints, vanilla_dict, unnormalized_dict, normalized_dict
):
    F = len(families)
    C = len(checkpoints)
    table = np.zeros((F, C, 3))

    print()
    print("\\begin{tabular}{|l|l|c|c|c|}\\hline")
    print("Family & Checkpoint & Original & Unnormalized & Normalized \\\\\\hline")

    for f, fam in enumerate(families):
        for c, check in enumerate(checkpoints):
            v = vanilla_dict[fam][check]
            u = unnormalized_dict[fam][check]
            n = normalized_dict[fam][check]
            table[f, c, :] = v, u, n

            print(f"{fam} & {check} & {v} & {u} & {n} \\\\\\hline")
    print("\\end{tabular}")

    return table


if __name__ == "__main__":
    n_agents = 11
    nstates = 30
    folder = "intervention_js_div"

    families = ["a2c", "dqn", "ddqn", "c51", "rainbow", "vsarsa", "vqn", "ppo"]
    # checkpoints = [str(100 * 10 ** i) for i in range(6)]
    checkpoints = [""]

    vanilla_dict = {}
    unnormalized_dict = {}
    normalized_dict = {}

    for fam in families:

        vanilla_dict[fam] = {}
        unnormalized_dict[fam] = {}
        normalized_dict[fam] = {}

        for check in checkpoints:
            dir = f"storage/results/{folder}/{fam}/{n_agents}_agents/{nstates}_states/trajectory"
            vdata = np.loadtxt(dir + f"/vanilla{check}.txt")
            data = np.loadtxt(dir + f"/88_interventions{check}.txt")
            _, normalized_dict[fam][check] = plot_js_divergence_matrix(
                data,
                vdata,
                f"Normalized Sampled JS Divergence over Actions for {fam} at {check}",
                normalize=True,
                fname=f"jsdiv_{fam}{check}_normalized",
            )
            (
                vanilla_dict[fam][check],
                unnormalized_dict[fam][check],
            ) = plot_js_divergence_matrix(
                data,
                vdata,
                f"Unnormalized Sampled JS Divergence over Actions for {fam} at {check}",
                normalize=False,
                fname=f"jsdiv_{fam}{check}",
            )

    print_image_name_table(families)
    print_values_table(
        families, checkpoints, vanilla_dict, unnormalized_dict, normalized_dict
    )
