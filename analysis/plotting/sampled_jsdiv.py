import os

import matplotlib

font = {"size": 14}

matplotlib.rc("font", **font)

import matplotlib.pyplot as plt
import numpy as np


def plot_js_divergence_matrix(data, vanilla, title, normalize):
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
    plt.savefig(f"storage/plots/sampled_jsdivmat/{title}.png")


if __name__ == "__main__":
    n_agents = 11
    nstates = 30
    folder = "intervention_js_div"

    for fam in ["a2c", "dqn", "ddqn", "c51", "rainbow", "vsarsa"]:
        dir = f"storage/results/{folder}/{fam}/{n_agents}_agents/{nstates}_states/trajectory"
        vdata = np.loadtxt(dir + "/vanilla.txt")
        data = np.loadtxt(dir + "/88_interventions.txt")
        plot_js_divergence_matrix(
            data,
            vdata,
            f"Normalized Sampled JS Divergence over Actions for {fam}, {n_agents} Agents",
            normalize=True,
        )
        plot_js_divergence_matrix(
            data,
            vdata,
            f"Unnormalized Sampled JS Divergence over Actions for {fam}, {n_agents} Agents",
            normalize=False,
        )
