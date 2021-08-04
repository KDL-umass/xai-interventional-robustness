import os

import matplotlib

font = {"size": 15}

matplotlib.rc("font", **font)

import matplotlib.pyplot as plt
import numpy as np


def plot_action_dist(data, title, xmin, xmax):
    fig = plt.figure(figsize=(20, 4), facecolor="white")
    plt.suptitle(title)
    for i in range(6):
        plt.subplot(1, 6, i + 1)
        plt.hist(data[:, i])
        plt.xlim(xmin, xmax)
        plt.xlabel(f"Action {i} Prob")
        if i == 0:
            plt.ylabel("Action Frequency")


def shannon(dist):
    dist = dist + 1e-10  # eps to prevent div 0
    return -np.sum(dist * np.log(dist))


def js_divergence(dists):
    weight = 1 / len(dists)  # equally weight distributions
    left = shannon(np.sum(weight * dists, axis=1))  # sum along columns
    right = sum([shannon(dist) for dist in dists])
    return left - right


def plot_js_divergence_matrix(data, title):
    agent = data[:, 0]
    state = data[:, 1]  # 0 indexed
    intv = data[:, 2]  # 0 indexed

    mat = np.zeros((np.max(state).astype(int) + 1, np.max(intv).astype(int) + 1))

    for s in np.unique(state).astype(int):
        for i in np.unique(intv).astype(int):
            si = (data[:, 1] == s) * (data[:, 2] == i)
            wh = np.where(si)[0]
            agents = data[wh, 3:]
            assert len(data[wh, 0]) == len(np.unique(data[wh, 0]))
            mat[s, i] = js_divergence(agents)

    im = plt.matshow(mat, interpolation="none")
    cbar = plt.colorbar(im)
    cbar.set_label("JS Divergence of Action Distributions")
    ax = plt.gca()
    ax.tick_params(axis="x", top=False, bottom=True, labelbottom=True, labeltop=False)
    plt.title(title)
    plt.xlabel("Intervention Number")
    plt.ylabel("State of Interest")

    os.makedirs("storage/plots/jsdivmat", exist_ok=True)
    plt.savefig(f"storage/plots/jsdivmat/{title}.png")


if __name__ == "__main__":
    for fam in ["a2c", "dqn"]:
        dir = f"storage/results/intervention_action_dists/{fam}/11_agents/30_states/t100_horizon"
        plot_js_divergence_matrix(
            np.loadtxt(dir + "/88_interventions.txt"),
            f"JS Divergence of Actions for {fam}, 11 Agents, t=100 horizon",
        )