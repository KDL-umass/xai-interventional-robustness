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
    return -np.sum(dist * np.log2(dist))


def js_divergence(dists):
    weight = 1 / len(dists)  # equally weight distributions
    left = shannon(np.sum(weight * dists, axis=0))  # sum along columns
    right = sum([weight * shannon(dist) for dist in dists])
    return left - right


def variance_max_actions(dists):
    return np.std(np.argmax(dists, axis=1))


def plot_js_divergence_matrix(data, vanilla, title, normalize):
    agent = data[:, 0]
    state = data[:, 1]  # 0 indexed
    intv = data[:, 2]  # 0 indexed
    van_state = vanilla[:, 1]  # 0 indexed
    intv_mat = np.zeros((np.max(state).astype(int) + 1, np.max(intv).astype(int) + 1))
    van_mat = np.zeros((np.max(state).astype(int) + 1, 1))
    mat = np.concatenate((van_mat, intv_mat), axis=1)

    for s in np.unique(state).astype(int):
        van_s = vanilla[:, 1] == s
        van_wh = np.where(van_s)[0]
        van_agent = vanilla[van_wh, 3:]
        assert len(vanilla[van_wh, 0]) == len(np.unique(vanilla[van_wh, 0]))
        norm_van = js_divergence(van_agent) / np.log2(10)
        mat[s, 0] = norm_van
        if normalize == True:
            mat[s, 0] = 0.0
        avg = 0

        for i in np.unique(intv).astype(int):
            si = (data[:, 1] == s) * (data[:, 2] == i)
            wh = np.where(si)[0]
            agents = data[wh, 3:]
            assert len(data[wh, 0]) == len(np.unique(data[wh, 0]))
            mat[s, i + 1] = js_divergence(agents) / np.log2(10)
            if normalize == True:
                mat[s, i + 1] -= norm_van
            avg += mat[s, i + 1]

        avg /= len(np.unique(intv).astype(int))

    im = plt.matshow(mat, interpolation="none")
    cbar = plt.colorbar(im)
    cbar.set_label("JS Divergence of Action Distributions")
    if normalize == True:
        plt.clim(-1.0, 1.0)
    else:
        plt.clim(0, 1.0)
    ax = plt.gca()
    ax.tick_params(axis="x", top=False, bottom=True, labelbottom=True, labeltop=False)
    plt.title(title)
    plt.xlabel("Intervention Number")
    plt.ylabel("State of Interest")

    os.makedirs("storage/plots/jsdivmat", exist_ok=True)
    plt.savefig(f"storage/plots/jsdivmat/{title}.png")


# def plot_individual_distributions(agents, mat):
#     im = plt.matshow(agents, interpolation="none", aspect="auto")
#     cbar = plt.colorbar(im)
#     plt.clim(0.0, 1.0)
#     ax = plt.gca()
#     ax.tick_params(axis="x", top=False, bottom=True, labelbottom=True, labeltop=False)
#     plt.title(mat)
#     plt.xlabel("actions")
#     plt.ylabel("runs")
#     os.makedirs(f"storage/plots/jsdivmat/distributions - {title}", exist_ok=True)
#     plt.savefig(f"storage/plots/jsdivmat/distributions - {title}/{i}_{s}.png")
#     plt.close()


def plot_max_action_divergence_matrix(data, title):
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
            mat[s, i] = variance_max_actions(agents)

    norm_mat = mat / np.log2(10)

    im = plt.matshow(norm_mat, interpolation="none")
    cbar = plt.colorbar(im)
    cbar.set_label("Variance of Max Action")
    plt.clim(0.0, 1.0)
    ax = plt.gca()
    ax.tick_params(axis="x", top=False, bottom=True, labelbottom=True, labeltop=False)
    plt.title(title)
    plt.xlabel("Intervention Number")
    plt.ylabel("State of Interest")

    os.makedirs("storage/plots/jsdivmat", exist_ok=True)
    plt.savefig(f"storage/plots/jsdivmat/{title}.png")


if __name__ == "__main__":
    n_agents = 11
    nstates = 30
    horizon = 100
    use_trajectory_starts = True

    for fam in ["a2c", "dqn", "ddqn", "c51", "rainbow"]:
        if use_trajectory_starts:
            dir = f"storage/results/intervention_action_dists/{fam}/{n_agents}_agents/{nstates}_states/trajectory"
        else:
            dir = f"storage/results/intervention_action_dists/{fam}/{n_agents}_agents/{nstates}_states/t{horizon}_horizon"
        plot_js_divergence_matrix(
            np.loadtxt(dir + "/88_interventions.txt"),
            np.loadtxt(dir + "/vanilla.txt"),
            f"Normalized JS Divergence of Actions for {fam}, {n_agents} Agents, t={horizon} horizon",
            normalize=True,
        )
        plot_js_divergence_matrix(
            np.loadtxt(dir + "/88_interventions.txt"),
            np.loadtxt(dir + "/vanilla.txt"),
            f"Unnormalized JS Divergence of Actions for {fam}, {n_agents} Agents, t={horizon} horizon",
            normalize=False,
        )
        plot_max_action_divergence_matrix(
            np.loadtxt(dir + "/88_interventions.txt"),
            f"Max action divergence matrix for {fam}, {n_agents} Agents, t={horizon} horizon",
        )
