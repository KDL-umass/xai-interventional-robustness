import os


from envs.wrappers.paths import get_num_interventions

font = {"size": 15}

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

cmap = rcParams["image.cmap"]
cmap = plt.get_cmap(cmap).reversed()
matplotlib.rc("font", **font)

import numpy as np
from analysis.src.ce import norm_sym_cross_entropy


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


def variance_max_actions(dists):
    return np.std(np.argmax(dists, axis=1))


def plot_ce_matrix(data, vanilla, title, normalize):
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
        norm_van = norm_sym_cross_entropy(van_agent) #/ np.log2(10)
        mat[s, 0] = norm_van
        if normalize == True:
            mat[s, 0] = 0.0
        avg = 0

        for i in np.unique(intv).astype(int):
            si = (data[:, 1] == s) * (data[:, 2] == i)
            wh = np.where(si)[0]
            agents = data[wh, 3:]
            assert len(data[wh, 0]) == len(np.unique(data[wh, 0]))
            mat[s, i + 1] = norm_sym_cross_entropy(agents) #/ np.log2(10)
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

    os.makedirs("storage/plots/sampled_cemat", exist_ok=True)
    plt.savefig(f"storage/plots/sampled_cemat/{title}.png")


# def plot_individual_distributions(agents, mat):
#     im = plt.matshow(agents, interpolation="none", aspect="auto")
#     cbar = plt.colorbar(im)
#     plt.clim(0.0, 1.0)
#     ax = plt.gca()
#     ax.tick_params(axis="x", top=False, bottom=True, labelbottom=True, labeltop=False)
#     plt.title(mat)
#     plt.xlabel("actions")
#     plt.ylabel("runs")
#     os.makedirs(f"storage/plots/cemat/distributions - {title}", exist_ok=True)
#     plt.savefig(f"storage/plots/cemat/distributions - {title}/{i}_{s}.png")
#     plt.close()


def plot_max_action_ce_matrix(data, title):
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

    os.makedirs("storage/plots/sampled_cemat", exist_ok=True)
    plt.savefig(f"storage/plots/sampled_cemat/{title}.png")


if __name__ == "__main__":
    n_agents = 11
    nstates = 10
    cesampling = True
    folder = "intervention_ce" if cesampling else "intervention_action_dists"

    environment = "SpaceInvaders"
    nintv = get_num_interventions(environment)

    for fam in ["a2c", "dqn", "ddqn", "c51", "rainbow"]:
        dir = f"storage/results/{folder}/{fam}/{n_agents}_agents/{nstates}_states/trajectory"

        vdata = np.loadtxt(dir + "/vanilla.txt")
        data = np.loadtxt(dir + f"/{nintv}_interventions.txt")
        plot_ce_matrix(
            data,
            vdata,
            f"Normalized JS Divergence over Actions for {fam}, {n_agents} Agents",
            normalize=True,
        )
        plot_ce_matrix(
            data,
            vdata,
            f"JS Divergence over Actions for {fam}, {n_agents} Agents",
            normalize=False,
        )
        plot_max_action_ce_matrix(
            data,
            f"Max action ce matrix for {fam}, {n_agents} Agents, ({nstates} states)",
        )
