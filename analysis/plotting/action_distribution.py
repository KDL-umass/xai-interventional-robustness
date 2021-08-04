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
    res = 0
    for i in range(len(dist)):
        res += dist[i] * np.log(dist[i])
    return -res


def js_divergence(dists):
    weight = 1 / len(dists)  # equally weight distributions
    left = shannon(np.sum(weight * dists, axis=1))  # sum along columns
    right = sum([shannon(dist) for dist in dists])
    return left - right


def plot_js_divergence_matrix(data, title):
    agent = data[0, :]
    state = data[1, :]  # 0 indexed
    intv = data[2, :]  # 0 indexed

    mat = np.zeros((np.max(state).astype(int) + 1, np.max(intv).astype(int) + 1))

    for s in np.unique(state):
        for i in np.unique(intv):
            sel = data[data[1, :] == s * data[2, :] == i, :]
            print(sel)


if __name__ == "__main__":
    dir = f"storage/results/intervention_action_dists/a2c/11_agents/30_states/t100_horizon"
    plot_js_divergence_matrix(np.loadtxt(dir + "/88_interventions.txt"), "")
