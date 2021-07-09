import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 250
from scipy import stats


def plot_dist(intervention_name, vanilla_name, agent_name):
    binwidth = 50

    plt.figure(figsize=(14, 5), facecolor="white")
    plt.rc("font", size=13)

    interv = np.loadtxt("results/" + intervention_name + ".txt")
    vanilla = np.loadtxt("results/" + vanilla_name + ".txt")

    if len(interv.shape) == 1:
        interv = interv.reshape(1, -1)

    maxscore = int(max(np.max(interv), np.max(vanilla))) + 50

    plt.subplot(1, 2, 1)
    plt.hist(interv.ravel(), bins=range(0, maxscore, binwidth))
    plt.title(
        f"{agent_name} Performance Under Interventions\nInterventions: {interv.shape[0]}, Trials (seeds): {interv.shape[1]}"
    )
    plt.xlim(0, maxscore)
    plt.xlabel("Return")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(vanilla.ravel(), bins=range(0, maxscore, binwidth))
    plt.title(
        f"{agent_name} Performance Vanilla\nVanilla trials (seeds): {np.size(vanilla)}"
    )
    plt.xlim(0, maxscore)
    plt.xlabel("Return")
    plt.ylabel("Frequency")

    print(stats.ttest_ind(vanilla.ravel(), interv.ravel()))


def plot_vanilla_comparison(lives):
    binwidth = 50

    plt.figure(figsize=(14, 10), facecolor="white")
    plt.rc("font", size=13)
    plt.subplot(2, 2, 1)

    random = np.loadtxt(f"results/random_performance_vanilla_lives_{lives}.txt")
    ddt = np.loadtxt(f"results/ddt_performance_vanilla_lives_{lives}.txt")
    cnn = np.loadtxt(f"results/cnn_performance_vanilla_lives_{lives}.txt")

    maxscore = (
        int(max(np.amax(random.ravel()), np.amax(ddt.ravel()), np.amax(cnn.ravel())))
        + 50
    )

    plt.hist(random.ravel(), bins=range(0, maxscore, binwidth))
    plt.title(f"Random Performance Vanilla\nVanilla trials (seeds): {np.size(random)}")
    plt.xlim(0, maxscore)
    plt.xlabel("Return")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 2)
    plt.hist(ddt.ravel(), bins=range(0, maxscore, binwidth))
    plt.title(f"DDT Performance Vanilla\nVanilla trials (seeds): {np.size(ddt)}")
    plt.xlim(0, maxscore)
    plt.xlabel("Return")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 4)
    plt.hist(cnn.ravel(), bins=range(0, maxscore, binwidth))
    plt.title(f"CNN Performance Vanilla\nVanilla trials (seeds): {np.size(cnn)}")
    plt.xlim(0, maxscore)
    plt.xlabel("Return")
    plt.ylabel("Frequency")

    plt.tight_layout()
