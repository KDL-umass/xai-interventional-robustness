import os
import numpy as np
import matplotlib.pyplot as plt

from runners.src.run_intervention_eval import model_names, supported_environments
from analysis.checkpoints import all_checkpoints as checkpoints

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
colorMap = {fam: colors[i] for i, fam in enumerate(model_names)}


def getPerformanceFile(env, fam):
    return f"storage/results/performance/{env}/{fam}/returns.txt"


def loadPerformance(env, fam):
    fname = getPerformanceFile(env, fam)
    return np.loadtxt(fname, delimiter=",", skiprows=1)


def makeOrderFile(env):
    performances = []
    for fam in model_names:
        data = loadPerformance(env, fam)
        performances.append((np.max(data[:, 1]), fam))  # max mean value
    order = list(sorted(performances).__reversed__())
    order = list(map(lambda x: x[1], order))
    print(env, "order", order)

    os.makedirs(f"storage/plots/performance/{env}", exist_ok=True)
    with open(f"storage/plots/performance/{env}/order.txt", "w") as f:
        f.writelines([o + "\n" for o in order])


def getYAxisLims(env):
    min = float("inf")
    max = float("-inf")
    for fam in model_names:
        data = loadPerformance(env, fam)
        if min > np.min(data[:, 1]):
            min = np.min(data[:, 1])
        if max < np.max(data[:, 1]):
            max = np.max(data[:, 1])
    return round(min), round(max)


def plotData(env, fam, ax):
    data = loadPerformance(env, fam)

    t = data[:, 0]
    mean = data[:, 1]
    std = data[:, 2]

    (line,) = ax.plot(t, mean, label=None, color=colorMap[fam])
    ax.fill_between(t, mean + std, mean - std, alpha=0.2, color=line.get_color())


def plotFamily(env, fam, ax=plt.gca()):
    plotData(env, fam, ax)
    ax.set_yticks(getYAxisLims(env))
    ax.set_xticks(checkpoints)
    xlabs = ["" for c in checkpoints]
    xlabs[0] = "0"
    order = np.log10(checkpoints[-1])
    xlabs[-1] = f"{int(checkpoints[-1] / 10**order)}e{int(order)}"
    ax.set_xticklabels(xlabs)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_ylim(getYAxisLims(env))
    ax.label_outer()

    ax.yaxis.set_tick_params(size=4, labelsize=7, labelrotation=-90)
    ax.xaxis.set_tick_params(size=4, labelsize=7)


def plotEachFamily(env):
    with open(f"storage/plots/performance/{env}/order.txt") as f:
        model_names = [l.strip() for l in f.readlines()]
    for fam in model_names:
        plt.figure(1, figsize=(4, 3)).clear()
        ax = plt.gca()
        plotData(env, fam, ax)
        plt.xlabel("Frames")
        plt.ylabel("Points")
        plt.title(f"Performance of {fam} on {env}")
        plt.savefig(f"storage/plots/performance/{env}/{fam}.png", dpi=600)

    plt.figure(figsize=(5, 3))
    ax = plt.gca()
    for fam in model_names:
        plotData(env, fam, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Points")
    plt.title(f"Performance on {env}")
    plt.legend(model_names, bbox_to_anchor=(1, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"storage/plots/performance/{env}_performance.png", dpi=600)


def plotAllFamilies(
    env, gridspec=plt.figure(1).add_gridspec(len(model_names), 1), show=False
):
    with open(f"storage/plots/performance/{env}/order.txt") as f:
        model_names = [l.strip() for l in f.readlines()]
    axes = gridspec.subplots(sharex=True, sharey=True)

    for i, fam in enumerate(model_names):
        plotFamily(env, fam, axes[i])

    # axes[0].set_title("Performance", fontsize=10)
    axes[len(model_names) - 1].set_xlabel("Frames", fontsize=10, labelpad=5)

    if show:
        plt.show()


if __name__ == "__main__":
    for env in supported_environments:
        os.makedirs(f"storage/plots/performance/{env}/", exist_ok=True)
        # plotAllFamilies(env, show=True)
        makeOrderFile(env)
        plotEachFamily(env)
    pass
