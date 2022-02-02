import os
import sys

import matplotlib
from matplotlib import cm, rcParams
from matplotlib.colors import Normalize
from analysis.plotting.performance import plotAllFamilies

from analysis.src.js_divergence import get_js_divergence_matrix
from envs.wrappers.paths import get_num_interventions

from runners.src.performance_plot import *

import matplotlib.pyplot as plt
import numpy as np

from runners.src.run_intervention_eval import (
    supported_environments,
    model_names,
)

from analysis.checkpoints import checkpoints

from analysis.plotting.sampled_jsdiv import *

cmap = rcParams["image.cmap"]
cmap = plt.get_cmap(cmap).reversed()


def plotPerformance(fig, env, label=True):
    if env == "Breakout":
        gs = fig.add_gridspec(
            1, len(model_names), left=0.1, right=0.9, top=1.20, bottom=1.1
        )
    elif env == "Amidar":
        gs = fig.add_gridspec(
            1, len(model_names), left=0.125, right=0.9, top=1.20, bottom=1.1
        )
    else:
        gs = fig.add_gridspec(
            1, len(model_names), left=0.1, right=0.9, top=1.15, bottom=1.05
        )
    plotAllFamilies(env, gs, label=label)


def resizeMegaplot(env, normalized):
    # without showPoints
    topshift = {"Breakout": 0.95, "Amidar": 0.95, "SpaceInvaders": 0.95}[env]
    bottomshift = {"Breakout": 0.07, "Amidar": 0.07, "SpaceInvaders": 0.07}[env]
    rightshift = {"Breakout": 0.9, "Amidar": 0.9, "SpaceInvaders": 0.9}[env]
    leftshift = {
        "Breakout": 0.1,
        "Amidar": 0.13,
        "SpaceInvaders": 0.1 if normalized else 0.1,
    }[env]
    hspace = {"Breakout": 0.2, "Amidar": 0.1, "SpaceInvaders": 0.1}[env]
    wspace = {"Breakout": 0.2, "Amidar": 0.1, "SpaceInvaders": 0.1}[env]

    plt.subplots_adjust(
        top=topshift,
        bottom=bottomshift,
        left=leftshift,
        right=rightshift,
        hspace=hspace,
        wspace=wspace,
    )


def colorBar(fig, normalized, env):
    cbar_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    if env == "Breakout":
        cbar_ax.set_title(
            r"IR ($\mathcal{R}$)",
            {"fontsize": 8, "fontweight": "bold"},
            # y=-2.2,
        )
    else:
        cbar_ax.set_title(
            r"IR ($\mathcal{R}$)",
            {"fontsize": 8, "fontweight": "bold"},
            # y=-1.8,
        )
    vmin = -1 if normalized else 0
    plt.colorbar(
        cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=1), cmap=cmap),
        cax=cbar_ax,
        orientation="vertical",
    )
    cbar_ax.tick_params(labelsize=6)


def horizontalMegaPlot(normalized, nAgents=11, nStates=30, env=None):
    """
    https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subfigures.html#sphx-glr-gallery-subplots-axes-and-figures-subfigures-py
    """
    font = {"size": 10}
    matplotlib.rc("font", **font)

    if env is None:
        env = sys.argv[1]

    with open(f"storage/plots/returns/{env}/order.txt") as f:
        model_names = [l.strip() for l in f.readlines()]

    figsize = {"Breakout": (8, 3.5), "Amidar": (9, 3.5), "SpaceInvaders": (10, 3.5)}[
        env
    ]

    fig, axes = plt.subplots(
        len(checkpoints),
        len(model_names),
        sharex=True,
        sharey=True,
        # figsize=(500 * px, 1000 * px),
        figsize=figsize,
    )

    nintv = get_num_interventions(env)

    for f, fam in enumerate(model_names):
        # add performance at rightmost side
        data = load_returns_100_data(f"storage/models/{env}/{fam}")[env + "Toybox"]
        performance = get_checkpoint_performances(
            f"storage/models/{env}", env, fam, checkpoints
        )

        performanceOrder = 1000

        # add jsdiv plots
        for c, check in enumerate(checkpoints):
            ax = axes[c, f]
            if c == 0:
                ax.xaxis.set_label_position("top")
                ax.set_xlabel(
                    xlabel=f"{fam.upper()}",
                    fontdict={
                        "fontsize": 10,
                        "fontweight": "bold",
                        # "rotation_mode": "anchor",
                        # "position": (-1000, 0.5),
                        # "verticalalignment": "top",
                        # "horizontalalignment": "center",
                    },
                    labelpad=10.0,
                    loc="center",
                    # rotation=-90,
                )
            # if c == len(checkpoints) - 1:
            if f == len(model_names) - 1:
                order = int(np.floor(np.log10(check)))
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(
                    f"{check // 10**order}e{order}",
                    {"fontsize": 10},
                    rotation=0,
                    loc="center",
                    labelpad=14,
                )
                # ax.tick_params(labelbottom=True)

            dir = f"storage/results/intervention_js_div/{env}/{fam}/{nAgents}_agents/{nStates}_states/trajectory/check_{check}"

            vdata = np.loadtxt(dir + f"/vanilla.txt")
            data = np.loadtxt(dir + f"/{nintv}_interventions.txt")

            subplot_js_divergence_matrix(
                ax,
                data,
                vdata,
                normalized,
                title=format(performance[c] / performanceOrder, ".2f"),
                # + f"e{performanceOrder}",
                transpose=False,
            )
            print(f"plot, {f},{c}")

    resizeMegaplot(env, normalized)
    plt.figure(fig.number)
    colorBar(fig, normalized, env)
    plotPerformance(fig, env, label=False)

    title_type = "Normalized " if normalized else ""
    file_type = "normalized" if normalized else "unnormalized"

    # fig.suptitle(f"{title_type}Interventional Robustness: {env}", fontsize=11)

    plt.margins(0)
    # annotations
    framesXY = (0.895, 1)
    plt.annotate(
        "Frames",
        xy=framesXY,
        xytext=framesXY,
        textcoords="figure fraction",
        fontsize=10,
        rotation=-90,
        fontweight="bold",
    )

    framesXY = (0.03, 0.46)
    plt.annotate(
        "State",
        xy=framesXY,
        xytext=framesXY,
        textcoords="figure fraction",
        rotation=90,
        size=10,
        fontweight="bold",
    )

    framesXY = (0.05, 1.05)
    plt.annotate(
        "Score",
        xy=framesXY,
        xytext=framesXY,
        textcoords="figure fraction",
        rotation=90,
        size=10,
        fontweight="bold",
    )

    framesXY = (0.45, 0.01 if env == "Breakout" else 0.01)
    plt.annotate(
        "Intervention",
        xy=framesXY,
        xytext=framesXY,
        textcoords="figure fraction",
        size=10,
        fontweight="bold",
    )

    # save
    plt.savefig(
        f"storage/plots/sampled_jsdivmat/{env}_{file_type}_horizontal.png",
        bbox_inches="tight",
        dpi=600,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(f"Making megaplot for {sys.argv[1]}")
        if len(sys.argv) > 2:
            horizontalMegaPlot(True)
        else:
            horizontalMegaPlot(False)
    else:
        print("error")
