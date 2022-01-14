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


cmap = rcParams["image.cmap"]
cmap = plt.get_cmap(cmap).reversed()


def subplot_js_divergence_matrix(ax, data, vanilla, normalize, title=""):
    font = {"size": 10}
    matplotlib.rc("font", **font)

    mat, nmat, van_mat, intv_mat, n_intv_mat = get_js_divergence_matrix(data, vanilla)

    # invert!
    mat = 1 - mat

    if normalize:
        mat = nmat
        mat = (2 - (mat + 1)) - 1

    plt.sca(ax)
    im = ax.imshow(mat.T, interpolation="none", cmap=cmap)

    ax.spines["bottom"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.spines["left"].set_color("white")

    ax.spines["bottom"].set_linewidth(0)
    ax.spines["top"].set_linewidth(0)
    ax.spines["right"].set_linewidth(0)
    ax.spines["left"].set_linewidth(0)

    if normalize:
        im.set_clim(-1.0, 1.0)
    else:
        im.set_clim(0, 1.0)

    ax.set_xticks(list(range(0, mat.shape[0] + 1, 15)))
    ax.set_xticklabels(list(range(0, mat.shape[0] + 1, 15)), fontsize=7)

    ax.set_yticks(list(range(0, mat.shape[1], 15)))
    ax.set_yticklabels(list(range(0, mat.shape[1], 15)), fontsize=7)

    ax.tick_params(left=False, bottom=False)


def plot_js_divergence_matrix(
    data, vanilla, title, normalize, env, family="", checkpoint="", fname=None
):
    mat, nmat, van_mat, intv_mat, n_intv_mat = get_js_divergence_matrix(data, vanilla)
    if normalize:
        mat = nmat

    # invert!
    mat = 1 - mat

    im = plt.matshow(mat.T, interpolation="none", aspect="auto", cmap=cmap)

    if normalize:
        im.set_clim(-1.0, 1.0)
    else:
        im.set_clim(0, 1.0)

    ax = plt.gca()
    plt.xlabel("State")
    plt.ylabel("Intervention")
    ax.tick_params(axis="x", top=True, bottom=False, labelbottom=False, labeltop=True)

    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label(r"Interventional Robustness ($\mathcal{R}$)")
    if normalize:
        plt.clim(-1.0, 1.0)
    else:
        plt.clim(0, 1.0)

    os.makedirs(f"storage/plots/sampled_jsdivmat/{env}", exist_ok=True)
    if fname is not None:
        plt.savefig(
            f"storage/plots/sampled_jsdivmat/{env}/{fname}.png", bbox_inches="tight"
        )
    else:
        plt.savefig(
            f"storage/plots/sampled_jsdivmat/{env}/{title}.png", bbox_inches="tight"
        )
    plt.close(plt.gcf())

    return van_mat.mean(), intv_mat.mean()


def individualPlots(normalized):
    font = {"size": 10}
    matplotlib.rc("font", **font)

    n_agents = 11
    nstates = 30

    for env in supported_environments:
        with open(f"storage/plots/returns/{env}/order.txt") as f:
            model_names = [l.strip() for l in f.readlines()]

        nintv = get_num_interventions(env)

        for f, fam in enumerate(model_names):

            for c, check in enumerate(checkpoints):
                if check == "":
                    dir = f"storage/results/intervention_js_div/{env}/{fam}/{n_agents}_agents/{nstates}_states/trajectory"
                else:
                    dir = f"storage/results/intervention_js_div/{env}/{fam}/{n_agents}_agents/{nstates}_states/trajectory/check_{check}"

                vdata = np.loadtxt(dir + f"/vanilla.txt")
                data = np.loadtxt(dir + f"/{nintv}_interventions.txt")

                title_type = "Normalized " if normalized else ""
                file_type = "normalized" if normalized else "unnormalized"
                name = f"{title_type}Interventional Robustness\nfor {fam} at {check} frames, {env}"
                plot_js_divergence_matrix(
                    data,
                    vdata,
                    name,
                    normalized,
                    env,
                    family=fam,
                    checkpoint=check,
                    fname=f"jsdiv_{fam}{check}_{file_type}",
                )


def colorBar(fig, normalized, env):
    cbar_ax = fig.add_axes([0.1, 0.0, 0.80, 0.02])
    if env == "Breakout":
        cbar_ax.set_title(
            r"Interventional Robustness ($\mathcal{R}$)",
            {"fontsize": 8, "fontweight": "bold"},
            y=-2.2,
        )
    else:
        cbar_ax.set_title(
            r"Interventional Robustness ($\mathcal{R}$)",
            {"fontsize": 8, "fontweight": "bold"},
            y=-1.8,
        )
    vmin = -1 if normalized else 0
    plt.colorbar(
        cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=1), cmap=cmap),
        cax=cbar_ax,
        orientation="horizontal",
    )
    cbar_ax.tick_params(labelsize=6)


def plotPerformance(fig, env):
    gs = fig.add_gridspec(
        len(model_names), 1, left=0.95, right=1.1, top=0.95, bottom=0.07
    )
    plotAllFamilies(env, gs)


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
    wspace = {"Breakout": 0.2, "Amidar": 0.1, "SpaceInvaders": 0.0}[env]

    plt.subplots_adjust(
        top=topshift,
        bottom=bottomshift,
        left=leftshift,
        right=rightshift,
        hspace=hspace,
        wspace=wspace,
    )


def megaPlot(normalized, nAgents=11, nStates=30, env=None):
    """
    https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subfigures.html#sphx-glr-gallery-subplots-axes-and-figures-subfigures-py
    """
    font = {"size": 10}
    matplotlib.rc("font", **font)

    if env is None:
        env = sys.argv[1]

    with open(f"storage/plots/returns/{env}/order.txt") as f:
        model_names = [l.strip() for l in f.readlines()]

    figsize = (3.5, 8) if env == "Breakout" else (3.5, 10)

    fig, axes = plt.subplots(
        len(model_names),
        len(checkpoints),
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
            ax = axes[f, c]
            if c == len(checkpoints) - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(
                    ylabel=f"{fam.upper()}",
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
                    rotation=-90,
                )
            if f == 0:
                order = int(np.floor(np.log10(check)))
                ax.set_title(f"{check // 10**order}e{order}", {"fontsize": 10})
            if f == len(model_names) - 1:
                ax.tick_params(labelbottom=True)

            dir = f"storage/results/intervention_js_div/{env}/{fam}/{nAgents}_agents/{nStates}_states/trajectory/check_{check}"

            vdata = np.loadtxt(dir + f"/vanilla.txt")
            data = np.loadtxt(dir + f"/{nintv}_interventions.txt")

            subplot_js_divergence_matrix(
                ax,
                data,
                vdata,
                normalized,
                title=format(performance[c] / performanceOrder, ".2f")
                # + f"e{performanceOrder}",
            )
            print(f"plot, {f},{c}")

    resizeMegaplot(env, normalized)
    plt.figure(fig.number)
    colorBar(fig, normalized, env)
    plotPerformance(fig, env)

    title_type = "Normalized " if normalized else ""
    file_type = "normalized" if normalized else "unnormalized"

    # fig.suptitle(f"{title_type}Interventional Robustness: {env}", fontsize=11)

    plt.margins(0)
    # annotations
    framesXY = (0.01, 1.002 if env == "Breakout" else 0.999)
    plt.annotate(
        "Frames:",
        xy=framesXY,
        xytext=framesXY,
        textcoords="figure fraction",
        fontsize=8,
        fontweight="bold",
    )

    framesXY = (0.03, 0.46)
    plt.annotate(
        "Intervention",
        xy=framesXY,
        xytext=framesXY,
        textcoords="figure fraction",
        rotation=90,
        size=10,
        fontweight="bold",
    )

    framesXY = (1.15, 0.45)
    plt.annotate(
        "Performance",
        xy=framesXY,
        xytext=framesXY,
        textcoords="figure fraction",
        rotation=-90,
        size=10,
        fontweight="bold",
    )

    framesXY = (0.53, 0.08 if env == "Breakout" else 0.07)
    plt.annotate(
        "State",
        xy=framesXY,
        xytext=framesXY,
        textcoords="figure fraction",
        size=10,
        fontweight="bold",
    )

    # save
    plt.savefig(
        f"storage/plots/sampled_jsdivmat/{env}_{file_type}.png",
        bbox_inches="tight",
        dpi=600,
    )


if __name__ == "__main__":

    if len(sys.argv) > 1:
        print(f"Making megaplot for {sys.argv[1]}")
        if len(sys.argv) > 2:
            megaPlot(True)
        else:
            megaPlot(False)
    else:
        print("Plotting individual plots")
        individualPlots(False)
        individualPlots(True)
