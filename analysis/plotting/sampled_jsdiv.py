import os
import sys

import matplotlib
from matplotlib import cm, rcParams
from matplotlib.colors import Normalize
from numpy.lib.npyio import save
from analysis.plotting.tables import print_image_name_table, print_values_table

from analysis.src.js_divergence import get_js_divergence_matrix
from envs.wrappers.paths import get_num_interventions


import matplotlib.pyplot as plt
import numpy as np

from runners.src.run_intervention_eval import (
    supported_environments,
    model_names,
    checkpoints,
)
from runners.src.performance_plot import *

# cmap = plt.get_cmap("magma").reversed()
cmap = rcParams["image.cmap"]
cmap = plt.get_cmap(cmap).reversed()
showPoints = True


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

    if showPoints:
        if title != "":
            ax.set_xlabel(title, {"fontsize": 8})
            ax.xaxis.set_label_position("top")


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

    if showPoints:
        figsize = (3.5, 9) if env == "Breakout" else (3.5, 10)
    else:
        figsize = (3.5, 6) if env == "Breakout" else (3.5, 10)

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

        # performanceOrder = int(np.floor(np.max(np.log10(performance))))
        performanceOrder = 1000

        # add jsdiv plots
        for c, check in enumerate(checkpoints):
            ax = axes[f, c]
            if c == 0:
                ax.set_ylabel(f"{fam.upper()}", {"fontsize": 8})
            if c == len(checkpoints) - 1:
                ax.set_ylabel(f"{fam.upper()}", {"fontsize": 8})
                ax.yaxis.set_label_position("right")
            if f == 0:
                order = int(np.floor(np.log10(check)))
                ax.set_title(f"{check // 10**order}e{order}", {"fontsize": 10})
                # ax.set_xlabel(f"{check} Frames")
                # ax.xaxis.set_label_position("top")
            if f == len(model_names) - 1:
                ax.tick_params(labelbottom=True)
                # ax.set_xlabel(f"State")
            # else:
            #     ax = None

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

            # with showPoints
            if showPoints:
                topshift = {"Breakout": 0.945, "Amidar": 0.92, "SpaceInvaders": 0.92}[
                    env
                ]
                bottomshift = {"Breakout": 0.05, "Amidar": 0.07, "SpaceInvaders": 0.07}[
                    env
                ]
                rightshift = {"Breakout": 1, "Amidar": 1, "SpaceInvaders": 1}[env]
                leftshift = {
                    "Breakout": 0.125,
                    "Amidar": 0.125,
                    "SpaceInvaders": 0.125,
                }[env]
                hspace = {"Breakout": 0, "Amidar": 0.5, "SpaceInvaders": 0.3}[env]
                wspace = {"Breakout": 0.3, "Amidar": 0.15, "SpaceInvaders": 0.2}[env]
            else:
                # without showPoints
                topshift = {"Breakout": 0.945, "Amidar": 0.92, "SpaceInvaders": 0.935}[
                    env
                ]
                bottomshift = {"Breakout": 0.05, "Amidar": 0.07, "SpaceInvaders": 0.07}[
                    env
                ]
                rightshift = {"Breakout": 1, "Amidar": 1, "SpaceInvaders": 1}[env]
                leftshift = {
                    "Breakout": 0.15,
                    "Amidar": 0.13,
                    "SpaceInvaders": 0.13,
                }[env]
                hspace = {"Breakout": -0.7, "Amidar": 0.5, "SpaceInvaders": 0.1}[env]
                wspace = {"Breakout": 0.3, "Amidar": 0.15, "SpaceInvaders": 0.2}[env]

            plt.subplots_adjust(
                top=topshift,
                bottom=bottomshift,
                left=leftshift,
                right=rightshift,
                hspace=hspace,
                wspace=wspace,
            )
            plt.figure(fig.number)

            # hpad = {"Breakout": 1, "Amidar": -5, "SpaceInvaders": -15}
            # wpad = {"Breakout": -20, "Amidar": 1, "SpaceInvaders": 1}
            # plt.tight_layout(h_pad=hpad[env], w_pad=wpad[env])
            # plt.tight_layout()
            # plt.constrained_layout()
            cbar_ax = fig.add_axes([0.15, 0.0, 0.85, 0.02])
            cbar_ax.set_title(
                r"Interventional Robustness ($\mathcal{R}$)", {"fontsize": 8}, y=-1.8
            )
            vmin = -1 if normalized else 0
            plt.colorbar(
                cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=1), cmap=cmap),
                cax=cbar_ax,
                orientation="horizontal",
            )
            cbar_ax.tick_params(labelsize=6)
            print(f"plot, {f},{c}")

    title_type = "Normalized " if normalized else ""
    file_type = "normalized" if normalized else "unnormalized"

    fig.suptitle(f"{title_type}Interventional Robustness: {env}", fontsize=11)

    plt.margins(0)
    # annotations
    if showPoints:
        framesXY = (0.02, 0.9675)
        plt.annotate(
            "Frames:\nPoints:",
            xy=framesXY,
            xytext=framesXY,
            textcoords="figure fraction",
            fontsize=8,
        )
    else:
        framesXY = (0.025, 0.985)
        plt.annotate(
            "Frames:",
            xy=framesXY,
            xytext=framesXY,
            textcoords="figure fraction",
            fontsize=8,
        )

    framesXY = (0.01, 0.5)
    plt.annotate(
        "Intervention",
        xy=framesXY,
        xytext=framesXY,
        textcoords="figure fraction",
        rotation=90,
        size=10,
    )

    if showPoints:
        framesXY = (0.54, 0.07)
        plt.annotate(
            "State", xy=framesXY, xytext=framesXY, textcoords="figure fraction", size=10
        )
    else:
        framesXY = (0.53, 0.0725)
        plt.annotate(
            "State", xy=framesXY, xytext=framesXY, textcoords="figure fraction", size=10
        )

    plt.savefig(
        f"storage/plots/sampled_jsdivmat/{env}_{file_type}.png",
        bbox_inches="tight",
        dpi=600,
    )
    # plt.subplot_tool(targetfig=fig)
    # plt.show()
    # plt.close(fig)


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
