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


def subplot_js_divergence_matrix(ax, data, vanilla, normalize, title=""):
    font = {"size": 4}
    matplotlib.rc("font", **font)

    mat, nmat, van_mat, intv_mat, n_intv_mat = get_js_divergence_matrix(data, vanilla)

    if normalize:
        mat = nmat

    plt.sca(ax)
    im = ax.imshow(mat, interpolation="none")

    if normalize:
        im.set_clim(-1.0, 1.0)
    else:
        im.set_clim(0, 1.0)

    ax.set_xticks(list(range(0, mat.shape[1], 20)))
    ax.set_xticklabels(list(range(0, mat.shape[1], 20)))

    ax.set_yticks(list(range(0, mat.shape[0] + 1, 10)))
    ax.set_yticklabels(list(range(0, mat.shape[0] + 1, 10)))

    ax.tick_params(left=False, bottom=False)

    if title != "":
        ax.set_xlabel(title)
        ax.xaxis.set_label_position("top")


def plot_js_divergence_matrix(
    data, vanilla, title, normalize, env, family="", checkpoint="", fname=None
):
    mat, nmat, van_mat, intv_mat, n_intv_mat = get_js_divergence_matrix(data, vanilla)
    if normalize:
        mat = nmat

    ax = plt.gca()

    im = plt.matshow(mat, interpolation="none", aspect="auto")

    if normalize:
        im.set_clim(-1.0, 1.0)
    else:
        im.set_clim(0, 1.0)

    ax.tick_params(axis="x", top=False, bottom=True, labelbottom=True, labeltop=False)
    plt.xlabel("Intervention")
    plt.ylabel("State")

    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.set_label("JS Divergence of Action Distributions")
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
    font = {"size": 14}
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

                type = "Normalized" if normalized else "Unnormalized"
                name = f"{type} JS Divergence over Actions\nfor {fam} at {check}, {env}"
                plot_js_divergence_matrix(
                    data,
                    vdata,
                    name,
                    normalized,
                    env,
                    family=fam,
                    checkpoint=check,
                    fname=f"jsdiv_{fam}{check}_{type.lower()}",
                )


def megaPlot(normalized, nAgents=11, nStates=30, env=None):
    """
    https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subfigures.html#sphx-glr-gallery-subplots-axes-and-figures-subfigures-py
    """
    font = {"size": 4}
    matplotlib.rc("font", **font)

    if env is None:
        env = sys.argv[1]

    with open(f"storage/plots/returns/{env}/order.txt") as f:
        model_names = [l.strip() for l in f.readlines()]

    fig, axes = plt.subplots(
        len(model_names), len(checkpoints), sharex=True, sharey=True
    )

    nintv = get_num_interventions(env)

    for f, fam in enumerate(model_names):
        # add performance at rightmost side
        data = load_returns_100_data(f"storage/models/{env}/{fam}")[env + "Toybox"]
        performance = get_checkpoint_performances(
            f"storage/models/{env}", env, fam, checkpoints
        )

        # add jsdiv plots
        for c, check in enumerate(checkpoints):
            ax = axes[f, c]
            if c == 0:
                ax.set_ylabel(f"{fam.upper()}\nState")
            if c == len(checkpoints) - 1:
                ax.set_ylabel(f"{fam.upper()}")
                ax.yaxis.set_label_position("right")
            if f == 0:
                ax.set_title(f"{check} Frames")
                # ax.set_xlabel(f"{check} Frames")
                # ax.xaxis.set_label_position("top")
            if f == len(model_names) - 1:
                ax.tick_params(labelbottom=True)
                ax.set_xlabel(f"Intervention")
            # else:
            #     ax = None

            dir = f"storage/results/intervention_js_div/{env}/{fam}/{nAgents}_agents/{nStates}_states/trajectory/check_{check}"

            vdata = np.loadtxt(dir + f"/vanilla.txt")
            data = np.loadtxt(dir + f"/{nintv}_interventions.txt")

            subplot_js_divergence_matrix(
                ax, data, vdata, normalized, title=format(performance[c], ".3E")
            )

            if env == "Breakout":
                matplotlib.rcParams["figure.figsize"] = 100, 100
            else:
                matplotlib.rcParams["figure.figsize"] = 100, 50

            cmap = rcParams["image.cmap"]

            topshift = {"Breakout": 0.89, "Amidar": 0.90, "SpaceInvaders": 0.91}[env]

            plt.figure(fig.number)
            if env == "Breakout":
                fig.subplots_adjust(right=0.85, top=topshift)
            else:
                fig.subplots_adjust(right=0.85, top=topshift)

            # hpad = {"Breakout": 1, "Amidar": -5, "SpaceInvaders": -15}
            # wpad = {"Breakout": -20, "Amidar": 1, "SpaceInvaders": 1}
            # plt.tight_layout(h_pad=hpad[env], w_pad=wpad[env])
            # plt.tight_layout()
            # plt.constrained_layout()

            cbar_ax = fig.add_axes([0.89, 0.09, 0.025, 0.825])
            cbar_ax.set_title("JS Divergence")
            vmin = -1 if normalized else 0
            fig.colorbar(
                cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=1), cmap=cmap),
                cax=cbar_ax,
            )

            type = "Normalized" if normalized else "Unnormalized"

            fig.suptitle(f"{type} JS Divergence over Actions: {env}", fontsize=10)
            plt.savefig(
                f"storage/plots/sampled_jsdivmat/{env}_{type.lower()}.png",
                bbox_inches="tight",
                dpi=600,
            )
            plt.subplot_tool(targetfig=fig)
            plt.close(fig)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        print(f"Making megaplot for {sys.argv[1]}")
        # megaPlot(True)
        megaPlot(False)
    else:
        print("Plotting individual plots")
        individualPlots(False)
        individualPlots(True)
