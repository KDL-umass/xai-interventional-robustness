import os

import matplotlib
from matplotlib import cm, rcParams
from matplotlib.colors import Normalize
from numpy.lib.npyio import save
from analysis.plotting.tables import print_image_name_table, print_values_table

from analysis.src.js_divergence import get_js_divergence_matrix
from envs.wrappers.paths import get_num_interventions


import matplotlib.pyplot as plt
import numpy as np

from runners.src.run_intervention_eval import supported_environments, model_names


def plot_js_divergence_matrix(
    data, vanilla, title, normalize, env, family="", checkpoint="", fname=None, ax=None
):
    save_here = ax is None

    mat, nmat, van_mat, intv_mat = get_js_divergence_matrix(data, vanilla)
    if normalize:
        mat = nmat

    if ax is not None:
        plt.sca(ax)
    ax = plt.gca()

    if save_here:
        im = plt.matshow(mat, interpolation="none")
    else:
        im = ax.imshow(mat, interpolation="none")

    if normalize:
        im.set_clim(-1.0, 1.0)
    else:
        im.set_clim(0, 1.0)

    if save_here:
        ax.tick_params(
            axis="x", top=False, bottom=True, labelbottom=True, labeltop=False
        )
    else:
        ax.set_xticks(list(range(0, mat.shape[1], 5)))
        ax.set_yticks(list(range(0, mat.shape[0], 5)))
        ax.tick_params(
            axis="both",
            top=False,
            bottom=True,
            labelbottom=False,
            labeltop=False,
            left=True,
            right=False,
            labelleft=False,
            labelright=False,
            which="minor",
            labelsize=1,
            width=1,
            length=1,
            direction="out",
        )
        ax.set_title(f"{family} at {checkpoint} frames")
    plt.xlabel("Intervention Number")
    plt.ylabel("State of Interest")

    if save_here:
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


if __name__ == "__main__":
    n_agents = 11
    nstates = 30
    folder = "intervention_js_div"

    checkpoints = [10 ** i for i in range(2, 8)]

    model_names = ["vqn", "vsarsa", "ppo", "dqn", "a2c"]
    # supported_environments = ["SpaceInvaders"]
    # supported_environments = ["Amidar"]
    supported_environments = ["Breakout"]

    vanilla_dict = {}
    unnormalized_dict = {}
    normalized_dict = {}

    megaPlot = True

    if megaPlot:
        font = {"size": 2}
    else:
        font = {"size": 14}

    matplotlib.rc("font", **font)

    for env in supported_environments:
        if megaPlot:
            fig, axes = plt.subplots(
                len(model_names),
                len(checkpoints),
            )
            normfig, normaxes = plt.subplots(
                len(model_names),
                len(checkpoints),
            )

        nintv = get_num_interventions(env)
        vanilla_dict[env] = {}
        unnormalized_dict[env] = {}
        normalized_dict[env] = {}

        for f, fam in enumerate(model_names):
            vanilla_dict[env][fam] = {}
            unnormalized_dict[env][fam] = {}
            normalized_dict[env][fam] = {}

            for c, check in enumerate(checkpoints):
                if megaPlot:
                    ax = axes[f, c]
                    normax = normaxes[f, c]
                else:
                    ax = None
                    normax = None

                if check == "":
                    dir = f"storage/results/{folder}/{env}/{fam}/{n_agents}_agents/{nstates}_states/trajectory"
                else:
                    dir = f"storage/results/{folder}/{env}/{fam}/{n_agents}_agents/{nstates}_states/trajectory/check_{check}"

                vdata = np.loadtxt(dir + f"/vanilla.txt")
                data = np.loadtxt(dir + f"/{nintv}_interventions.txt")

                normname = f"Normalized JS Divergence over Actions\nfor {fam} at {check}, {env}"
                _, normalized_dict[env][fam][check] = plot_js_divergence_matrix(
                    data,
                    vdata,
                    normname,
                    True,
                    env,
                    family=fam,
                    checkpoint=check,
                    fname=f"jsdiv_{fam}{check}_normalized",
                    ax=normax,
                )

                fname = f"Unnormalized JS Divergence over Actions\nfor {fam} at {check}, {env}"
                (
                    vanilla_dict[env][fam][check],
                    unnormalized_dict[env][fam][check],
                ) = plot_js_divergence_matrix(
                    data,
                    vdata,
                    fname,
                    False,
                    env,
                    family=fam,
                    checkpoint=check,
                    fname=f"jsdiv_{fam}{check}",
                    ax=ax,
                )

        if megaPlot:
            matplotlib.rcParams["figure.figsize"] = 100, 50

            cmap = rcParams["image.cmap"]

            topshift = {"Breakout": 0.93, "Amidar": 0.97, "SpaceInvaders": 0.99}[env]
            # UNNORMALIZED
            plt.figure(fig.number)
            plt.tight_layout()
            fig.subplots_adjust(right=0.85, top=topshift)
            cbar_ax = fig.add_axes([0.89, 0.09, 0.025, 0.825])
            cbar_ax.set_title("JS Divergence")
            fig.colorbar(
                cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap),
                cax=cbar_ax,
            )

            fig.suptitle(f"Unnormalized JS Divergence over Actions: {env}", fontsize=10)
            plt.savefig(
                f"storage/plots/sampled_jsdivmat/{env}_matrix.png",
                bbox_inches="tight",
                dpi=600,
            )
            plt.close(fig)

            # NORMALIZED
            plt.figure(normfig.number)
            plt.tight_layout()
            normfig.subplots_adjust(right=0.85, top=topshift)
            cbar_ax = normfig.add_axes([0.89, 0.09, 0.025, 0.825])
            cbar_ax.set_title("JS Divergence")
            normfig.colorbar(
                cm.ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=cmap),
                cax=cbar_ax,
            )
            normfig.suptitle(
                f"Normalized JS Divergence over Actions: {env}", fontsize=10
            )
            plt.savefig(
                f"storage/plots/sampled_jsdivmat/{env}_normalized.png",
                bbox_inches="tight",
                dpi=600,
            )
            plt.close(normfig)

        print_image_name_table(model_names, env)
        print_values_table(
            env,
            model_names,
            checkpoints,
            vanilla_dict,
            unnormalized_dict,
            normalized_dict,
        )
