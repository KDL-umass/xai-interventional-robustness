import os

import matplotlib
from analysis.plotting.tables import print_image_name_table, print_values_table

from analysis.src.js_divergence import get_js_divergence_matrix
from envs.wrappers.paths import get_num_interventions

font = {"size": 14}

matplotlib.rc("font", **font)

import matplotlib.pyplot as plt
import numpy as np


def plot_js_divergence_matrix(data, vanilla, title, normalize, env, fname=None):
    mat, nmat, van_mat, intv_mat = get_js_divergence_matrix(data, vanilla)
    if normalize:
        mat = nmat

    im = plt.matshow(mat, interpolation="none")
    cbar = plt.colorbar(im)
    cbar.set_label("JS Divergence of Action Distributions")
    if normalize:
        plt.clim(-1.0, 1.0)
    else:
        plt.clim(0, 1.0)
    ax = plt.gca()
    ax.tick_params(axis="x", top=False, bottom=True, labelbottom=True, labeltop=False)
    plt.title(title)
    plt.xlabel("Intervention Number")
    plt.ylabel("State of Interest")

    os.makedirs(f"storage/plots/sampled_jsdivmat/{env}", exist_ok=True)
    if fname is not None:
        plt.savefig(
            f"storage/plots/sampled_jsdivmat/{env}/{fname}.png", bbox_inches="tight"
        )
    else:
        plt.savefig(
            f"storage/plots/sampled_jsdivmat/{env}/{title}.png", bbox_inches="tight"
        )

    return van_mat.mean(), intv_mat.mean()


if __name__ == "__main__":
    n_agents = 11
    nstates = 30
    folder = "intervention_js_div"

    families = ["a2c", "dqn", "ddqn", "c51", "rainbow", "vsarsa", "vqn", "ppo"]
    # checkpoints = [str(100 * 10 ** i) for i in range(6)]
    checkpoints = [""]

    vanilla_dict = {}
    unnormalized_dict = {}
    normalized_dict = {}

    for env in ["SpaceInvaders"]:
        nintv = get_num_interventions(env)
        vanilla_dict[env] = {}
        unnormalized_dict[env] = {}
        normalized_dict[env] = {}

        for fam in families:
            vanilla_dict[env][fam] = {}
            unnormalized_dict[env][fam] = {}
            normalized_dict[env][fam] = {}

            for check in checkpoints:
                if check == "":
                    dir = f"storage/results/{folder}/{env}/{fam}/{n_agents}_agents/{nstates}_states/trajectory"
                else:
                    dir = f"storage/results/{folder}/{env}/{fam}/{n_agents}_agents/{nstates}_states/trajectory/check_{check}"
                vdata = np.loadtxt(dir + f"/vanilla.txt")
                data = np.loadtxt(dir + f"/{nintv}_interventions.txt")
                _, normalized_dict[env][fam][check] = plot_js_divergence_matrix(
                    data,
                    vdata,
                    f"Normalized Sampled JS Divergence over Actions\nfor {fam} at {check}, {env}",
                    True,
                    env,
                    fname=f"jsdiv_{fam}{check}_normalized",
                )
                (
                    vanilla_dict[env][fam][check],
                    unnormalized_dict[env][fam][check],
                ) = plot_js_divergence_matrix(
                    data,
                    vdata,
                    f"Unnormalized Sampled JS Divergence over Actions\nfor {fam} at {check}, {env}",
                    False,
                    env,
                    fname=f"jsdiv_{fam}{check}",
                )

        print_image_name_table(families, env)
        print_values_table(
            env, families, checkpoints, vanilla_dict, unnormalized_dict, normalized_dict
        )
