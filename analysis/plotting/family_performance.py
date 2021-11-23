import numpy as np
import matplotlib.pyplot as plt
from envs.wrappers.paths import get_num_interventions

from runners.src.result_paths import (
    get_intervention_data_dir,
    get_trajectory_intervention_data_dir,
)

from analysis.src.js_divergence import get_js_divergence_matrix
from runners.src.run_intervention_eval import checkpoints


def get_family_jsdiv(
    family,
    environment,
    num_agents,
    num_states,
):
    files = {}
    nintv = get_num_interventions(environment)
    for checkpoint in checkpoints:
        dir = get_trajectory_intervention_data_dir(
            family,
            environment,
            num_agents,
            num_states,
            checkpoint,
            True,
        )
        vdata = np.loadtxt(dir + f"/vanilla.txt")
        data = np.loadtxt(dir + f"/{nintv}_interventions.txt")
        mat, nmat, van_mat, intv_mat = get_js_divergence_matrix(data, vdata)


if __name__ == "__main__":
    family = "a2c"
    environment = "SpaceInvaders"
    num_agents = 11
    num_states = 30

    get_family_jsdiv(family, environment, num_agents, num_states)
