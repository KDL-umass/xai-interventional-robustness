import numpy as np
from analysis.src.js_divergence import get_js_divergence_matrix
from envs.wrappers.paths import get_num_interventions


from runners.src.run_intervention_eval import (
    supported_environments,
    model_names,
    checkpoints,
)


def print_image_name_table(families, env):
    print()
    print("\\begin{tabular}{ccc}")
    print("Family & Unnormalized & Normalized \\\\")
    for fam in families:
        print(
            fam
            + " & \\includegraphics[width=0.4\\textwidth]{plots/"
            + env
            + "/jsdiv_"
            + fam
            + ".png} & \includegraphics[width=0.4\\textwidth]{plots/"
            + env
            + "/jsdiv_"
            + fam
            + "_normalized.png}\\\\"
        )
    print("\\end{tabular}")


def print_values_table(
    env, families, checkpoints, vanilla_dict, unnormalized_dict, normalized_dict
):
    F = len(families)
    C = len(checkpoints)
    table = np.zeros((F, C, 3))

    print()
    print("\\begin{tabular}{|l|l|c|c|c|}\\hline")
    print("Family & Checkpoint & Original & Unnormalized & Normalized \\\\\\hline")

    for f, fam in enumerate(families):
        for c, check in enumerate(checkpoints):
            v = vanilla_dict[env][fam][check]
            u = unnormalized_dict[env][fam][check]
            n = normalized_dict[env][fam][check]
            table[f, c, :] = v, u, n

            if check == "":
                check = 10000000

            print(
                f"{fam} & {'{:.0e}'.format(check)} & {round(v, 4)} & {round(u, 4)} & {round(n, 4)} \\\\\\hline"
            )
    print("\\end{tabular}")

    return table


def makeTables(nAgents=11, nStates=30):
    vanilla_dict = {}
    unnormalized_dict = {}
    normalized_dict = {}

    for env in supported_environments:
        nintv = get_num_interventions(env)

        vanilla_dict[env] = {}
        unnormalized_dict[env] = {}
        normalized_dict[env] = {}

        for fam in model_names:
            vanilla_dict[env][fam] = {}
            unnormalized_dict[env][fam] = {}
            normalized_dict[env][fam] = {}

            for check in checkpoints:
                dir = f"storage/results/intervention_js_div/{env}/{fam}/{nAgents}_agents/{nStates}_states/trajectory/check_{check}"

                vdata = np.loadtxt(dir + f"/vanilla.txt")
                data = np.loadtxt(dir + f"/{nintv}_interventions.txt")

                mat, nmat, van_mat, intv_mat, n_intv_mat = get_js_divergence_matrix(
                    data, vdata
                )

                vanilla_dict[env][fam][check] = van_mat.mean()
                normalized_dict[env][fam][check] = n_intv_mat.mean()
                unnormalized_dict[env][fam][check] = intv_mat.mean()

        print_image_name_table(model_names, env)
        print_values_table(
            env,
            model_names,
            checkpoints,
            vanilla_dict,
            unnormalized_dict,
            normalized_dict,
        )


if __name__ == "__main__":
    makeTables()
