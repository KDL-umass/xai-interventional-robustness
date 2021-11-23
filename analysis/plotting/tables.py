import numpy as np


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
