import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import difflib, statistics
from collections import Counter, defaultdict

mpl.rcParams["figure.dpi"] = 250

# Define constants specifically for Space Invaders
ACTION_MEANING_SPI = {
    0: "NOOP",
    1: "FIRE",
    2: "RIGHT",
    3: "LEFT",
    4: "RIGHTFIRE",
    5: "LEFTFIRE",
}

ACTION_LOOKUP = {v: k for (k, v) in ACTION_MEANING_SPI.items()}

intv_name = dict(
    [(n, "Drop one enemy") for n in range(36)]
    + [(n, "Shift shields") for n in range(36, 65)]
    + [(n, "Shift agent") for n in range(65, 75)]
    + [(n, "Drop full row and column") for n in range(75, 87)]
)
intv_number = {
    "drop_enemy": list(range(36)),
    "shift_shields": list(range(36, 65)),
    "shift_agent": list(range(65, 75)),
    "drop_row_col": list(range(75, 87)),
}
seed = list(range(30))

"""
TASKS: 
1. Compare action trajectories for any given seed and intervention type 
2. Summarize matching % for any intervention number 
3. Summarize matching % for any intervention type 
4. Summarize action frequencies in terms of firing / moving 
5. Action transition diagrams 
"""

# Input parameters (seed, intv, agent_name) compare individual trajectories
def compare_trajectories(intv=0, seed=0, agent_name="cnn"):
    dir_path = "storage/results/action_trajectories/{}/{}/{}/".format(intv, seed, agent_name)
    action_file = dir_path + str(agent_name) + ".txt"
    with open(action_file, "r") as f:
        seq = f.readlines()

    dir_path_vanilla = "storage/results/action_trajectories/{}/{}/{}/".format(
        -1, seed, agent_name
    )
    action_file_vanilla = dir_path_vanilla + str(agent_name) + ".txt"
    with open(action_file_vanilla, "r") as f:
        seq_vanilla = f.readlines()

    act_seq = seq[0].rstrip("\n")
    act_seq_vanilla = seq_vanilla[0].rstrip("\n")

    rw = float(seq[1].rstrip("\n"))
    rw_vanilla = float(seq_vanilla[1].rstrip("\n"))

    print("Intervention: ", intv_name[intv])
    if len(act_seq) > len(act_seq_vanilla):
        print("Longer trajectory after intervention")
    elif len(act_seq) == len(act_seq_vanilla):
        print("Equal length trajectories;", end="  ")
        if act_seq == act_seq_vanilla:
            print("No change after intervention")
    else:
        print("Shorter trajectory after intervention")

    print("Reward w/ intervention: ", rw)
    print("Reward w/o intervention: ", rw_vanilla)

    s = difflib.SequenceMatcher(None, act_seq, act_seq_vanilla)
    for block in s.get_matching_blocks():
        if block[0] == 0:
            match_percent = block[2] / (min(len(act_seq), len(act_seq_vanilla))) * 100
        else:
            match_percent = 0
        break
    # print("{:.2f} %".format(match_percent))
    return round(match_percent, 2)


# Summarize trajectories by intervention number for all seeds
def compare_trajectories_seed(intv=1, agent_name="cnn"):
    matching = []
    for s in seed:
        dir_path = "storage/results/action_trajectories/{}/{}/{}/".format(
            intv, s, agent_name
        )
        action_file = dir_path + str(agent_name) + ".txt"
        with open(action_file, "r") as f:
            seq = f.readlines()

        dir_path_vanilla = "storage/results/action_trajectories/{}/{}/{}/".format(
            -1, s, agent_name
        )
        action_file_vanilla = dir_path_vanilla + str(agent_name) + ".txt"
        with open(action_file_vanilla, "r") as f:
            seq_vanilla = f.readlines()

        act_seq = seq[0].rstrip("\n")
        act_seq_vanilla = seq_vanilla[0].rstrip("\n")

        sq = difflib.SequenceMatcher(None, act_seq, act_seq_vanilla)
        for block in sq.get_matching_blocks():
            if block[0] == 0:
                match_percent = (
                    block[2] / (min(len(act_seq), len(act_seq_vanilla))) * 100
                )
            else:
                match_percent = 0
            break
        matching.append(match_percent)

    # print('Intervention type: ', intv_name[intv], ' Match: {:.2f}'.format(np.mean(matching)))
    return round(np.mean(matching), 2)


# Summarize trajectories by intervention type
def compare_trajectories_intv(int_name="drop_enemy", agent_name="cnn"):
    matching = []
    for i in intv_number[int_name]:
        matching.append(compare_trajectories_seed(intv=i, agent_name=agent_name))

    # print('Intervention type: ', int_name, ' Match: {:.2f}'.format(np.mean(matching)))
    return round(np.mean(matching), 2)


# For a randomly chosen intervention number, we plot the matching percentage for every agent
def plot_trajectories_seed():
    intv_list = [
        np.random.choice(np.arange(36)),
        np.random.choice(np.arange(36, 65)),
        np.random.choice(np.arange(65, 75)),
        np.random.choice(np.arange(75, 87)),
    ]
    intv_labels = ["drop_enemy", "shift_shield", "shift_agent", "drop_row_col"]
    intv_label_names = [
        "drop_enemy_" + str(intv_list[0]),
        "shift_shield_" + str(intv_list[1]),
        "shift_agent_" + str(intv_list[2]),
        "drop_row_col_" + str(intv_list[3]),
    ]

    for agent in ["cnn", "ddt", "random"]:
        plt.figure(figsize=(14, 5), facecolor="white")
        plt.rc("font", size=13)
        match_percent = []
        for intv in intv_list:
            match_percent.append(compare_trajectories_seed(intv=intv, agent_name=agent))

        plt.barh(np.arange(len(intv_label_names)), match_percent, align="center")
        plt.yticks(np.arange(len(intv_label_names)), intv_label_names)
        plt.xticks([])
        plt.xlim([0, 100])
        plt.title("Agent: " + str(agent))
        for i, v in enumerate(match_percent):
            plt.text(v, i, " " + str(v), color="black", va="center", fontweight="bold")


# For a given intervention type, we plot the matching percentage summarized across all intervention numbers and seeds for that intervention
def plot_trajectories_intv(int_type):
    agent = "cnn"

    if int_type == "drop_one_enemy":
        plt.figure(figsize=(14, 5), facecolor="white")
        plt.rc("font", size=13)
        match_percent = []
        for i in np.arange(36):
            match_percent.append(compare_trajectories_seed(intv=i, agent_name=agent))

        plt.bar(np.arange(36), match_percent, align="center")
        plt.xticks(np.arange(36))
        plt.ylim([0, 100])
        plt.title("Agent: " + str(agent) + " : Drop one enemy")

    elif int_type == "shift_shields":
        match_percent = []
        plt.figure(figsize=(14, 5), facecolor="white")
        plt.rc("font", size=13)
        for i in np.arange(36, 65):
            match_percent.append(compare_trajectories_seed(i, agent))

        plt.bar(np.arange(36, 65), match_percent, align="center")
        plt.xticks(np.arange(36, 65))
        plt.ylim([0, 100])
        plt.title("Agent: " + str(agent) + " : Shift shields")

    elif int_type == "shift_agent":
        match_percent = []
        plt.figure(figsize=(14, 5), facecolor="white")
        plt.rc("font", size=13)
        for i in np.arange(65, 75):
            match_percent.append(compare_trajectories_seed(i, agent))

        plt.bar(np.arange(65, 75), match_percent, align="center")
        plt.xticks(np.arange(65, 75))
        plt.ylim([0, 100])
        plt.title("Agent: " + str(agent) + " : Shift agent")

    elif int_type == "drop_row_col":
        match_percent = []
        plt.figure(figsize=(14, 5), facecolor="white")
        plt.rc("font", size=13)
        for i in np.arange(75, 87):
            match_percent.append(compare_trajectories_seed(i, agent))

        plt.bar(np.arange(75, 87), match_percent, align="center")
        plt.xticks(np.arange(75, 87))
        plt.ylim([0, 100])
        plt.title("Agent: " + str(agent) + " : Drop row/column")


def plot_trajectories_summary(agent):
    # All interventions
    interventions = ["drop_enemy", "shift_shields", "shift_agent", "drop_row_col"]

    plt.figure(figsize=(14, 5), facecolor="white")
    plt.rc("font", size=13)
    match_percent = []
    for iv in interventions:
        match_percent.append(compare_trajectories_intv(int_name=iv, agent_name=agent))

    plt.barh(np.arange(len(interventions)), match_percent, align="center")
    plt.yticks(np.arange(len(interventions)), interventions)
    plt.xticks([])
    plt.xlim([0, 100])
    plt.title("Agent: " + str(agent))
    for i, v in enumerate(match_percent):
        plt.text(v, i, " " + str(v), color="black", va="center", fontweight="bold")


#####################################################################################################

# Summarize action frequencies
def action_freq(seed, intv, agent_name):
    dir_path = "storage/results/action_trajectories/{}/{}/{}/".format(
        intv, seed, agent_name
    )
    action_file = dir_path + str(agent_name) + ".txt"
    with open(action_file, "r") as f:
        seq = f.readlines()

    dir_path_vanilla = "storage/results/action_trajectories/{}/{}/{}/".format(
        -1, seed, agent_name
    )
    action_file_vanilla = dir_path_vanilla + str(agent_name) + ".txt"
    with open(action_file_vanilla, "r") as f:
        seq_vanilla = f.readlines()

    act_seq = list(seq[0].rstrip("\n"))
    act_seq_vanilla = list(seq_vanilla[0].rstrip("\n"))

    rw = float(seq[1].rstrip("\n"))
    rw_vanilla = float(seq_vanilla[1].rstrip("\n"))

    actions = {}
    actions_vanilla = {}
    print("#################")
    act_seq_freq = Counter(act_seq).most_common()
    print("W/ Intervention; Episode Reward: ", rw)
    for a, b in act_seq_freq:
        actions[a] = b
        print(ACTION_MEANING_SPI[int(a)], ":", b)

    print("#################")
    print("Vanilla; Episode Reward: ", rw_vanilla)
    act_seq_vanilla_freq = Counter(act_seq_vanilla).most_common()
    for a, b in act_seq_vanilla_freq:
        actions_vanilla[a] = b
        print(ACTION_MEANING_SPI[int(a)], ":", b)

    return actions, actions_vanilla


# Plot a summary of the action frequencies given a seed, intervention and agent name
def plot_action(seed, intv, agent):
    fig, ax = plt.subplots(1, 1, figsize=(14, 5), facecolor="white")
    plt.rc("font", size=13)

    a1, a1_v = action_freq(seed, intv, agent)
    xlabels = [ACTION_MEANING_SPI[int(x)] for x in a1.keys()]
    xlabels_v = [ACTION_MEANING_SPI[int(x)] for x in a1_v.keys()]

    assert xlabels == xlabels_v

    ind = np.arange(len(a1))
    width = 0.35
    ax.bar(ind, a1.values(), width, align="center")
    ax.bar(ind + width, a1_v.values(), width, align="center")
    ax.legend(("w/ intv", "w/o intv"))
    plt.xticks(np.arange(len(a1)), xlabels)
    plt.ylim([0, 500])
    plt.title(
        "Agent: "
        + str(agent)
        + " Intervention: "
        + str(intv_name[intv])
        + " Seed: "
        + str(seed)
    )


# Summarize action frequencies by seed for any intervention number and agent type
def action_freq_seed(intv, agent_name):
    actions = defaultdict(list)
    actions_vanilla = defaultdict(list)
    for s in seed:
        dir_path = "storage/results/action_trajectories/{}/{}/{}/".format(
            intv, s, agent_name
        )
        action_file = dir_path + str(agent_name) + ".txt"
        with open(action_file, "r") as f:
            seq = f.readlines()

        dir_path_vanilla = "storage/results/action_trajectories/{}/{}/{}/".format(
            -1, s, agent_name
        )
        action_file_vanilla = dir_path_vanilla + str(agent_name) + ".txt"
        with open(action_file_vanilla, "r") as f:
            seq_vanilla = f.readlines()

        act_seq = [int(x) for x in seq[0].rstrip("\n")]
        act_seq_vanilla = [int(x) for x in seq_vanilla[0].rstrip("\n")]
        for a in range(0, 6):
            actions[a].append(act_seq.count(a))
            actions_vanilla[a].append(act_seq_vanilla.count(a))

    actions_seed = {}
    actions_vanilla_seed = {}
    for a, counts in actions.items():
        actions_seed[a] = statistics.mean(counts)

    for a, counts in actions_vanilla.items():
        actions_vanilla_seed[a] = statistics.mean(counts)
    return actions_seed, actions_vanilla_seed


# Plot action frequency summaries by seed for any intervention number
def plot_action_seed(intv, agent):
    fig, ax = plt.subplots(1, 1, figsize=(14, 5), facecolor="white")
    plt.rc("font", size=13)

    a1, a1_v = action_freq_seed(intv, agent)
    xlabels = [ACTION_MEANING_SPI[x] for x in a1.keys()]
    xlabels_v = [ACTION_MEANING_SPI[x] for x in a1_v.keys()]

    # Quick check if the orders of the dictionaries
    assert xlabels == xlabels_v

    ind = np.arange(len(a1))
    width = 0.35
    ax.bar(ind, a1.values(), width, align="center")
    ax.bar(ind + width, a1_v.values(), width, align="center")
    ax.legend(("w/ intv", "w/o intv"))
    plt.xticks(np.arange(len(a1)), xlabels)
    plt.ylim([0, 500])
    plt.title("Agent: " + str(agent) + " Intervention: " + str(intv_name[intv]))


# Summarize the actions frequencies by intervention number
def action_freq_intv(int_name, agent_name):
    a_s_list = []
    a_v_s_list = []
    for intv in intv_number[int_name]:
        a_s, a_v_s = action_freq_seed(intv, agent_name)
        a_s_list.append(a_s)
        a_v_s_list.append(a_v_s)

    sums = Counter()
    counters = Counter()
    for itemset in a_s_list:
        sums.update(itemset)
        counters.update(itemset.keys())
    actions_summary = {x: round(float(sums[x]) / counters[x], 2) for x in sums.keys()}

    sums = Counter()
    counters = Counter()
    for itemset in a_v_s_list:
        sums.update(itemset)
        counters.update(itemset.keys())
    actions_vanilla_summary = {
        x: round(float(sums[x]) / counters[x], 2) for x in sums.keys()
    }

    # print("Intervention: ", int_name)
    # print(actions_summary)
    # print(actions_vanilla_summary)
    # print("#############################")
    return actions_summary, actions_vanilla_summary


def plot_action_intv(int_name, agent):
    fig, ax = plt.subplots(1, 1, figsize=(14, 5), facecolor="white")
    plt.rc("font", size=13)

    a1, a1_v = action_freq_intv(int_name, agent)
    xlabels = [ACTION_MEANING_SPI[x] for x in a1.keys()]
    xlabels_v = [ACTION_MEANING_SPI[x] for x in a1_v.keys()]

    # Quick check if the orders of the dictionaries
    assert xlabels == xlabels_v

    ind = np.arange(len(a1))
    width = 0.35
    ax.bar(ind, a1.values(), width, align="center")
    ax.bar(ind + width, a1_v.values(), width, align="center")
    ax.legend(("w/ intv", "w/o intv"))
    plt.xticks(np.arange(len(a1)), xlabels)
    plt.ylim([0, 500])
    plt.title("Agent: " + str(agent))


if __name__ == "__main__":
    compare_trajectories(intv=66, seed=0, agent_name="ddt")
