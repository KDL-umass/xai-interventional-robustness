import os
import numpy as np
import matplotlib.pyplot as plt

from runners.src.run_intervention_eval import model_locations, supported_environments


def plot_returns_100(runs_dir, fname, timesteps=-1):
    data = load_returns_100_data(runs_dir)
    lines = {}
    fig, axes = plt.subplots(1, len(data))
    if len(data) == 1:
        axes = [axes]
    for i, env in enumerate(sorted(data.keys())):
        ax = axes[i]
        subplot_returns_100(ax, env, data[env], lines, timesteps=timesteps)
    plt.title(f"{env} performance: {fname}")
    plt.xlabel("Training Frames")
    plt.ylabel("Return")
    fig.legend(
        list(lines.values()),
        list(lines.keys()),
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
    )
    os.makedirs("storage/plots/returns/", exist_ok=True)
    plt.savefig(f"storage/plots/returns/{fname}_returns.png", bbox_inches="tight")


def load_returns_100_data(runs_dir):
    data = {}

    def add_data(agent, env, file):
        if env not in data:
            data[env] = {}
        data[env][agent] = np.genfromtxt(file, delimiter=",").reshape((-1, 3))

    count = 1
    for agent_dir in os.listdir(runs_dir):
        agent = agent_dir.split("_")[0]
        agent_path = os.path.join(runs_dir, agent_dir)
        if os.path.isdir(agent_path):
            agent = agent + "_" + str(count)
            count = count + 1
            for env in os.listdir(agent_path):
                env_path = os.path.join(agent_path, env)
                if os.path.isdir(env_path):
                    returns100path = os.path.join(env_path, "returns100.csv")
                    if os.path.exists(returns100path):
                        add_data(agent, env, returns100path)

    return data


def subplot_returns_100(ax, env, data, lines, timesteps=-1):
    for agent in data:
        agent_data = data[agent]
        x = agent_data[:, 0]
        mean = agent_data[:, 1]
        std = agent_data[:, 2]

        if timesteps > 0:
            x[-1] = timesteps

        if agent in lines:
            ax.plot(x, mean, label=agent, color=lines[agent].get_color())
        else:
            (line,) = ax.plot(x, mean, label=agent)
            lines[agent] = line
        ax.fill_between(
            x, mean + std, mean - std, alpha=0.2, color=lines[agent].get_color()
        )
        ax.set_title(env)
        ax.set_xlabel("timesteps")
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 5))


def plot_family_performance(parent_runs_dir, env):
    data = get_family_performance(parent_runs_dir + "/" + env)
    lines = {}
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))

    for i, fam in enumerate(sorted(data.keys())):
        subplot_family_returns(axes, data[fam], fam)
    plt.xlabel("Training Frames")
    plt.ylabel("Return")
    plt.title(f"{env} performance")
    plt.legend(loc="upper left")
    os.makedirs("storage/plots/returns/", exist_ok=True)
    plt.savefig(f"storage/plots/returns/all_family_returns.png", bbox_inches="tight")


def get_family_performance(runs_parent_dir):
    final_data = {}
    for fam in model_locations:
        data = load_returns_100_data(runs_parent_dir + f"/{fam}")
        plist = []
        for key in data["SpaceInvadersToybox"]:
            plist.append(data["SpaceInvadersToybox"][key])

        shapes = map(lambda x: np.shape(x), plist)
        m = max(shapes)
        avg = np.zeros(m)

        avg = np.concatenate(plist)
        avg = np.sort(avg, axis=0)
        avg = np.sort(avg, axis=0)

        final_data[fam] = avg
    return final_data


def subplot_family_returns(ax, data, label, timesteps=-1):
    t = data[:, 0]
    mean = data[:, 1]
    std = data[:, 2]

    (line,) = ax.plot(t, mean, label=label)
    ax.fill_between(t, mean + std, mean - std, alpha=0.2, color=line.get_color())
    ax.set_title(label)
    ax.set_xlabel("timesteps")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 5))


def get_performance(environment="SpaceInvaders"):
    for agent_family in model_locations:
        arys = np.array([])
        csvs = np.array([dir for dir in model_locations[agent_family][environment]])
        csvs = [dir + "/" + environment + "Toybox/returns-test.csv" for dir in csvs]
        for csv in csvs:
            with open(csv) as filename:
                ary = np.loadtxt(filename, delimiter=",")
            arys = np.append(arys, ary, axis=0)
        arys = arys.reshape(-1, 3)
        avgs = np.average(arys, axis=0)
        print(agent_family)
        print(avgs[1])
        print(avgs[2])


if __name__ == "__main__":

    for fam in model_locations:
        for env in model_locations[fam]:
            plot_returns_100(f"storage/models/{env}/{fam}/", fam)

    for env in supported_environments:
        plot_family_performance("storage/models/", env)
