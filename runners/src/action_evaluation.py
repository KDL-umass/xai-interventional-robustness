import numpy as np

from analysis.src.ce import norm_sym_cross_entropy

from envs.wrappers.all_toybox_wrapper import (
    ToyboxEnvironment,
    customAmidarResetWrapper,
    customBreakoutResetWrapper,
    customSpaceInvadersResetWrapper,
)

agent_family_that_selects_max_action = [
    "dqn",
    "ddqn",
    "rainbow",
    "c51",
    "vsarsa",
    "vqn",
]


def policy_action_distribution(
    agent_family, agt, env, obs, samples, dist_type="analytic"
):
    if dist_type == "analytic":
        act, p_dist = agt.act(obs)
        dist = p_dist.detach().cpu().numpy()

        if agent_family in agent_family_that_selects_max_action:
            idx = np.argmax(dist)
            dist = np.zeros(dist.shape)
            dist[idx] = 1.0
    elif dist_type == "empirical":
        n = env.action_space.n
        actions = np.zeros((samples,))
        for i in range(samples):
            act, p_dist = agt.act(obs)
            if type(act) == int:
                actions[i] = act
            else:
                actions[i] = act.detach().cpu().numpy()
        dist = [np.count_nonzero(actions == a) / samples for a in range(n)]
    else:
        raise ValueError("Dist unknown")
    return dist


def get_action_distribution_header(envs):
    header = "agent,state,intv,"
    for a in range(envs[0].action_space.n):
        header += "action_" + str(a) + ","
    header = header[:-1]  # remove last comma
    return header


def collect_action_distributions(agent_family, agents, envs, env_labels, samples):
    n = len(envs) * len(agents)
    dists = np.zeros((n, envs[0].action_space.n + 3))
    row = 0
    for e, env in enumerate(envs):
        for a, agt in enumerate(agents):
            dists[row, 0] = a
            dists[row, 1:3] = env_labels[e]
            dists[row, 3:] = policy_action_distribution(
                agent_family, agt, env, env.reset(), samples
            )
            row += 1
            print(f"\r\r\t\tCAD: Sampling {round(row / n * 100)}% complete", end="")
    print()
    return dists


def get_ce(agent_family, agents, envs, env_labels, dir):
    n = len(envs)
    m = len(agents)
    result_table = np.zeros((n, 4))  # env_labels + ce = 4 cols

    for e, env in enumerate(envs):
        result_table[e, 1:3] = env_labels[e]

        actions = np.zeros((len(agents), envs[0].action_space.n))
        intv_obs = env.reset()
        for a, agt in enumerate(agents):
            actions[a, :] = policy_action_distribution(
                agent_family, agt, env, intv_obs, 1, "empirical"
            )
            # print(
            #     f"\r\r\tGJD: Sampling {round((e*m + a) / (n*m-1) * 100)}% complete",
            #     end="",
            # )

        #### UNCOMMENT TO SAVE ACTION HISTOGRAMS ####
        #### COMMENTED OUT TO IMPROVE RUNTIME
        # print("Saving action histograms...", end="")
        # label_actions = np.argmax(actions, axis=1)
        # str_label_actions = [str(i) for i in label_actions]
        # with open(dir + f"/actions.csv", "a+") as file:
        #     output = (
        #         str(env_labels[e][0])
        #         + ", "
        #         + str(env_labels[e][1])
        #         + ", "
        #         + ",".join(str_label_actions)
        #         + "\n"
        #     )
        #     file.write(output)
        # print("done")
        #### END ACTION HISTOGRAMS

        print(actions)
        result_table[e, 3] = norm_sym_cross_entropy(actions)

    return result_table


def average_ce(agent_family, agents, envs, env_labels, num_samples, dir):
    if agent_family in agent_family_that_selects_max_action:
        # only need to run one iteration
        return get_ce(agent_family, agents, envs, env_labels, dir)

    dists = []
    for i in range(num_samples):
        dist = get_ce(agent_family, agents, envs, env_labels, dir)
        dists.append(dist)
        print(f"\nAJD: Sampling {round(i / (num_samples-1) * 100)}% complete")
    print()
    dists = np.mean(dists, axis=0)  # average over trials
    return dists


def evaluate_ce(
    agent_family,
    environment,
    agents,
    num_states_to_intervene_on,
    interventions,
    num_samples,
    device,
    dir,
):

    if environment == "SpaceInvaders":
        custom_wrapper = customSpaceInvadersResetWrapper
    elif environment == "Amidar":
        custom_wrapper = customAmidarResetWrapper
    elif environment == "Breakout":
        custom_wrapper = customBreakoutResetWrapper

    envs = [
        ToyboxEnvironment(
            environment + "Toybox",
            device=device,
            custom_wrapper=custom_wrapper(state_num, intv, 3),
        )
        for state_num in range(num_states_to_intervene_on)
        for intv in interventions
    ]

    env_labels = [
        (state_num, intv)
        for state_num in range(num_states_to_intervene_on)
        for intv in interventions
    ]

    dists = average_ce(
        agent_family, agents, envs, env_labels, num_samples, dir
    )

    header = "agent,state,intv,ce"

    if len(interventions) == 1:
        np.savetxt(dir + f"/vanilla.txt", dists, header=header)
    else:
        np.savetxt(
            dir + f"/{len(interventions)}_interventions.txt",
            dists,
            header=header,
        )

    pass


def evaluate_distributions(
    agent_family,
    environment,
    agents,
    num_states_to_intervene_on,
    num_interventions,
    num_samples,
    device,
    dir,
):
    print("vanilla")
    evaluate_ce(
        agent_family,
        environment,
        agents,
        num_states_to_intervene_on,
        [-1],
        num_samples,
        device,
        dir,
    )

    print("interventions")
    evaluate_ce(
        agent_family,
        environment,
        agents,
        num_states_to_intervene_on,
        list(range(num_interventions)),
        num_samples,
        device,
        dir,
    )
