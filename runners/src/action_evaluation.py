import numpy as np
import torch

from analysis.src.js_divergence import js_divergence

from envs.wrappers.all_toybox_wrapper import (
    ToyboxEnvironment,
    customSpaceInvadersResetWrapper,
)

agent_family_that_selects_max_action = [
    "a2c",
    "dqn",
    "ddqn",
    "rainbow",
    "c51",
    "vsarsa",
]


def load_agent(dir, device):
    path = dir + "/preset.pt"
    agt = torch.load(path, map_location=torch.device(device))
    agt = agt.test_agent()
    return agt


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
        dist = [np.count_nonzero(actions == act) / samples for act in range(n)]
    else:
        raise ValueError("Dist unknown")
    return dist


def collect_action_distributions(
    agent_family, agents, envs, env_labels, samples, dist_type
):
    n = len(envs) * len(agents)
    dists = np.zeros((n, envs[0].action_space.n + 3))
    row = 0
    for e, env in enumerate(envs):
        for a, agt in enumerate(agents):
            dists[row, 0] = a
            dists[row, 1:3] = env_labels[e]
            dists[row, 3:] = policy_action_distribution(
                agent_family, agt, env, env.reset(), samples, dist_type
            )
            row += 1
            print(f"\r\rSampling {round(row / n * 100)}% complete", end="")
    print()
    return dists


def get_js_divergence(agent_family, agents, envs, env_labels):
    n = len(envs)
    result_table = np.zeros((n, 4))  # env_labels + js_divergence = 4 cols
    row = 0
    for e, env in enumerate(envs):
        result_table[row, 1:3] = env_labels[e]

        actions = np.zeros((len(agents), envs[0].action_space.n))
        for a, agt in enumerate(agents):
            actions[a, :] = policy_action_distribution(
                agent_family, agt, env, env.reset(), 1, "empirical"
            )

        result_table[row, 3] = js_divergence(actions)

        row += 1
    return result_table


def get_action_distribution_header(envs, sample_jsdiv):
    header = "agent,state,intv,"
    if sample_jsdiv:
        header += "jsdiv"

    else:
        for a in range(envs[0].action_space.n):
            header += "action_" + str(a) + ","
        header = header[:-1]  # remove last comma
        return header


def average_js_divergence(agent_family, agents, envs, env_labels, num_samples):
    if agent_family in agent_family_that_selects_max_action:
        # only need to run one iteration
        return get_js_divergence(agent_family, agents, envs, env_labels)

    dists = []
    for i in range(num_samples):
        dist = get_js_divergence(agent_family, agents, envs, env_labels)
        dists.append(dist)
        print(f"\r\rSampling {round(i / (num_samples-1) * 100)}% complete", end="")
    print()
    dists = np.mean(dists, axis=0)  # average over trials
    return dists


def evaluate_action_distributions(
    agent_family,
    agents,
    use_trajectory_starts,
    num_states_to_intervene_on,
    interventions,
    num_samples,
    sample_js_div,
    dist_type,
    environment,
    device,
    dir,
):
    envs = [
        ToyboxEnvironment(
            environment + "Toybox",
            device=device,
            custom_wrapper=customSpaceInvadersResetWrapper(
                state_num=state_num,
                intv=intv,
                lives=3,
                use_trajectory_starts=use_trajectory_starts,
            ),
        )
        for state_num in range(num_states_to_intervene_on)
        for intv in interventions
    ]

    env_labels = [
        (state_num, intv)
        for state_num in range(num_states_to_intervene_on)
        for intv in interventions
    ]

    if sample_js_div:
        dists = average_js_divergence(
            agent_family, agents, envs, env_labels, num_samples
        )
    else:
        dists = collect_action_distributions(
            agent_family,
            agents,
            envs,
            env_labels,
            num_samples,
            dist_type,
        )

    header = get_action_distribution_header(envs, sample_js_div)

    if len(interventions) == 1:
        np.savetxt(dir + "/vanilla.txt", dists, header=header)
    else:
        np.savetxt(
            dir + f"/{len(interventions)}_interventions.txt", dists, header=header
        )

    pass


def evaluate_distributions(
    agent_family,
    agents,
    use_trajectory_starts,
    num_states_to_intervene_on,
    num_interventions,
    num_samples,
    sample_js_div,
    dist_type,
    environment,
    device,
    dir,
):
    # vanilla
    evaluate_action_distributions(
        agent_family,
        agents,
        use_trajectory_starts,
        num_states_to_intervene_on,
        [-1],
        num_samples,
        sample_js_div,
        dist_type,
        environment,
        device,
        dir,
    )

    # interventions
    evaluate_action_distributions(
        agent_family,
        agents,
        use_trajectory_starts,
        num_states_to_intervene_on,
        list(range(num_interventions)),
        num_samples,
        sample_js_div,
        dist_type,
        environment,
        device,
        dir,
    )

    pass
