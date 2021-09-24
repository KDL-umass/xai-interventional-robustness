import os
import argparse
import numpy as np
import torch

from envs.wrappers.start_states import (
    sample_start_states,
    sample_start_states_from_trajectory
) 
import envs.wrappers.space_invaders.interventions.interventions as si_interventions 
import envs.wrappers.amidar.interventions.interventions as amidar_interventions 
import envs.wrappers.breakout.interventions.interventions as breakout_interventions 

from envs.wrappers.all_toybox_wrapper import (
    ToyboxEnvironment,
    customSpaceInvadersResetWrapper,
    customAmidarResetWrapper, 
    customBreakoutResetWrapper
)

a2c_model_root = "/Users/kavery/Downloads/runs_a2c_total_10"
dqn_model_root = "/Users/kavery/Downloads/runs_dqn_total_10"
ddqn_model_root = "/Users/kavery/Downloads/runs_ddqn_total_10"
rainbow_model_root = "/Users/kavery/Downloads/runs_rainbow_total_10"
c51_model_root = "/Users/kavery/Downloads/runs_c51_total_10"

model_locations = {
    "a2c": [
        *[a2c_model_root + "/" + folder for folder in os.listdir(a2c_model_root)],
    ],
    "dqn": [
        *[dqn_model_root + "/" + folder for folder in os.listdir(dqn_model_root)],
    ],
    "ddqn": [
        *[ddqn_model_root + "/" + folder for folder in os.listdir(ddqn_model_root)],
    ],
    "rainbow": [
        *[rainbow_model_root + "/" + folder for folder in os.listdir(rainbow_model_root)],
    ],
    "c51": [
        *[c51_model_root + "/" + folder for folder in os.listdir(c51_model_root)],
    ],
}

agent_family_that_selects_max_action = ["a2c", "dqn", "ddqn", "rainbow", "c51"]


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
    else:
        n = env.action_space.n
        actions = np.zeros((samples,))
        for i in range(samples):
            act, p_dist = agt.act(obs)
            if type(act) == int:
                actions[i] = act
            else:
                actions[i] = act.detach().cpu().numpy()
        dist = [np.count_nonzero(actions == act) / samples for act in range(n)]
    return dist


def collect_action_distributions(
    agent_family, agents, envs, env_labels, samples, dist_type
):
    n = len(envs) * len(agents)
    dists = np.zeros((n, envs[0].action_space.n + 3))
    row = 0
    for a, agt in enumerate(agents):
        for e, env in enumerate(envs):
            dists[row, 0] = a
            dists[row, 1:3] = env_labels[e]
            dists[row, 3:] = policy_action_distribution(
                agent_family, agt, env, env.reset(), samples, dist_type
            )
            row += 1
            print(f"\r\rSampling {round(row / n * 100)}% complete", end="")
    print()
    return dists


def get_intervention_data_dir(
    agent_family, num_agents, num_states_to_intervene_on, start_horizon
):
    return f"storage/results/intervention_action_dists/{agent_family}/{num_agents}_agents/{num_states_to_intervene_on}_states/t{start_horizon}_horizon"


def get_trajectory_intervention_data_dir(
    agent_family, num_agents, num_states_to_intervene_on
):
    return f"storage/results/intervention_action_dists/{agent_family}/{num_agents}_agents/{num_states_to_intervene_on}_states/trajectory/"


def evaluate_interventions(agent_family, device, use_trajectory_starts, environment="SpaceInvaders"):
    action_distribution_samples = 100
    num_states_to_intervene_on = 30  # q in literature
    start_horizon = 100  # sample from t=100

    # agent setup
    agents = [load_agent(dir, device) for dir in model_locations[agent_family]]

    if use_trajectory_starts:
        dir = get_trajectory_intervention_data_dir(
            agent_family, len(agents), num_states_to_intervene_on
        )
    else:
        dir = get_intervention_data_dir(
            agent_family, len(agents), num_states_to_intervene_on, start_horizon
        )
    os.makedirs(dir, exist_ok=True)

    # state setup
    if use_trajectory_starts:
        assert len(agents) == 11
        agent = agents[0]  # first agent will be one sampled from
        sample_start_states_from_trajectory(agent, num_states_to_intervene_on, environment)
        if environment == "SpaceInvaders":
            num_interventions = si_interventions.create_intervention_states(num_states_to_intervene_on, True)
        elif environment == "Amidar":
            num_interventions = amidar_interventions.create_intervention_states(num_states_to_intervene_on, True)
        else:
            num_interventions = breakout_interventions.create_intervention_states(num_states_to_intervene_on, True)
    
    else:
        sample_start_states(num_states_to_intervene_on, start_horizon, environment)
        if environment == "SpaceInvaders":
            num_interventions = si_interventions.create_intervention_states(num_states_to_intervene_on, False)
        elif environment == "Amidar":
            num_interventions = amidar_interventions.create_intervention_states(num_states_to_intervene_on, False)
        else:
            num_interventions = breakout_interventions.create_intervention_states(num_states_to_intervene_on, False)

    # vanilla
    print("Vanilla:")
    envs = [
        ToyboxEnvironment(
            environment+"Toybox",
            device=device,
            custom_wrapper=customSpaceInvadersResetWrapper(
                state_num=state_num,
                intv=-1,
                lives=3,
                use_trajectory_starts=use_trajectory_starts,
            ),
        )
        for state_num in range(num_states_to_intervene_on)
    ]
    env_labels = [(i, -1) for i in range(num_states_to_intervene_on)]
    dists = collect_action_distributions(
        agent_family,
        agents,
        envs,
        env_labels,
        action_distribution_samples,
        dist_type="analytic",
    )

    # create header
    header = "agent,state,intv,"
    for a in range(envs[0].action_space.n):
        header += "action_" + str(a) + ","
    header = header[:-1]  # remove last comma

    np.savetxt(dir + "/vanilla.txt", dists, header=header)

    # interventions
    print("Interventions:")
    envs = [
        ToyboxEnvironment(
            environment+"Toybox",
            device=device,
            custom_wrapper=customSpaceInvadersResetWrapper(
                state_num=state_num,
                intv=intv,
                lives=3,
                use_trajectory_starts=use_trajectory_starts,
            ),
        )
        for state_num in range(num_states_to_intervene_on)
        for intv in range(num_interventions)
    ]
    env_labels = [
        (state_num, intv)
        for state_num in range(num_states_to_intervene_on)
        for intv in range(num_interventions)
    ]
    dists = collect_action_distributions(
        agent_family,
        agents,
        envs,
        env_labels,
        action_distribution_samples,
        dist_type="analytic",
    )

    np.savetxt(dir + f"/{num_interventions}_interventions.txt", dists, header=header)


if __name__ == "__main__":
    # get_performance()
    parser = argparse.ArgumentParser(
        description="Process experiment settings.", add_help=True
    )
    parser.add_argument(
        "-g", "--gpu", action="store_true", help="Add gpu flag to use CUDA"
    )
    args = parser.parse_args()
    if args.gpu:
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}.")

    for agent_family in model_locations:
        print(f"Evaluating agent family: {agent_family}")
        evaluate_interventions(
            agent_family=agent_family, device=device, use_trajectory_starts=True
        )

    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
