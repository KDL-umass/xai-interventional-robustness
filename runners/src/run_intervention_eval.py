import os
import argparse
import numpy as np
import torch

from envs.wrappers.space_invaders.interventions.start_states import sample_start_states
from envs.wrappers.space_invaders.interventions.interventions import (
    create_intervention_states,
)
from envs.wrappers.space_invaders.all_toybox_wrapper import (
    ToyboxEnvironment,
    customSpaceInvadersResetWrapper,
)

# kc_model_root = "/mnt/nfs/work1/jensen/kclary/all_models"
pp_model_root = "/mnt/nfs/scratch1/ppruthi/runs_a2c_total_10"
ka_model_root = "/mnt/nfs/scratch1/kavery/runs_dqn_total_10"

model_locations = {
    "a2c": [
        # kc_model_root + "/a2c",
        *[pp_model_root + "/" + folder for folder in os.listdir(pp_model_root)],
    ],
    "dqn": [
        # kc_model_root + "/dqn",
        *[ka_model_root + "/" + folder for folder in os.listdir(ka_model_root)],
    ],
}


def load_agent(dir, device):
    path = dir + "/preset.pt"
    agt = torch.load(path, map_location=torch.device(device))
    agt = agt.test_agent()
    return agt


def policy_action_distribution(agt, env, obs, samples):
    n = env.action_space.n
    actions = np.zeros((samples,))
    for i in range(samples):
        act = agt.act(obs)
        if type(act) == int:
            actions[i] = act
        else:
            actions[i] = act.cpu().numpy()
    dist = [np.count_nonzero(actions == act) / samples for act in range(n)]
    return dist


def collect_action_distributions(agents, envs, env_labels, samples):
    n = len(envs) * len(agents)
    dists = np.zeros((n, envs[0].action_space.n + 3))
    row = 0
    for a, agt in enumerate(agents):
        for e, env in enumerate(envs):
            dists[row, 0] = a
            dists[row, 1:3] = env_labels[e]
            dists[row, 3:] = policy_action_distribution(agt, env, env.reset(), samples)
            row += 1
            print(f"\r\rSampling {round(row / n * 100)}% complete", end="")
    print()
    return dists


def get_intervention_data_dir(
    agent_family, num_agents, num_states_to_intervene_on, start_horizon
):
    return f"storage/results/intervention_action_dists/{agent_family}/{num_agents}_agents/{num_states_to_intervene_on}_states/t{start_horizon}_horizon"


def evaluate_interventions(agent_family, device):
    action_distribution_samples = 100
    num_states_to_intervene_on = 30  # q in literature
    start_horizon = 100  # sample from t=100

    sample_start_states(num_states_to_intervene_on, start_horizon)
    num_interventions = create_intervention_states(num_states_to_intervene_on)

    # setup
    agents = [load_agent(dir, device) for dir in model_locations[agent_family]]
    dir = get_intervention_data_dir(
        agent_family, len(agents), num_states_to_intervene_on, start_horizon
    )
    os.makedirs(dir, exist_ok=True)

    # vanilla
    print("Vanilla:")
    envs = [
        ToyboxEnvironment(
            "SpaceInvadersToybox",
            device=device,
            custom_wrapper=customSpaceInvadersResetWrapper(
                state_num=state_num, intv=-1, lives=3
            ),
        )
        for state_num in range(num_states_to_intervene_on)
    ]
    env_labels = [(i, -1) for i in range(num_states_to_intervene_on)]
    dists = collect_action_distributions(
        agents,
        envs,
        env_labels,
        action_distribution_samples,
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
            "SpaceInvadersToybox",
            device=device,
            custom_wrapper=customSpaceInvadersResetWrapper(
                state_num=state_num, intv=intv, lives=3
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
        agents, envs, env_labels, action_distribution_samples
    )

    np.savetxt(dir + f"/{num_interventions}_interventions.txt", dists, header=header)


if __name__ == "__main__":
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
        evaluate_interventions(agent_family=agent_family, device=device)
