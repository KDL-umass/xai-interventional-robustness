import os
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

kc_model_root = "/mnt/nfs/work1/jensen/kclary/all_models"


def load_agent(path, device):
    agt = torch.load(path, map_location=torch.device(device))
    agt = agt.test_agent()
    return agt


def policy_action_distribution(agt, env, obs, samples):
    n = env.action_space.n
    actions = np.zeros((samples,))
    for i in range(samples):
        actions[i] = agt.act(obs).numpy()
    dist = [np.count_nonzero(actions == act) / samples for act in range(n)]
    return dist


def collect_action_distributions(agents, envs, samples):
    n = len(envs) * len(agents)
    dists = np.zeros((n, envs[0].action_space.n))
    row = 0
    for env in envs:
        for agt in agents:
            dists[row, :] = policy_action_distribution(agt, env, env.reset(), samples)
            row += 1
            print(f"\r\r Sampling {round(row / n * 100)}% complete", end="")
    print()
    return dists


def get_intervention_data_dir(
    agent_family, num_agents, num_states_to_intervene_on, start_horizon
):
    return f"storage/results/intervention_action_dists/{agent_family}/{num_agents}_agents/{num_states_to_intervene_on}_states/t{start_horizon}_horizon"


def evaluate_interventions(agent_family):
    device = "cpu"
    # device = "cuda"

    action_distribution_samples = 250

    num_states_to_intervene_on = 25  # only default starting state
    start_horizon = 100  # sample from t=100
    sample_start_states(num_states_to_intervene_on, start_horizon)
    num_interventions = create_intervention_states(num_states_to_intervene_on)

    agents = [load_agent(kc_model_root + f"/{agent_family}/preset.pt", device)]
    dir = get_intervention_data_dir(
        agent_family, len(agents), num_states_to_intervene_on, start_horizon
    )
    os.makedirs(dir, exist_ok=True)

    # vanilla
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
    dists = collect_action_distributions(agents, envs, action_distribution_samples)

    np.savetxt(dir + "/vanilla.txt", dists)

    # interventions
    # envs = [
    #     ToyboxEnvironment(
    #         "SpaceInvadersToybox",
    #         device=device,
    #         custom_wrapper=customSpaceInvadersResetWrapper(
    #             state_num=state_num, intv=intv, lives=3
    #         ),
    #     )
    #     for state_num in range(num_states_to_intervene_on)
    #     for intv in range(num_interventions)
    # ]
    # dists = collect_action_distributions(agents, envs, action_distribution_samples)

    # np.savetxt(
    #     dir + f"/{num_interventions}_interventions.txt",
    #     dists,
    # )


if __name__ == "__main__":
    evaluate_interventions(agent_family="a2c")
