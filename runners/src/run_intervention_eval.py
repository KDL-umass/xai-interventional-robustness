import os
import argparse
import torch

from runners.src.action_evaluation import *

from envs.wrappers.start_states import (
    sample_start_states,
    sample_start_states_from_trajectory,
)
import envs.wrappers.space_invaders.interventions.interventions as si_interventions
import envs.wrappers.amidar.interventions.interventions as amidar_interventions
import envs.wrappers.breakout.interventions.interventions as breakout_interventions

# a2c_model_root = "/Users/kavery/Downloads/runs_a2c_total_10"
# dqn_model_root = "/Users/kavery/Downloads/runs_dqn_total_10"
# ddqn_model_root = "/Users/kavery/Downloads/runs_ddqn_total_10"
# rainbow_model_root = "/Users/kavery/Downloads/runs_rainbow_total_10"
# c51_model_root = "/Users/kavery/Downloads/runs_c51_total_10"

a2c_model_root = "/mnt/nfs/scratch1/ppruthi/runs_a2c_total_10"
a2c_supplementary = "/mnt/nfs/scratch1/kavery/a2c_76cd60f_2021-09-02_17:19:00_007989"
dqn_model_root = "/mnt/nfs/scratch1/kavery/runs_dqn_total_10"
ddqn_model_root = "/mnt/nfs/scratch1/kavery/runs_ddqn_total_10"
c51_model_root = "/mnt/nfs/scratch1/kavery/runs_c51_total_10"
rainbow_model_root = "/mnt/nfs/scratch1/kavery/runs_rainbow_total_10"

model_locations = {
    "a2c": [
        *[a2c_model_root + "/" + folder for folder in os.listdir(a2c_model_root)],
        a2c_supplementary,
    ],
    "dqn": [
        *[dqn_model_root + "/" + folder for folder in os.listdir(dqn_model_root)],
    ],
    "ddqn": [
        *[ddqn_model_root + "/" + folder for folder in os.listdir(ddqn_model_root)],
    ],
    "c51": [
        *[c51_model_root + "/" + folder for folder in os.listdir(c51_model_root)],
    ],
    "rainbow": [
        *[
            rainbow_model_root + "/" + folder
            for folder in os.listdir(rainbow_model_root)
        ],
    ],
}

agent_family_that_selects_max_action = ["a2c", "dqn", "ddqn", "rainbow", "c51"]


def load_agent(dir, device):
    path = dir + "/preset.pt"
    agt = torch.load(path, map_location=torch.device(device))
    agt = agt.test_agent()
    return agt


def get_intervention_data_dir(
    agent_family, num_agents, num_states_to_intervene_on, start_horizon, sample_js_div
):
    if sample_js_div:
        return f"storage/results/intervention_js_div/{agent_family}/{num_agents}_agents/{num_states_to_intervene_on}_states/t{start_horizon}_horizon"
    else:
        return f"storage/results/intervention_action_dists/{agent_family}/{num_agents}_agents/{num_states_to_intervene_on}_states/t{start_horizon}_horizon"


def get_trajectory_intervention_data_dir(
    agent_family, num_agents, num_states_to_intervene_on, sample_js_div
):
    if sample_js_div:
        return f"storage/results/intervention_js_div/{agent_family}/{num_agents}_agents/{num_states_to_intervene_on}_states/trajectory/"
    else:
        return f"storage/results/intervention_action_dists/{agent_family}/{num_agents}_agents/{num_states_to_intervene_on}_states/trajectory/"


def agent_setup(
    agent_family,
    use_trajectory_starts,
    num_states_to_intervene_on,
    start_horizon,
    sample_js_div,
    device,
):
    agents = [load_agent(dir, device) for dir in model_locations[agent_family]]

    if use_trajectory_starts:
        dir = get_trajectory_intervention_data_dir(
            agent_family, len(agents), num_states_to_intervene_on, sample_js_div
        )
    else:
        dir = get_intervention_data_dir(
            agent_family,
            len(agents),
            num_states_to_intervene_on,
            start_horizon,
            sample_js_div,
        )
    os.makedirs(dir, exist_ok=True)
    return agents, dir


def state_setup(
    agents,
    use_trajectory_starts,
    num_states_to_intervene_on,
    start_horizon,
    environment,
):
    if use_trajectory_starts:
        assert len(agents) == 11
        agent = agents.pop()  # first agent will be one sampled from
        sample_start_states_from_trajectory(
            agent, num_states_to_intervene_on, environment
        )
    else:
        sample_start_states(num_states_to_intervene_on, start_horizon, environment)

    if environment == "SpaceInvaders":
        num_interventions = si_interventions.create_intervention_states(
            num_states_to_intervene_on, use_trajectory_starts
        )
    elif environment == "Amidar":
        num_interventions = amidar_interventions.create_intervention_states(
            num_states_to_intervene_on, use_trajectory_starts
        )
    elif environment == "Breakout":
        num_interventions = breakout_interventions.create_intervention_states(
            num_states_to_intervene_on, use_trajectory_starts
        )
    else:
        raise ValueError(
            "Unknown environment supplied. Please use SpaceInvaders, Amidar, or Breakout."
        )

    return num_interventions


def evaluate_interventions(agent_family, environment, device):
    action_distribution_samples = 100
    num_states_to_intervene_on = 40
    dist_type = "analytic"

    start_horizon = 100  # sample from t=100
    use_trajectory_starts = True

    sample_js_div = True  # use new js divergence sampling
    js_div_samples = 100

    agents, dir = agent_setup(
        agent_family,
        use_trajectory_starts,
        num_states_to_intervene_on,
        start_horizon,
        sample_js_div,
        device,
    )

    num_interventions = state_setup(
        agents,
        use_trajectory_starts,
        num_states_to_intervene_on,
        start_horizon,
        environment,
    )

    num_samples = js_div_samples if sample_js_div else action_distribution_samples

    evaluate_distributions(
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
    )


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
            agent_family=agent_family,
            environment="SpaceInvaders",
            device=device,
        )

    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
