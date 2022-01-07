import os
import argparse
import torch
from pprint import pprint

from runners.src.action_evaluation import *

from envs.wrappers.start_states import (
    sample_start_states,
    sample_start_states_from_trajectory,
)
import envs.wrappers.space_invaders.interventions.interventions as si_interventions
import envs.wrappers.amidar.interventions.interventions as amidar_interventions
import envs.wrappers.breakout.interventions.interventions as breakout_interventions
from runners.src.result_paths import (
    get_intervention_data_dir,
    get_trajectory_intervention_data_dir,
)


def model_root(model, env):
    dir = f"storage/models/{env}/{model}"
    os.makedirs(dir, exist_ok=True)
    return dir


model_names = ["a2c", "dqn", "ddqn", "c51", "rainbow", "vsarsa", "vqn", "ppo"]
# model_names = ["vqn", "ppo"]
supported_environments = ["SpaceInvaders", "Amidar", "Breakout"]
# supported_environments = ["SpaceInvaders"]
# supported_environments = ["Amidar"]
# supported_environments = ["Breakout"]
# checkpoints = list(range(0, 100000, 10000))
# checkpoints.extend(list(range(100000, 1000000, 100000)))
# checkpoints.extend(list(range(1000000, 11000000, 1000000)))
checkpoints = [50000, 100000, 500000, 1000000, 5000000, 10000000]

# supported_environments = ["SpaceInvaders"]
# model_names = ["dqn", "vsarsa", "vqn", "rainbow"]

# supported_environments = ["Amidar"]
# model_names = ["dqn", "vsarsa", "vqn"]
# model_names = ["ppo"]

# supported_environments = ["Breakout"]
# model_names = ["vqn", "vsarsa", "dqn"]
# model_names = ["a2c", "ppo"]
# model_names = ["a2c", "dqn", "ddqn", "c51", "rainbow", "vsarsa", "vqn", "ppo"]
# model_names = ["dqn"]

model_locations = {
    fam: {
        env: [
            model_root(fam, env) + "/" + folder
            for folder in os.listdir(model_root(fam, env))
        ]
        for env in supported_environments
    }
    for fam in model_names
}

# pprint(model_locations)

# print(checkpoints)


def load_agent(dir, device, checkpoint=None):
    if checkpoint is None:
        path = dir + "/preset.pt"
    else:
        path = dir + f"/preset{checkpoint}.pt"

    print(f"Loading agent from: {path}")

    agt = torch.load(path, map_location=torch.device(device))
    agt = agt.test_agent()
    return agt


def agent_setup(
    agent_family,
    environment,
    checkpoint,
    use_trajectory_starts,
    num_states_to_intervene_on,
    start_horizon,
    sample_js_div,
    device,
):
    print(model_locations[agent_family][environment])
    agents = [
        load_agent(dir, device, checkpoint)
        for dir in model_locations[agent_family][environment]
    ]

    assert len(agents) >= 11, f"Num agents is {len(agents)}"
    if len(agents) > 11:
        [agents.pop() for _ in range(11, len(agents))]

    if use_trajectory_starts:
        dir = get_trajectory_intervention_data_dir(
            agent_family,
            environment,
            len(agents),
            num_states_to_intervene_on,
            checkpoint,
            sample_js_div,
        )
    else:
        dir = get_intervention_data_dir(
            agent_family,
            environment,
            len(agents),
            num_states_to_intervene_on,
            start_horizon,
            checkpoint,
            sample_js_div,
        )
    os.makedirs(dir, exist_ok=True)
    return agents, dir


def state_setup(
    agents,
    environment,
    use_trajectory_starts,
    num_states_to_intervene_on,
    start_horizon,
    device,
):
    if use_trajectory_starts:
        agent = agents.pop()  # first agent will be one sampled from
        sample_start_states_from_trajectory(
            agent, num_states_to_intervene_on, environment, device
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
    action_distribution_samples = 30
    num_states_to_intervene_on = 30

    dist_type = "analytic"

    start_horizon = 100  # sample from t=100
    use_trajectory_starts = True

    sample_js_div = True  # use new js divergence sampling
    js_div_samples = 30

    for checkpoint in checkpoints:
        print("Checkpoint", checkpoint)

        agents, dir = agent_setup(
            agent_family,
            environment,
            checkpoint,
            use_trajectory_starts,
            num_states_to_intervene_on,
            start_horizon,
            sample_js_div,
            device,
        )

        num_interventions = state_setup(
            agents,
            environment,
            use_trajectory_starts,
            num_states_to_intervene_on,
            start_horizon,
            device,
        )

        num_samples = js_div_samples if sample_js_div else action_distribution_samples

        evaluate_distributions(
            agent_family,
            environment,
            checkpoint,
            agents,
            use_trajectory_starts,
            num_states_to_intervene_on,
            num_interventions,
            num_samples,
            sample_js_div,
            dist_type,
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
        for environment in model_locations[agent_family]:
            print(f"Evaluating agent family: {agent_family}")
            evaluate_interventions(
                agent_family=agent_family,
                environment=environment,
                device=device,
            )

    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
