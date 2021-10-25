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
from runners.src.result_paths import (
    get_intervention_data_dir,
    get_trajectory_intervention_data_dir,
)


def model_root(model, env):
    dir = f"storage/models/{env}/{model}"
    os.makedirs(dir, exist_ok=True)
    return dir


# model_names = ["a2c", "dqn", "ddqn", "c51", "rainbow", "vsarsa", "vqn", "ppo"]
model_names = ["a2c", "ppo"]
# model_names = ["dqn"]
# supported_environments = ["SpaceInvaders", "Amidar", "Breakout"]
supported_environments = ["SpaceInvaders", "Amidar"]
# supported_environments = ["Amidar"]

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

checkpoints = [100 * 10 ** i for i in range(6)]
print(checkpoints)


def load_agent(dir, device, checkpoint=None):
    if checkpoint is None:
        path = dir + "/preset.pt"
    else:
        path = dir + f"/preset{checkpoint}.pt"

    print(f"Loading agent from: {path}")

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
