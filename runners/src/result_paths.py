def get_trajectory_intervention_data_dir(
    agent_family,
    environment,
    num_agents,
    num_states_to_intervene_on,
    checkpoint,
):
    return f"storage/results/intervention_ce/{environment}/{agent_family}/{num_agents}_agents/{num_states_to_intervene_on}_states/trajectory/check_{checkpoint}"


def get_intervention_action_distribution_dir(
    agent_family,
    environment,
    num_agents,
    num_states_to_intervene_on,
    checkpoint,
):

    return f"storage/results/intervention_action_dists/{environment}/{agent_family}/{num_agents}_agents/{num_states_to_intervene_on}_states/trajectory/check_{checkpoint}"
