def get_intervention_data_dir(
    agent_family,
    environment,
    num_agents,
    num_states_to_intervene_on,
    start_horizon,
    checkpoint,
    sample_js_div,
):
    if sample_js_div:
        return f"storage/results/intervention_js_div/{environment}/{agent_family}/{num_agents}_agents/{num_states_to_intervene_on}_states/t{start_horizon}_horizon/check_{checkpoint}"
    else:
        return f"storage/results/intervention_action_dists/{environment}/{agent_family}/{num_agents}_agents/{num_states_to_intervene_on}_states/t{start_horizon}_horizon /check_{checkpoint}"


def get_trajectory_intervention_data_dir(
    agent_family,
    environment,
    num_agents,
    num_states_to_intervene_on,
    checkpoint,
    sample_js_div,
):
    if sample_js_div:
        return f"storage/results/intervention_js_div/{environment}/{agent_family}/{num_agents}_agents/{num_states_to_intervene_on}_states/trajectory/check_{checkpoint}"
    else:
        return f"storage/results/intervention_action_dists/{environment}/{agent_family}/{num_agents}_agents/{num_states_to_intervene_on}_states/trajectory/check_{checkpoint}"
