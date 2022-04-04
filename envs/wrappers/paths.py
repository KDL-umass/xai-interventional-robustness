import os

space_invaders_env_id = "SpaceInvadersToyboxNoFrameskip-v4"
amidar_env_id = "AmidarToyboxNoFrameskip-v4"
breakout_env_id = "BreakoutToyboxNoFrameskip-v4"


def get_root_intervention_dir(environment):
    prefix = "trajectory_"
    os.makedirs(f"storage/states/{prefix}interventions/{environment}", exist_ok=True)
    return f"storage/states/{prefix}interventions/{environment}"


def get_intervention_dir(state_num, environment):
    prefix = "trajectory_"
    os.makedirs(
        f"storage/states/{prefix}interventions/{environment}/{state_num}", exist_ok=True
    )
    return f"storage/states/{prefix}interventions/{environment}/{state_num}"


def get_start_state_path(state_num, environment):
    prefix = "trajectory_"
    os.makedirs(f"storage/states/{prefix}starts/{environment}", exist_ok=True)
    return f"storage/states/{prefix}starts/{environment}/{state_num}.json"


def get_num_interventions(environment):
    if environment == "SpaceInvaders":
        return 88
    if environment == "Amidar":
        return 69
    if environment == "Breakout":
        return 38  # 42
