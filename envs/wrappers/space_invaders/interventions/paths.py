import os

env_id = "SpaceInvadersToyboxNoFrameskip-v4"


def get_intervention_dir(state_num):
    os.makedirs(f"storage/states/interventions/{state_num}", exist_ok=True)
    return f"storage/states/interventions/{state_num}"


def get_start_state_path(state_num):
    os.makedirs(f"storage/states/starts", exist_ok=True)
    return f"storage/states/starts/{state_num}.json"
