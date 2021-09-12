import os

env_id = "SpaceInvadersToyboxNoFrameskip-v4"


def get_intervention_dir(state_num, use_trajectory_starts):
    prefix = "trajectory_" if use_trajectory_starts else ""
    os.makedirs(f"storage/states/{prefix}interventions/{state_num}", exist_ok=True)
    return f"storage/states/{prefix}interventions/{state_num}"


def get_start_state_path(state_num, use_trajectory_starts):
    prefix = "trajectory_" if use_trajectory_starts else ""
    os.makedirs(f"storage/states/{prefix}starts", exist_ok=True)
    return f"storage/states/{prefix}starts/{state_num}.json"
