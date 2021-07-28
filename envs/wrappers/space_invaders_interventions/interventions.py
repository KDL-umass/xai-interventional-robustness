import os
import gym
import random, json
import numpy as np
import matplotlib.pyplot as plt

from toybox import Toybox, Input
from toybox.interventions.space_invaders import SpaceInvadersIntervention

from envs.wrappers.space_invaders_interventions.reset_wrapper import (
    SpaceInvadersResetWrapper,
)
from envs.wrappers.space_invaders_features.feature_vec_wrapper import (
    SpaceInvadersFeatureVecWrapper,
)


def write_intervention_json(state, count):
    with open(
        "storage/states/interventions/intervened_state_" + str(count) + ".json", "w",
    ) as outfile:
        json.dump(state, outfile)


# ENEMY INTERVENTIONS


def get_drop_one_enemy_json(count):
    """Drop one enemy interventions."""

    # Create environment
    env_id = "SpaceInvadersToyboxNoFrameskip-v4"
    env = gym.make(env_id)
    num_enemies = len(env.toybox.state_to_json()["enemies"])

    for enum in range(num_enemies):
        # Get JSON state and modify it to get the interventions
        state = env.toybox.state_to_json()
        state["enemies"][enum]["alive"] = False
        write_intervention_json(state, count)
        count += 1

    return count


def get_drop_enemy_rowcol_interventions_json(count):
    """Drop row or column of enemies."""
    env_id = "SpaceInvadersToyboxNoFrameskip-v4"

    env = gym.make(env_id)
    num_enemies = len(env.toybox.state_to_json()["enemies"])

    envlist = []

    for rowcol in ["row", "col"]:
        for band in range(6):
            state = env.toybox.state_to_json()
            for i in range(num_enemies):
                if (rowcol == "row" and i // 6 == band) or (
                    rowcol == "col" and i % 6 == band
                ):
                    state["enemies"][i]["alive"] = False

            write_intervention_json(state, count)

            count = count + 1
    return count


# AGENT INTERVENTIONS #


def get_shift_agent_interventions_json(count):
    """Start the agent in different positions."""
    env_id = "SpaceInvadersToyboxNoFrameskip-v4"

    shift_sizes = list(range(5, 150, 5))

    env = gym.make(env_id)
    for shift in shift_sizes:
        state = env.toybox.state_to_json()
        state["ship"]["x"] += shift

        write_intervention_json(state, count)
        count = count + 1
    return count


def get_ship_speed_interventions_json(count):
    """Set the agent's firing rate to be faster."""
    env_id = "SpaceInvadersToyboxNoFrameskip-v4"
    env = gym.make(env_id)

    speeds = list(range(1, 10))

    for speed in speeds:
        state = env.toybox.state_to_json()
        state["ship"]["speed"] = speed

        write_intervention_json(state, count)
        count = count + 1

    return count


# SHIELD INTERVENTTIONS #


def get_shift_shields_interventions_json(count):
    """Slightly shift the shields."""
    env_id = "SpaceInvadersToyboxNoFrameskip-v4"
    env = gym.make(env_id)

    shift_sizes = list(range(-25, 25, 5))

    for shift in shift_sizes:
        state = env.toybox.state_to_json()
        num_shields = len(state["shields"])
        for num in range(num_shields):
            state["shields"][num]["x"] += shift

        write_intervention_json(state, count)
        count = count + 1

    return count


def get_remove_shield_interventions_json(count):
    """Drop a shield."""
    env_id = "SpaceInvadersToyboxNoFrameskip-v4"

    env = gym.make(env_id)
    num_shields = len(env.toybox.state_to_json()["shields"])

    for sh in range(num_shields):
        env = gym.make(env_id)
        state = env.toybox.state_to_json()
        del state["shield"][sh]

        write_intervention_json(state, count)
        count = count + 1

    return count


def add_shield_interventions_json(count):
    """Add a shield."""

    env_id = "SpaceInvadersToyboxNoFrameskip-v4"
    env = gym.make(env_id)

    # Create and modify state
    state = env.toybox.state_to_json()
    additional_shield = state["shields"][0].copy()
    state["shields"].append(additional_shield)
    state["shields"][-1]["x"] = 100

    write_intervention_json(state, count)
    return count + 1


def get_flip_shield_icons_json(count):
    """Flip the shield icons vertically."""

    # Create environment
    env_id = "SpaceInvadersToyboxNoFrameskip-v4"
    env = gym.make(env_id)
    state = env.toybox.state_to_json()
    num_shields = len(state["shields"])

    for snum in range(num_shields):
        # Get JSON state and modify it to get the interventions
        icon = np.array(state["shields"][snum]["data"])
        icon = np.flipud(icon)
        icon = icon.tolist()
        state["shields"][snum]["data"] = icon

    write_intervention_json(state, count)
    return count + 1


def create_json_states():
    """Create JSON states for all interventions."""
    try:
        os.rmdir("storage/states/interventions")
    except:
        print("", end="")
    os.makedirs("storage/states/interventions", exist_ok=True)
    count = 0
    count = get_drop_one_enemy_json(count)
    count = get_shift_shields_interventions_json(count)
    count = get_shift_agent_interventions_json(count)
    count = get_drop_enemy_rowcol_interventions_json(count)
    count = get_flip_shield_icons_json(count)
    print(f"Created {count} intervention states in `storage/states/interventions/`.")
    return count


def get_single_intervened_environment(intervention_number, want_feature_vec, lives):
    env_id = "SpaceInvadersToyboxNoFrameskip-v4"
    env = gym.make(env_id)
    if want_feature_vec:
        env = SpaceInvadersFeatureVecWrapper(env)
    env = SpaceInvadersResetWrapper(env, intv=intervention_number, lives=lives)
    return env


def get_intervened_environments(want_feature_vec, lives):
    count = create_json_states()
    envlist = []
    for i in range(count):
        env = get_single_intervened_environment(i, want_feature_vec, lives)
        envlist.append(env)
    return envlist


def get_env_list(want_feature_vec, vanilla, lives):
    """
    Get JSON intervention environments (if `vanilla` False), wrapped with feature vec if `want_feature_vec`.

    If `vanilla` is provided, `lives` will be ignored.
    """
    if not vanilla:
        envlist = get_intervened_environments(want_feature_vec, lives)
    else:
        env_id = "SpaceInvadersToyboxNoFrameskip-v4"
        env = gym.make(env_id)
        if want_feature_vec:
            env = SpaceInvadersFeatureVecWrapper(env)
        envlist = [env]
    return envlist


if __name__ == "__main__":
    create_json_states()
