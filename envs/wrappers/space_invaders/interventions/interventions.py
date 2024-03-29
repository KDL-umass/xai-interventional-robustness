from models.random import RandomAgent
import gym
import json
import numpy as np

from envs.wrappers.paths import (
    get_intervention_dir,
    get_root_intervention_dir,
    space_invaders_env_id,
)
from envs.wrappers.space_invaders.interventions.reset_wrapper import (
    SpaceInvadersResetWrapper,
)

from envs.wrappers.start_states import (
    get_start_env,
    sample_start_states,
)


def write_intervention_json(state, state_num, count):
    environment = "SpaceInvaders"
    with open(
        f"{get_intervention_dir(state_num, environment)}/{count}.json",
        "w",
    ) as outfile:
        json.dump(state, outfile)


# ENEMY INTERVENTIONS


def get_drop_one_enemy(env, state_num, count):
    """Drop one enemy interventions."""
    num_enemies = len(env.toybox.state_to_json()["enemies"])

    for enum in range(num_enemies):
        # Get JSON state and modify it to get the interventions
        state = env.toybox.state_to_json()
        state["enemies"][enum]["alive"] = False
        write_intervention_json(state, state_num, count)
        count += 1

    return count


def get_drop_enemy_rowcol_interventions(env, state_num, count):
    """Drop row or column of enemies."""

    num_enemies = len(env.toybox.state_to_json()["enemies"])

    for rowcol in ["row", "col"]:
        for band in range(6):
            state = env.toybox.state_to_json()
            for i in range(num_enemies):
                if (rowcol == "row" and i // 6 == band) or (
                    rowcol == "col" and i % 6 == band
                ):
                    state["enemies"][i]["alive"] = False

            write_intervention_json(state, state_num, count)

            count = count + 1
    return count


# AGENT INTERVENTIONS #


def get_shift_agent_interventions(env, state_num, count):
    """Start the agent in different positions."""

    shift_sizes = list(range(5, 150, 5))

    for shift in shift_sizes:
        state = env.toybox.state_to_json()
        state["ship"]["x"] += shift

        write_intervention_json(state, state_num, count)
        count = count + 1
    return count


def get_ship_speed_interventions(env, state_num, count):
    """Set the agent's firing rate to be faster."""
    env = gym.make(space_invaders_env_id)

    speeds = list(range(1, 10))

    for speed in speeds:
        state = env.toybox.state_to_json()
        state["ship"]["speed"] = speed

        write_intervention_json(state, state_num, count)
        count = count + 1

    return count


# SHIELD INTERVENTTIONS #


def get_shift_shields_interventions(env, state_num, count):
    """Slightly shift the shields."""
    env = gym.make(space_invaders_env_id)

    shift_sizes = list(range(-25, 25, 5))

    for shift in shift_sizes:
        state = env.toybox.state_to_json()
        num_shields = len(state["shields"])
        for num in range(num_shields):
            state["shields"][num]["x"] += shift

        write_intervention_json(state, state_num, count)
        count = count + 1

    return count


def get_remove_shield_interventions(env, state_num, count):
    """Drop a shield."""
    num_shields = len(env.toybox.state_to_json()["shields"])

    for sh in range(num_shields):
        env = gym.make(space_invaders_env_id)
        state = env.toybox.state_to_json()
        del state["shield"][sh]

        write_intervention_json(state, state_num, count)
        count = count + 1

    return count


def add_shield_interventions(env, state_num, count):
    """Add a shield."""
    state = env.toybox.state_to_json()
    additional_shield = state["shields"][0].copy()
    state["shields"].append(additional_shield)
    state["shields"][-1]["x"] = 100

    write_intervention_json(state, state_num, count)
    return count + 1


def get_flip_shield_icons(env, state_num, count):
    """Flip the shield icons vertically."""

    state = env.toybox.state_to_json()
    num_shields = len(state["shields"])

    for snum in range(num_shields):
        # Get JSON state and modify it to get the interventions
        icon = np.array(state["shields"][snum]["data"])
        icon = np.flipud(icon)
        icon = icon.tolist()
        state["shields"][snum]["data"] = icon

    write_intervention_json(state, state_num, count)
    return count + 1


def create_intervention_states(num_states: int):
    """Create JSON states for all interventions."""
    dir = get_root_intervention_dir("SpaceInvaders")
    for state_num in range(num_states):
        # 0th state is the default start state of the game
        env = get_start_env(
            state_num,
            lives=3,
            environment="SpaceInvaders",
        )

        count = 0
        count = get_drop_one_enemy(env, state_num, count)
        print(f"Interventions 0-{count-1} drop one enemy.")
        prev = count
        count = get_shift_shields_interventions(env, state_num, count)
        print(f"Interventions {prev}-{count-1} shift shields.")
        prev = count
        count = get_shift_agent_interventions(env, state_num, count)
        print(f"Interventions {prev}-{count-1} shift agent starts.")
        prev = count
        count = get_drop_enemy_rowcol_interventions(env, state_num, count)
        print(f"Interventions {prev}-{count-1} drop row/col of enemies.")
        prev = count
        count = get_flip_shield_icons(env, state_num, count)
        print(f"Interventions {prev}-{count-1} flip shield icons vertically.")
        print(f"Created {count} intervention states for state {state_num} in `{dir}`.")
    return count


def get_single_intervened_environment(state_num, intervention_number, lives):
    env = gym.make(space_invaders_env_id)
    env = SpaceInvadersResetWrapper(
        env, state_num=state_num, intv=intervention_number, lives=lives
    )
    return env


def get_intervened_environments(state_num, want_feature_vec, lives):
    # make them if they aren't created
    count = create_intervention_states(state_num + 1)
    envlist = []
    for intv in range(count):
        env = get_single_intervened_environment(
            state_num, intv, want_feature_vec, lives
        )
        envlist.append(env)
    return envlist


def get_all_intervened_environments(num_states, want_feature_vec, lives):
    count = create_intervention_states(num_states)
    envlist = []
    for state_num in range(num_states):
        for intv in range(count):
            env = get_single_intervened_environment(
                state_num, intv, want_feature_vec, lives
            )
            envlist.append(env)
    return envlist


if __name__ == "__main__":
    num_states = 1
    agent = RandomAgent(gym.make(space_invaders_env_id).action_space)
    sample_start_states(agent, num_states, "SpaceInvaders", device="cpu")
    create_intervention_states(num_states)