from models.random import RandomAgent
import os
import gym
import random, json
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from toybox import Toybox, Input
from toybox.interventions.amidar import AmidarIntervention, Amidar, Tile

from envs.wrappers.paths import (
    get_intervention_dir,
    amidar_env_id,
    get_root_intervention_dir,
)
from envs.wrappers.amidar.interventions.reset_wrapper import (
    AmidarResetWrapper,
)

from envs.wrappers.start_states import (
    get_start_env,
    sample_start_states,
)


def write_intervention_json(state, state_num, count):
    environment = "Amidar"
    with open(
        f"{get_intervention_dir(state_num, environment)}/{count}.json",
        "w",
    ) as outfile:
        json.dump(state, outfile)


# ENEMY INTERVENTIONS


def get_drop_one_enemy(env, state_num, count):
    """Drop one enemy interventions."""
    state = env.toybox.state_to_json()
    num_enemies = len(state["enemies"])

    for enemy in range(num_enemies):
        state = Toybox("amidar").state_to_json()
        # state["enemies"][enemy].remove(enemy)
        state["enemies"][enemy]["caught"] = True
        write_intervention_json(state, state_num, count)
        count += 1
    return count


def get_shift_enemy_interventions(env, state_num, count):
    """Start the enemy in different positions."""
    state = env.toybox.state_to_json()
    num_enemies = len(state["enemies"])
    x = [1500, 1984, 768, 1400]
    y = [0, 2000, 793, 2400]

    for i in range(len(x)):
        state = Toybox("amidar").state_to_json()
        # state["enemies"][enemy].remove(enemy)
        state["enemies"][i]["position"]["x"] = x[i]
        state["enemies"][i]["position"]["y"] = y[i]
        write_intervention_json(state, state_num, count)
        count += 1

        env.toybox.write_state_json(state)
        dir_path = "storage/states/interventions/Amidar/images"
        os.makedirs(dir_path, exist_ok=True)
        env.toybox.save_frame_image(dir_path + "/{}.png".format(count))
    return count


# AGENT INTERVENTIONS #
def get_shift_agent_interventions(env, state_num, count):
    """Start the agent in different positions."""
    state = env.toybox.state_to_json()
    num_enemies = len(state["enemies"])
    x = [1984, 0, 1700, 1984]
    y = [0, 1200, 2400, 1000]

    for i in range(len(x)):
        state = Toybox("amidar").state_to_json()
        state["player"]["position"]["x"] = x[i]
        state["player"]["position"]["y"] = y[i]
        write_intervention_json(state, state_num, count)
        count += 1

        env.toybox.write_state_json(state)
        dir_path = "storage/states/interventions/Amidar/images"
        os.makedirs(dir_path, exist_ok=True)
        env.toybox.save_frame_image(dir_path + "{}.png".format(count))
    return count


# TILE INTERVENTTIONS #


def get_remove_tile_interventions(env, state_num, count):
    """Drop a tile."""
    state = env.toybox.state_to_json()
    num_rows = len(state["board"]["tiles"])
    erase = [
        5,
        0,
        6,
        31,
        10,
        14,
        22,
        27,
        4,
        22,
        4,
        19,
        17,
        20,
        28,
        3,
        20,
        0,
        29,
        17,
        5,
        26,
        17,
        14,
        6,
        25,
        12,
        19,
        25,
        19,
        18,
    ]
    assert len(erase) == num_rows

    for row in range(num_rows):
        state = Toybox("amidar").state_to_json()
        index = erase[row]
        assert state["board"]["tiles"][row][index] == "Unpainted"
        state["board"]["tiles"][row][index] = "Empty"
        write_intervention_json(state, state_num, count)
        count += 1
    return count


def get_add_tile_interventions(env, state_num, count):
    """Add a tile."""
    state = env.toybox.state_to_json()
    num_rows = len(state["board"]["tiles"])
    erase = [
        0,
        5,
        12,
        30,
        13,
        20,
        0,
        26,
        6,
        17,
        29,
        23,
        0,
        22,
        19,
        6,
        16,
        27,
        0,
        12,
        7,
        22,
        20,
        10,
        0,
        13,
        8,
        23,
        30,
        3,
        18,
    ]
    assert len(erase) == num_rows

    for row in range(num_rows):
        if row % 6 != 0:
            state = Toybox("amidar").state_to_json()
            index = erase[row]
            # assert state["board"]["tiles"][row][index] == "Empty"
            state["board"]["tiles"][row][index] = "Unpainted"
            write_intervention_json(state, state_num, count)
            count += 1
    return count


def create_intervention_states(num_states: int):
    """Create JSON states for all interventions."""
    dir = get_root_intervention_dir("Amidar")

    for state_num in range(
        num_states
    ):  # 0th state is the default start state of the game
        env = get_start_env(
            state_num,
            lives=3,
            environment="Amidar",
        )

        count = 0
        prevcount = count
        count = get_remove_tile_interventions(env, state_num, count)
        print(f"{prevcount} to {count-1} remove_tile_interventions")
        prevcount = count
        count = get_add_tile_interventions(env, state_num, count)
        print(f"{prevcount} to {count-1} get_add_tile_interventions")
        prevcount = count
        count = get_drop_one_enemy(env, state_num, count)
        print(f"{prevcount} to {count-1} get_drop_one_enemy")
        prevcount = count
        count = get_shift_enemy_interventions(env, state_num, count)
        print(f"{prevcount} to {count-1} shift_enemy_interventions")
        prevcount = count
        count = get_shift_agent_interventions(env, state_num, count)
        print(f"{prevcount} to {count-1} shift_agent_interventions")
        prevcount = count

        print(f"Created {count} intervention states for state {state_num} in `{dir}`.")
    return count


def get_single_intervened_environment(state_num, intervention_number, lives):
    env = gym.make(amidar_env_id)
    env = AmidarResetWrapper(
        env, state_num=state_num, intv=intervention_number, lives=lives
    )
    return env


def get_intervened_environments(state_num, lives):
    # make them if they aren't created
    count = create_intervention_states(state_num + 1)
    envlist = []
    for intv in range(count):
        env = get_single_intervened_environment(state_num, intv, lives)
        envlist.append(env)
    return envlist


def get_all_intervened_environments(num_states, lives):
    count = create_intervention_states(num_states)
    envlist = []
    for state_num in range(num_states):
        for intv in range(count):
            env = get_single_intervened_environment(state_num, intv, lives)
            envlist.append(env)
    return envlist


if __name__ == "__main__":
    num_states = 1
    agent = RandomAgent(gym.make(amidar_env_id).action_space)
    sample_start_states(agent, num_states, environment="Amidar", device="cpu")
    create_intervention_states(num_states)
