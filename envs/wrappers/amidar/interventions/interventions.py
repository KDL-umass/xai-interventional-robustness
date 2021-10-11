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
)
from envs.wrappers.amidar.interventions.reset_wrapper import (
    AmidarResetWrapper,
)

from envs.wrappers.start_states import (
    get_start_env,
    sample_start_states,
    sample_start_states_from_trajectory,
)


def write_intervention_json(state, state_num, count, use_trajectory_starts):
    environment = "Amidar"
    with open(
        f"{get_intervention_dir(state_num, use_trajectory_starts, environment)}/{count}.json",
        "w",
    ) as outfile:
        json.dump(state, outfile)


# ENEMY INTERVENTIONS

def get_drop_one_enemy(env, state_num, count, use_trajectory_starts):
    """Drop one enemy interventions."""
    state = env.toybox.state_to_json()
    num_enemies = len(state["enemies"])

    for enemy in range(num_enemies):
        state = Toybox("amidar").state_to_json()
        # state["enemies"][enemy].remove(enemy)
        state["enemies"][enemy]["caught"] = True
        write_intervention_json(state, state_num, count, use_trajectory_starts)
        count += 1

        # env.toybox.write_state_json(state)
        # dir_path = "/Users/kavery/workspace/xai-interventional-robustness/storage/states/interventions/Amidar/images/"
        # env.toybox.save_frame_image(dir_path + "{}.png".format(count))
    return count

def get_shift_enemy_interventions(env, state_num, count, use_trajectory_starts):
    """Start the enemy in different positions."""
    state = env.toybox.state_to_json()
    num_enemies = len(state["enemies"])
    x = [1500,1984,768,1400]
    y = [0,2000,793,2400]

    for i in range(len(x)):
        state = Toybox("amidar").state_to_json()
        # state["enemies"][enemy].remove(enemy)
        state["enemies"][i]["position"]["x"] = x[i]
        state["enemies"][i]["position"]["y"] = y[i]
        write_intervention_json(state, state_num, count, use_trajectory_starts)
        count += 1

        env.toybox.write_state_json(state)
        dir_path = "/Users/kavery/workspace/xai-interventional-robustness/storage/states/interventions/Amidar/images/"
        env.toybox.save_frame_image(dir_path + "{}.png".format(count))
    return count

# AGENT INTERVENTIONS #
def get_shift_agent_interventions(env, state_num, count, use_trajectory_starts):
    """Start the enemy in different positions."""
    state = env.toybox.state_to_json()
    num_enemies = len(state["enemies"])
    x = [1984, 0, 1700, 1984]
    y = [0, 1200, 2400, 1000]

    for i in range(len(x)):
        state = Toybox("amidar").state_to_json()
        state["player"]["position"]["x"] = x[i]
        state["player"]["position"]["y"] = y[i]
        write_intervention_json(state, state_num, count, use_trajectory_starts)
        count += 1

        env.toybox.write_state_json(state)
        dir_path = "/Users/kavery/workspace/xai-interventional-robustness/storage/states/interventions/Amidar/images/"
        env.toybox.save_frame_image(dir_path + "{}.png".format(count))
    return count

# TILE INTERVENTTIONS #


def get_remove_tile_interventions(env, state_num, count, use_trajectory_starts):
    """Drop a tile."""
    state = env.toybox.state_to_json()
    num_rows = len(state["board"]["tiles"])
    erase = [5,0,6,31,10,14,
            22,27,4,22,4,19,
            17,20,28,3,20,0,
            29,17,5,26,17,14,
            6,25,12,19,25,19,18]
    assert len(erase) == num_rows

    for row in range(num_rows):
        state = Toybox("amidar").state_to_json()
        index = erase[row]
        assert state["board"]["tiles"][row][index] == "Unpainted"
        state["board"]["tiles"][row][index] = "Empty"
        write_intervention_json(state, state_num, count, use_trajectory_starts)
        count += 1
        
        # env.toybox.write_state_json(state)
        # dir_path = "/Users/kavery/workspace/xai-interventional-robustness/storage/states/interventions/Amidar/images/"
        # env.toybox.save_frame_image(dir_path + "{}.png".format(count))

    return count


def get_add_tile_interventions(env, state_num, count, use_trajectory_starts):
    """Add a tile."""
    state = env.toybox.state_to_json()
    num_rows = len(state["board"]["tiles"])
    erase = [0,5,12,30,13,20,
            0,26,6,17,29,23,
            0,22,19,6,16,27,
            0,12,7,22,20,10,
            0,13,8,23,30,3,18]
    assert len(erase) == num_rows

    for row in range(num_rows):
        if row%6 != 0:
            state = Toybox("amidar").state_to_json()
            index = erase[row]
            # assert state["board"]["tiles"][row][index] == "Empty"
            state["board"]["tiles"][row][index] = "Unpainted"
            write_intervention_json(state, state_num, count, use_trajectory_starts)
            count += 1
            
            # env.toybox.write_state_json(state)
            # dir_path = "/Users/kavery/workspace/xai-interventional-robustness/storage/states/interventions/Amidar/images/"
            # env.toybox.save_frame_image(dir_path + "{}.png".format(count))
    return count


def create_intervention_states(num_states: int, use_trajectory_starts: bool):
    """Create JSON states for all interventions."""
    dir = "storage/states/interventions/Amidar"
    if use_trajectory_starts:
        dir = "storage/states/trajectory_interventions/Amidar"
    os.makedirs(dir, exist_ok=True)

    path = dir + f"/{num_states-1}"
    # if os.path.isdir(path):
    #     count = len(os.listdir(path))
    #     print(
    #         f"Skipping already created {count} interventions for {num_states} states."
    #     )
    #     return count

    for state_num in range(
        num_states
    ):  # 0th state is the default start state of the game
        env = get_start_env(
            state_num, lives=3, use_trajectory_starts=use_trajectory_starts, environment="Amidar"
        )

        count = 0
        
        count = get_remove_tile_interventions(env, state_num, count, use_trajectory_starts)
        count = get_add_tile_interventions(env, state_num, count, use_trajectory_starts)
        count = get_drop_one_enemy(env, state_num, count, use_trajectory_starts)
        count = get_shift_enemy_interventions(env, state_num, count, use_trajectory_starts)
        count = get_shift_agent_interventions(env, state_num, count, use_trajectory_starts)

        print(f"Created {count} intervention states for state {state_num} in `{dir}`.")
    return count


def get_single_intervened_environment(
    state_num, intervention_number, lives
):
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
        env = get_single_intervened_environment(
            state_num, intv, lives
        )
        envlist.append(env)
    return envlist


def get_all_intervened_environments(num_states, lives):
    count = create_intervention_states(num_states)
    envlist = []
    for state_num in range(num_states):
        for intv in range(count):
            env = get_single_intervened_environment(
                state_num, intv, lives
            )
            envlist.append(env)
    return envlist


if __name__ == "__main__":
    # dir_path = "/Users/kavery/workspace/xai-interventional-robustness/storage/states/starts/Amidar/0.json"
    # Toybox("amidar").save_frame_image(dir_path + "{}.png".format(0))

    num_states = 1
    sample_start_states(num_states, 100, "Amidar")
    create_intervention_states(num_states, False)

    # num_states = 26
    # agent = RandomAgent(gym.make(amidar_env_id).action_space)
    # sample_start_states_from_trajectory(agent, num_states)
    # create_intervention_states(num_states, True)
