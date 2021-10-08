from models.random import RandomAgent
import os
import gym
import random, json
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from toybox import Toybox, Input
from toybox.interventions.breakout import BreakoutIntervention

from envs.wrappers.paths import (
    get_intervention_dir,
    breakout_env_id,
)
from envs.wrappers.breakout.interventions.reset_wrapper import (
    BreakoutResetWrapper,
)

from envs.wrappers.start_states import (
    get_start_env,
    sample_start_states,
    sample_start_states_from_trajectory,
)


def write_intervention_json(state, state_num, count, use_trajectory_starts):
    with open(
        f"{get_intervention_dir(state_num, use_trajectory_starts)}/{count}.json",
        "w",
    ) as outfile:
        json.dump(state, outfile)


# BRICK INTERVENTIONS


def get_add_brick_row_interventions(env, state_num, count, use_trajectory_starts):
    """Add row of blocks."""
    count += 1
    return count


def get_drop_brick_rowcol_interventions(env, state_num, count, use_trajectory_starts):
    """Drop row or column of blocks."""
    count = count + 1
    return count


# PADDLE INTERVENTIONS #


def get_shift_agent_interventions(env, state_num, count, use_trajectory_starts):
    """Start the agent in different positions."""
    count = count + 1
    return count


def get_agent_speed_interventions(env, state_num, count, use_trajectory_starts):
    """Set the agent's speed."""
    count = count + 1

    return count


def get_agent_width_interventions(env, state_num, count, use_trajectory_starts):
    """Change the width of the paddle"""
    count = count + 1
    return count


# BALL INTERVENTTIONS #


def get_shift_ball_interventions(env, state_num, count, use_trajectory_starts):
    """Start the ball in different positions."""
    count = count + 1
    return count


def get_ball_speed_interventions(env, state_num, count, use_trajectory_starts):
    """Drop a shield."""
    count = count + 1
    return count


def create_intervention_states(num_states: int, use_trajectory_starts: bool):
    """Create JSON states for all interventions."""
    dir = "storage/states/interventions/breakout"
    if use_trajectory_starts:
        dir = "storage/states/trajectory_interventions/breakout"
    os.makedirs(dir, exist_ok=True)

    path = dir + f"/{num_states-1}"
    if os.path.isdir(path):
        count = len(os.listdir(path))
        print(
            f"Skipping already created {count} interventions for {num_states} states."
        )
        return count

    for state_num in range(
        num_states
    ):  # 0th state is the default start state of the game
        env = get_start_env(
            state_num, lives=3, use_trajectory_starts=use_trajectory_starts, environment="Breakout"
        )

        count = 0
        # count = get_drop_one_enemy(env, state_num, count, use_trajectory_starts)
        # count = get_shift_shields_interventions(
        #     env, state_num, count, use_trajectory_starts
        # )
        # count = get_shift_agent_interventions(
        #     env, state_num, count, use_trajectory_starts
        # )
        # count = get_drop_enemy_rowcol_interventions(
        #     env, state_num, count, use_trajectory_starts
        # )
        # count = get_flip_shield_icons(env, state_num, count, use_trajectory_starts)
        
        print(f"Created {count} intervention states for state {state_num} in `{dir}`.")
    return count


def get_single_intervened_environment(
    state_num, intervention_number, want_feature_vec, lives
):
    env = gym.make(breakout_env_id)
    if want_feature_vec:
        env = BreakoutFeatureVecWrapper(env)
    env = BreakoutResetWrapper(
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
    sample_start_states(num_states, 100, "Breakout")
    # create_intervention_states(num_states, False)

    # num_states = 26
    # agent = RandomAgent(gym.make(breakout_env_id).action_space)
    # sample_start_states_from_trajectory(agent, num_states)
    # create_intervention_states(num_states, True)
