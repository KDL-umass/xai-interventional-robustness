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
    get_root_intervention_dir,
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
    environment = "Breakout"
    with open(
        f"{get_intervention_dir(state_num, use_trajectory_starts, environment)}/{count}.json",
        "w",
    ) as outfile:
        json.dump(state, outfile)


# BRICK INTERVENTIONS


def get_drop_brick_col_interventions(env, state_num, count, use_trajectory_starts):
    """Drop column of blocks."""
    state = env.toybox.state_to_json()
    num_cols = 18

    for col in range(num_cols):
        state = Toybox("breakout").state_to_json()

        for i in range(len(state["bricks"])):
            if col == state["bricks"][i]["col"]:
                # print(row)
                # print(state["bricks"][i]["row"])
                state["bricks"][i]["alive"] = False

        write_intervention_json(state, state_num, count, use_trajectory_starts)
        count = count + 1

        # env.toybox.write_state_json(state)
        # dir_path = "/Users/kavery/workspace/xai-interventional-robustness/storage/states/interventions/Breakout/images/"
        # env.toybox.save_frame_image(dir_path + "{}.png".format(count))

    return count


def get_drop_brick_row_interventions(env, state_num, count, use_trajectory_starts):
    """Drop row or column of blocks."""
    state = env.toybox.state_to_json()
    num_rows = 6

    for row in range(num_rows):
        state = Toybox("breakout").state_to_json()

        for i in range(len(state["bricks"])):
            if row == state["bricks"][i]["row"]:
                # print(row)
                # print(state["bricks"][i]["row"])
                state["bricks"][i]["alive"] = False

        write_intervention_json(state, state_num, count, use_trajectory_starts)

        count = count + 1

        # env.toybox.write_state_json(state)
        # dir_path = "/Users/kavery/workspace/xai-interventional-robustness/storage/states/interventions/Breakout/images/"
        # env.toybox.save_frame_image(dir_path + "{}.png".format(count))

    return count


# PADDLE INTERVENTIONS #


def get_shift_paddle_interventions(env, state_num, count, use_trajectory_starts):
    """Start the agent in different positions."""
    state = env.toybox.state_to_json()
    xs = [30.0, 90.0, 160.0, 210.0]
    for x in xs:
        state = Toybox("breakout").state_to_json()
        state["paddle"]["position"]["x"] = x
        write_intervention_json(state, state_num, count, use_trajectory_starts)
        count = count + 1

        # env.toybox.write_state_json(state)
        # dir_path = "/Users/kavery/workspace/xai-interventional-robustness/storage/states/interventions/Breakout/images/"
        # env.toybox.save_frame_image(dir_path + "{}.png".format(count))

    return count


def get_paddle_speed_interventions(env, state_num, count, use_trajectory_starts):
    """Set the agent's speed."""
    state = env.toybox.state_to_json()
    speeds = [1.0, 2.0, 3.0, 5.0, 6.0, 7.0]
    for speed in speeds:
        state = Toybox("breakout").state_to_json()
        state["paddle_speed"] = speed
        write_intervention_json(state, state_num, count, use_trajectory_starts)
        count = count + 1

        # env.toybox.write_state_json(state)
        # dir_path = "/Users/kavery/workspace/xai-interventional-robustness/storage/states/interventions/Breakout/images/"
        # env.toybox.save_frame_image(dir_path + "{}.png".format(count))

    return count


def get_paddle_width_interventions(env, state_num, count, use_trajectory_starts):
    """Change the width of the paddle"""
    state = env.toybox.state_to_json()
    widths = [6.0, 12.0, 36.0, 48.0]
    for width in widths:
        state = Toybox("breakout").state_to_json()
        state["paddle_width"] = width
        write_intervention_json(state, state_num, count, use_trajectory_starts)
        count = count + 1

        # env.toybox.write_state_json(state)
        # dir_path = "/Users/kavery/workspace/xai-interventional-robustness/storage/states/interventions/Breakout/images/"
        # env.toybox.save_frame_image(dir_path + "{}.png".format(count))

    return count


# BALL INTERVENTTIONS #


def get_ball_radius_interventions(env, state_num, count, use_trajectory_starts):
    """Start the ball in different positions."""
    state = env.toybox.state_to_json()
    radii = [0.5, 1.0, 3.0, 4.0]
    for radius in radii:
        state = Toybox("breakout").state_to_json()
        state["ball_radius"] = radius
        write_intervention_json(state, state_num, count, use_trajectory_starts)
        count = count + 1

        # env.toybox.write_state_json(state)
        # dir_path = "/Users/kavery/workspace/xai-interventional-robustness/storage/states/interventions/Breakout/images/"
        # env.toybox.save_frame_image(dir_path + "{}.png".format(count))

    return count


def create_intervention_states(num_states: int, use_trajectory_starts: bool):
    """Create JSON states for all interventions."""
    dir = get_root_intervention_dir(use_trajectory_starts, "Breakout")

    for state_num in range(
        num_states
    ):  # 0th state is the default start state of the game
        env = get_start_env(
            state_num,
            lives=3,
            use_trajectory_starts=use_trajectory_starts,
            environment="Breakout",
        )

        count = 0
        count = get_paddle_width_interventions(
            env, state_num, count, use_trajectory_starts
        )
        count = get_paddle_speed_interventions(
            env, state_num, count, use_trajectory_starts
        )
        count = get_shift_paddle_interventions(
            env, state_num, count, use_trajectory_starts
        )
        count = get_ball_radius_interventions(
            env, state_num, count, use_trajectory_starts
        )
        count = get_drop_brick_row_interventions(
            env, state_num, count, use_trajectory_starts
        )
        count = get_drop_brick_col_interventions(
            env, state_num, count, use_trajectory_starts
        )

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
    num_states = 2
    sample_start_states(num_states, 100, "Breakout")
    create_intervention_states(num_states, False)

    num_states = 2
    agent = RandomAgent(gym.make(breakout_env_id).action_space)
    sample_start_states_from_trajectory(agent, num_states, "Breakout", device="cpu")
    create_intervention_states(num_states, True)
