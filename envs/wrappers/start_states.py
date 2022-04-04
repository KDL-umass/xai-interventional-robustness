from envs.wrappers.all_toybox_wrapper import (
    ToyboxEnvironment,
    passThroughWrapper,
)
import os
import gym
import json

import numpy as np
from models.random import RandomAgent


from envs.wrappers.paths import (
    get_start_state_path,
    space_invaders_env_id,
    amidar_env_id,
    breakout_env_id,
)
from envs.wrappers.space_invaders.interventions.reset_wrapper import (
    SpaceInvadersResetWrapper,
)
from envs.wrappers.amidar.interventions.reset_wrapper import AmidarResetWrapper
from envs.wrappers.breakout.interventions.reset_wrapper import BreakoutResetWrapper


def get_start_env(state_num, lives, environment):
    if not os.path.isfile(get_start_state_path(state_num, environment)):
        print(get_start_state_path(state_num, environment))
        raise RuntimeError(
            "Start states have not been created yet. Please sample start states."
        )
    if environment == "SpaceInvaders":
        env = gym.make(space_invaders_env_id)
        env = SpaceInvadersResetWrapper(
            env,
            state_num,
            intv=-1,
            lives=lives,
        )
    elif environment == "Amidar":
        env = gym.make(amidar_env_id)
        env = AmidarResetWrapper(
            env,
            state_num,
            intv=-1,
            lives=lives,
        )
    elif environment == "Breakout":
        env = gym.make(breakout_env_id)
        env = BreakoutResetWrapper(
            env,
            state_num,
            intv=-1,
            lives=lives,
        )
    else:
        raise ValueError("Unknown environment specified.")

    return env


def sample_start_states_from_trajectory(agent, num_states, environment, device):
    if environment == "SpaceInvaders":
        random_agent = RandomAgent(gym.make(space_invaders_env_id).action_space)
    elif environment == "Amidar":
        random_agent = RandomAgent(gym.make(amidar_env_id).action_space)
    elif environment == "Breakout":
        random_agent = RandomAgent(gym.make(breakout_env_id).action_space)
    else:
        raise ValueError("Unknown environment specified.")

    print(environment)
    env = ToyboxEnvironment(environment + "Toybox", passThroughWrapper, device=device)

    obs = env.reset()
    action, _ = agent.act(obs)

    trajectory = [env.toybox.state_to_json()]
    done = False
    while not done:
        obs = env.step(action)
        done = obs["done"]
        action, _ = agent.act(obs)

        state = env.toybox.state_to_json()

        trajectory.append(state)

    L = len(trajectory)

    for state_num in range(1, num_states):
        t = np.random.randint(0, L)

        state = trajectory[t]

        if environment == "SpaceInvaders":
            env = gym.make(space_invaders_env_id)
        elif environment == "Amidar":
            env = gym.make(amidar_env_id)
        elif environment == "Breakout":
            env = gym.make(breakout_env_id)
        else:
            raise ValueError("Unknown environment specified.")

        obs = env.reset()
        env.toybox.write_state_json(state)

        randomWalkLength = 0  # 5
        for _ in range(randomWalkLength):
            obs, _, done, _ = env.step(random_agent.get_action(obs))

            if (
                done
            ):  # keep stepping until you get somewhere that isn't the end of the game
                obs = env.reset()
                done = False
                env.toybox.write_state_json(state)

        # write out sampled state
        state = env.toybox.state_to_json()

        with open(get_start_state_path(state_num, environment), "w") as f:
            json.dump(state, f)

    # 0th state is always standard start

    obs = env.reset()
    state = env.toybox.state_to_json()
    with open(get_start_state_path(0, environment), "w") as f:
        json.dump(state, f)

    print(
        f"Created {num_states} start states from trajectory near {get_start_state_path(0, environment)}"
    )
