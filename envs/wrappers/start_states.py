from envs.wrappers.all_toybox_wrapper import (
    ToyboxEnvironment,
    customSpaceInvadersResetWrapper,
    customAmidarResetWrapper,
    customBreakoutResetWrapper,
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


def get_start_env(state_num, lives, use_trajectory_starts, environment):
    if not os.path.isfile(
        get_start_state_path(state_num, use_trajectory_starts, environment)
    ):
        print(get_start_state_path(state_num, use_trajectory_starts, environment))
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
            use_trajectory_starts=use_trajectory_starts,
        )
    elif environment == "Amidar":
        env = gym.make(amidar_env_id)
        env = AmidarResetWrapper(
            env,
            state_num,
            intv=-1,
            lives=lives,
            use_trajectory_starts=use_trajectory_starts,
        )
    elif environment == "Breakout":
        env = gym.make(breakout_env_id)
        env = BreakoutResetWrapper(
            env,
            state_num,
            intv=-1,
            lives=lives,
            use_trajectory_starts=use_trajectory_starts,
        )
    else:
        raise ValueError("Unknown environment specified.")

    return env


def sample_start_states(num_states, horizon, environment):

    if environment == "SpaceInvaders":
        agt = RandomAgent(gym.make(space_invaders_env_id).action_space)
    elif environment == "Amidar":
        agt = RandomAgent(gym.make(amidar_env_id).action_space)
    elif environment == "Breakout":
        agt = RandomAgent(gym.make(breakout_env_id).action_space)
    else:
        raise ValueError("Unknown environment specified.")

    for state_num in range(num_states):
        env = ToyboxEnvironment(environment + "Toybox", passThroughWrapper)

        obs = env.reset()
        t = 0
        while t < horizon:
            t += 1
            if state_num == 0:  # 0th state will always be the default game start
                break

            obs = env.step(agt.get_action(obs))
            done = obs["done"]

            if done:  # keep sampling until we get a state at that time step
                t = 0
                obs = env.reset()

        state = env.toybox.state_to_json()

        with open(get_start_state_path(state_num, False, environment), "w") as f:
            json.dump(state, f)

    print(
        f"Created {num_states} start states near {get_start_state_path(state_num, False, environment)}"
    )


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

        with open(get_start_state_path(state_num, True, environment), "w") as f:
            json.dump(state, f)

    # 0th state is always standard start

    obs = env.reset()
    state = env.toybox.state_to_json()
    with open(get_start_state_path(0, True, environment), "w") as f:
        json.dump(state, f)

    print(
        f"Created {num_states} start states from trajectory near {get_start_state_path(0, True, environment)}"
    )
