from envs.wrappers.all_toybox_wrapper import (
    ToyboxEnvironment,
    customSpaceInvadersResetWrapper,
    customAmidarResetWrapper,
    customBreakoutResetWrapper,
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


def get_start_env(state_num, lives, use_trajectory_starts, environment="SpaceInvaders"):
    if not os.path.isfile(get_start_state_path(state_num, use_trajectory_starts)):
        with_suffix = "" if use_trajectory_starts else "out"
        raise RuntimeError(
            f"Start states up to {state_num} for {lives} lives with {with_suffix} trajectory have not been created yet. Please sample start states."
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
    else:
        env = gym.make(breakout_env_id)
        env = BreakoutResetWrapper(
            env,
            state_num,
            intv=-1,
            lives=lives,
            use_trajectory_starts=use_trajectory_starts,
        )

    return env


def sample_start_states(num_states, horizon, environment="SpaceInvaders"):
    if os.path.isfile(get_start_state_path(num_states - 1, False)):
        print("Skipping start state sampling because they exist already.")
        return

    if environment == "SpaceInvaders":
        agt = RandomAgent(gym.make(space_invaders_env_id).action_space)
    elif environment == "Amidar":
        agt = RandomAgent(gym.make(amidar_env_id).action_space)
    else:
        agt = RandomAgent(gym.make(breakout_env_id).action_space)

    for state_num in range(num_states):
        if environment == "SpaceInvaders":
            env = gym.make(space_invaders_env_id)
        elif environment == "Amidar":
            env = gym.make(amidar_env_id)
        else:
            env = gym.make(breakout_env_id)

        obs = env.reset()
        t = 0
        while t < horizon:
            t += 1
            if state_num == 0:  # 0th state will always be the default game start
                break

            obs, _, done, _ = env.step(agt.get_action(obs))

            if done:  # keep sampling until we get a state at that time step
                t = 0
                obs = env.reset()

        state = env.toybox.state_to_json()

        with open(get_start_state_path(state_num, False), "w") as f:
            json.dump(state, f)

    print(f"Created {num_states} start states.")


def sample_start_states_from_trajectory(agent, num_states, environment, device):
    # if os.path.isfile(get_start_state_path(num_states - 1, True)):
    #     print("Skipping start state sampling because they exist already.")
    #     return

    if environment == "SpaceInvaders":
        random_agent = RandomAgent(gym.make(space_invaders_env_id).action_space)
    elif environment == "Amidar":
        random_agent = RandomAgent(gym.make(amidar_env_id).action_space)
    else:
        random_agent = RandomAgent(gym.make(breakout_env_id).action_space)

    env = ToyboxEnvironment(
        environment + "Toybox",
        device=device,
    )

    obs = env.reset()
    action, probs = agent.act(obs)

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
            raise ValueError(
                f"Unknown environment: {environment}. Select SpaceInvaders, Amidar, or Breakout."
            )

        obs = env.reset()
        env.toybox.write_state_json(state)
        for _ in range(5):
            obs, _, done, _ = env.step(random_agent.get_action(obs))

            if (
                done
            ):  # keep stepping until you get somewhere that isn't the end of the game
                obs = env.reset()
                done = False
                env.toybox.write_state_json(state)

        # write out sampled state
        state = env.toybox.state_to_json()

        with open(get_start_state_path(state_num, True), "w") as f:
            json.dump(state, f)

    # 0th state is always standard start

    obs = env.reset()
    state = env.toybox.state_to_json()
    with open(get_start_state_path(0, True), "w") as f:
        json.dump(state, f)

    print(
        f"Created {num_states} start states from trajectory near {get_start_state_path(0, True)}."
    )
