from envs.wrappers.amidar.all_toybox_wrapper import (
    ToyboxEnvironment,
    customAmidarResetWrapper,
)
import os
import gym
import json

import numpy as np
from models.random import RandomAgent


from envs.wrappers.amidar.interventions.paths import (
    get_start_state_path,
    env_id,
)
from envs.wrappers.amidar.interventions.reset_wrapper import (
    AmidarResetWrapper,
)


def get_start_env(state_num, lives, use_trajectory_starts):
    if not os.path.isfile(get_start_state_path(state_num, False)):
        raise RuntimeError(
            "Start states have not been created yet. Please sample start states."
        )
    env = gym.make(env_id)
    env = AmidarResetWrapper(
        env,
        state_num,
        intv=-1,
        lives=lives,
        use_trajectory_starts=use_trajectory_starts,
    )
    return env


def sample_start_states(num_states, horizon):
    if os.path.isfile(get_start_state_path(num_states - 1, False)):
        print("Skipping start state sampling because they exist already.")
        return

    agt = RandomAgent(gym.make(env_id).action_space)
    for state_num in range(num_states):
        env = gym.make(env_id)
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


def sample_start_states_from_trajectory(agent, num_states):
    if os.path.isfile(get_start_state_path(num_states - 1, True)):
        print("Skipping start state sampling because they exist already.")
        return

    random_agent = RandomAgent(gym.make(env_id).action_space)

    env = ToyboxEnvironment(
        "AmidarToybox",
        device="cpu",
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

        env = gym.make(env_id)
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

    print(f"Created {num_states} start states from trajectory.")
