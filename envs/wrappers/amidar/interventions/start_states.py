import os
import gym
import json
from models.random import RandomAgent


from envs.wrappers.amidar.interventions.paths import (
    get_start_state_path,
    env_id,
)
from envs.wrappers.amidar.interventions.reset_wrapper import (
    AmidarResetWrapper,
)


def get_start_env(state_num, lives=3):
    if not os.path.isfile(get_start_state_path(state_num)):
        raise RuntimeError(
            "Start states have not been created yet. Please sample start states."
        )
    env = gym.make(env_id)
    env = AmidarResetWrapper(env, state_num, intv=-1, lives=lives)
    return env


def sample_start_states(num_states, horizon):
    if os.path.isfile(get_start_state_path(num_states - 1)):
        print("Skipping start state sampling because they exist already.")
        return

    agt = RandomAgent(gym.make(env_id).action_space)
    for state_num in range(0, num_states):
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

        with open(get_start_state_path(state_num), "w") as f:
            json.dump(state, f)

    print(f"Created {num_states} start states.")
