from abc import abstractmethod
from typing import *
import gym, json

from toybox.envs.atari.space_invaders import SpaceInvadersEnv
import toybox.interventions.space_invaders as spi
import matplotlib.pyplot as plt


class SpaceInvadersResetWrapper(gym.Wrapper):
    """Resets space invaders environment at the start of every episode to an intervened state."""

    def __init__(self, tbenv: SpaceInvadersEnv, intv: int, lives: int):
        super().__init__(tbenv)
        self.env = tbenv
        self.toybox = (
            tbenv.toybox
        )  # Why does this fail when ToyboxBaseEnv has a toybox attribute?
        self.intv = intv  # Intervention number 0 - ?
        self.lives = lives

    def reset(self):
        super().reset()
        return self.on_episode_start()

    @abstractmethod
    def on_episode_start(self):
        """On the start of each episode, set the state to the JSON state according to the intervention."""
        # Get JSON state
        if self.intv >= 0:
            with open(
                "interventions/intervention_states/intervened_state_"
                + str(self.intv)
                + ".json"
            ) as f:
                iv_state = json.load(f)
            iv_state["lives"] = self.lives

        else:
            iv_state = self.toybox.to_state_json()
            iv_state["lives"] = self.lives

        # Set state to the reset state
        self.env.cached_state = iv_state
        self.toybox.write_state_json(iv_state)
        obs = self.env.toybox.get_state()
        return obs


def wrap_space_env_reset(env, intv=-1, lives=1):
    env = SpaceInvadersResetWrapper(env, intv, lives)
    return env.reset()
