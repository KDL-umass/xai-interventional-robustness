"""
A fork of Atari wrappers modified from autonomous-learning-laboratory AtariEnvironment:

Exposes the Toybox gym env to allow custom wrappers
"""

import gym
from all.environments import GymEnvironment, DuplicateEnvironment

from all.environments.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    FireResetEnv,
    WarpFrame,
    LifeLostEnv,
)
from .interventions.space_invaders_reset_wrapper import (
    SpaceInvadersResetWrapper,
)


class DefaultSpaceInvadersResetWrapper(SpaceInvadersResetWrapper):
    def __init__(self, env):
        super().__init__(env, intv=-1, lives=3)


class ToyboxEnvironment(GymEnvironment):
    def __init__(
        self,
        name,
        custom_wrapper: SpaceInvadersResetWrapper = DefaultSpaceInvadersResetWrapper,
        *args,
        **kwargs
    ):
        # need these for duplication
        self._args = args
        self._kwargs = kwargs
        # construct the environment
        # toybox gives 4-channel obs by default, but you can enforce 3-channel with kwargs
        env = gym.make(name + "NoFrameskip-v4", alpha=False, grayscale=False)
        self.toybox = env.unwrapped.toybox

        env = custom_wrapper(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env)
        env = LifeLostEnv(env)
        # initialize
        super().__init__(env, *args, **kwargs)
        self._name = name

    @property
    def name(self):
        return self._name

    def duplicate(self, n):
        return DuplicateEnvironment(
            [
                ToyboxEnvironment(self._name, *self._args, **self._kwargs)
                for _ in range(n)
            ]
        )
