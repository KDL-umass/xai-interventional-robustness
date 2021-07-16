import gym


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, observation):
        return self.action_space.sample()

    def act(self, observation, reward, done):
        return self.action_space.sample()
