import numpy as np
from gym.spaces import Discrete


class ChangingNArmedBanditEnv:
    """
    An N-armed bandit that periodically rotates arm reward probabilities
    """

    def __init__(self, p, reward=1, penalty=-1, rotation_interval=None):
        """
        :param p: length-n iterable of arm reward-probabilities
        :param reward: scalar reward delivered on a succes
        :param penalty: scalar penalty delivered on a failure
        :param rotation_interval: number of pulls before probabilities rotate
        """
        self.p = np.array(p)
        self.reward = reward
        self.penalty = penalty
        self.rotation_interval = rotation_interval

        self.observation_space = Discrete(1)
        self.action_space = Discrete(len(p))

        self.step_count = 0

    def reset(self):
        self.p = np.roll(self.p, 1)
        return np.array([0])

    def step(self, action):
        self.step_count += 1
        if self.rotation_interval is not None and self.step_count % self.rotation_interval == 0:
            self.reset()

        reward = (np.random.random() <= self.p[action]).astype(int)
        reward = reward * (self.reward - self.penalty) + self.penalty
        return np.array([0]), reward, False, None
