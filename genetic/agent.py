from abc import abstractmethod
from typing import Tuple, List

import gym

from genetic.genome import Genome


class Agent:
    def __init__(self, *, env: gym.Env):
        self.env = env
        self.genome: Genome = None

    @abstractmethod
    def get_action(self, obs):
        pass

    def run_iteration(self, *, render: bool = False) -> Tuple[float, List]:
        obs = self.env.reset()
        done = False
        while not done:
            action = self.get_action(obs)
            obs, reward, done, info = self.env.step(action)
            if render:
                self.env.render(mode='human')
        return reward, info
