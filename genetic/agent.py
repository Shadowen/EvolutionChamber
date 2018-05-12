import itertools
from typing import Tuple, Callable

import gym
import numpy as np

from genetic.genome import Genome


class Agent:
    replica_number_counter = itertools.count()

    def __init__(self, *,
                 env: gym.Env,
                 build_agent: Callable[
                     [gym.Space, gym.Space], Tuple[Callable[[Genome, np.ndarray], np.ndarray], Genome]]):
        self.env = env

        self.get_action, self.genome = build_agent(self.env.observation_space, self.env.action_space)

    def run_iteration(self) -> float:
        obs = self.env.reset()
        done = False
        while not done:
            action = self.get_action(self.genome, obs)
            obs, reward, done, info = self.env.step(action)
        return reward, info
