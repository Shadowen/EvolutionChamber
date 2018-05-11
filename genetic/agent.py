import itertools
from typing import Tuple, Callable

import gym
import numpy as np

from genetic.genome import Genome
from snake import Direction


class Agent:
    replica_number_counter = itertools.count()

    def __init__(self, *,
                 env_constructor: Callable[[], gym.Env],
                 build_agent: Callable[
                     [gym.Space, gym.Space], Tuple[Callable[[Genome, np.ndarray], np.ndarray], Genome]]):
        self.game = env_constructor()

        self.get_action, self.genome = build_agent(self.game.observation_space, self.game.action_space)

    def run_iteration(self) -> float:
        obs = self.game.reset()
        done = False
        while not done:
            action = self.get_action(self.genome, obs)
            obs, reward, done, info = self.game.step(action)
        return reward
