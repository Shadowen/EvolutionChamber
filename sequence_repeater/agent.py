import gym
import numpy as np

import genetic
from genetic import Genome
from numpy_util import cat_ones
from snake import Game


class Agent(genetic.Agent):
    def __init__(self, *, env: Game):
        super().__init__(env=env)
        self.genome = self._init_genome(env.observation_space, env.action_space)

    @staticmethod
    def _init_genome(observation_space: gym.Space, action_space: gym.Space) -> Genome:
        weights = [np.random.uniform(-1, 1, size=[np.product(observation_space.n) + 1, action_space.n]), ]
        return Genome(weights)

    def get_action(self, ob):
        ob_reshaped = ob.reshape([1, np.product(ob.shape)])
        action_logits = cat_ones(ob_reshaped).dot(self.genome.values[0])
        return np.argmax(action_logits)
