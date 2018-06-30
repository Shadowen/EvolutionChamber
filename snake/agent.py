from typing import List

import gym
import numpy as np

import genetic
from genetic import Genome
from numpy_util import sigmoid, cat_ones
from snake.direction import Direction
from . import Game


class Agent(genetic.Agent):
    def __init__(self, *, env: Game, hidden_nodes: List[int]):
        super().__init__(env=env)
        self.hidden_nodes: List[int] = hidden_nodes
        self.genome = self._init_genome(env.observation_space, env.action_space)

    def _init_genome(self, observation_space: gym.Space, action_space: gym.Space) -> Genome:
        all_layer_sizes = [np.product(observation_space.shape)] + self.hidden_nodes + [action_space.n]
        weights = [np.random.uniform(-1, 1, size=[i + 1, o]) for (i, o) in
                   zip(all_layer_sizes[:-1], all_layer_sizes[1:])]
        return Genome(weights)

    def get_action(self, ob):
        ob_reshaped = ob.reshape([1, np.product(ob.shape)])
        h = ob_reshaped
        for i in range(len(self.genome.values) - 1):
            h = sigmoid(cat_ones(h).dot(self.genome.values[i]))
        action_logits = cat_ones(h).dot(self.genome.values[-1])
        return list(Direction)[np.argmax(action_logits)]
