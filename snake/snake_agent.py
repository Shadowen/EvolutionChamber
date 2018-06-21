import gym
import numpy as np

from genetic import Agent, Genome
from numpy_util import sigmoid, cat_ones
from snake import Game, Direction


class SnakeAgent(Agent):
    def __init__(self, *, env: Game):
        super().__init__(env=env, genome=self._init_genome(env.observation_space, env.action_space))

    @staticmethod
    def _init_genome(observation_space: gym.Space, action_space: gym.Space) -> Genome:
        hidden_nodes = [18, 18]
        weights = [np.random.uniform(-1, 1, size=[np.product(observation_space.shape) + 1, hidden_nodes[0]]),
                   np.random.uniform(-1, 1, size=[hidden_nodes[0] + 1, hidden_nodes[1]]),
                   np.random.uniform(-1, 1, size=[hidden_nodes[1] + 1, action_space.n]),
                   ]
        return Genome(weights)

    def get_action(self, ob):
        ob_reshaped = ob.reshape([1, np.product(ob.shape)])
        h1 = sigmoid(cat_ones(ob_reshaped).dot(self.genome.values[0]))
        h2 = sigmoid(cat_ones(h1).dot(self.genome.values[1]))
        action_logits = cat_ones(h2).dot(self.genome.values[2])
        return list(Direction)[np.argmax(action_logits)]
