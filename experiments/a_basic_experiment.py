from collections import deque
from time import time
from typing import Tuple, Callable

import gym
import numpy as np

from genetic import Genome
from genetic import Runner
from numpy_util import sigmoid, cat_ones
from snake import Game, DistanceObservationGame, Direction


class ExperimentRunner(Runner):
    @staticmethod
    def game_constructor() -> gym.Env:
        game = DistanceObservationGame(map_size=(30, 30))
        game = FitnessWrapper(game)
        return game

    @staticmethod
    def build_agent(observation_space: gym.Space, action_space: gym.Space) -> \
            Tuple[Callable[[Genome, np.ndarray], np.ndarray], Genome]:
        hidden_nodes = 18  # TODO: Tune this.
        weights = [np.random.uniform(-1, 1, size=[np.product(observation_space.shape) + 1, hidden_nodes]),
                   np.random.uniform(-1, 1, size=[hidden_nodes + 1, action_space.n]),
                   ]

        def _get_action(genome: Genome, ob: np.ndarray) -> np.ndarray:
            ob_reshaped = ob.reshape([1, np.product(ob.shape)])
            h1 = sigmoid(cat_ones(ob_reshaped).dot(genome.values[0]))
            action_logits = cat_ones(h1).dot(genome.values[1])
            return list(Direction)[np.argmax(action_logits)]

        return _get_action, Genome(weights)

    @staticmethod
    def run():
        r = ExperimentRunner(num_agents=200, num_champions=20, max_workers=1)
        steps = 100
        f_historical = deque(maxlen=10)

        for s in range(steps):
            start_time = time()
            f = r.single_iteration()
            end_time = time()
            f_historical.append(max(f))
            print(f"Generation {s} \t"
                  f"Fitness: {f_historical[-1]} (moving avg. {sum(f_historical) / len(f_historical)}) "
                  f"in {end_time-start_time} s")


class FitnessWrapper(gym.RewardWrapper):
    def __init__(self, env: Game):
        super(FitnessWrapper, self).__init__(env)
        self.env = env

    def reward(self, reward):
        snake_length = len(self.env.snake_tail)
        if snake_length < 10:
            return self.env.num_steps ** 2 * 2 ** snake_length
        else:
            return self.env.num_steps ** 2 * 2 ** 10 * (snake_length - 9)


if __name__ == '__main__':
    ExperimentRunner.run()
