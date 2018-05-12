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

    @classmethod
    def run(cls):
        from experiments.util import get_empty_data_file

        with open(get_empty_data_file('data.csv'), 'w') as f:
            r = cls.__new__(cls)
            r.__init__(num_agents=200, num_champions=20, max_workers=1, info_file=f)
            steps = 100
            f_historical = deque(maxlen=10)

            for s in range(1, steps + 1):
                start_time = time()
                f = r.single_iteration()
                end_time = time()
                f_historical.append(sum(f) / len(f))
                print(f"Generation {s} \t"
                      f"Fitness: {f_historical[-1]} (moving avg. {sum(f_historical) / len(f_historical)})\t"
                      f"Best: {max(f)}\t"
                      f"in {end_time-start_time} s")


class FitnessWrapper(gym.RewardWrapper):
    def __init__(self, env: Game):
        super(FitnessWrapper, self).__init__(env)
        self.env = env

    # TODO: Replace this with a __missing__ attribute?
    @property
    def info_fields(self):
        return self.env.info_fields

    def reward(self, reward):
        snake_length = len(self.env.snake_tail)
        if snake_length < 10:
            return self.env.num_steps ** 2 * 2 ** (snake_length - 2)
        else:
            return self.env.num_steps ** 2 * 2 ** 10 * (snake_length - 9)


if __name__ == '__main__':
    ExperimentRunner.run()
