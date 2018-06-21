from collections import deque
from time import time
from typing import Tuple, Callable

import gym
import numpy as np

from experiments.util import *
from genetic import Genome
from genetic import Runner
from gym_util.forwarding_wrappers import ForwardingRewardWrapper
from numpy_util import sigmoid, cat_ones
from snake import Game, DistanceObservationGame, Direction


class ExperimentRunner(Runner):
    @staticmethod
    def game_constructor() -> gym.Env:
        game = DistanceObservationGame(map_size=(20, 20), initial_snake_length=3)
        game = FitnessWrapper(game)
        return game

    @staticmethod
    def build_agent(observation_space: gym.Space, action_space: gym.Space) -> \
            Tuple[Callable[[Genome, np.ndarray], np.ndarray], Genome]:
        hidden_nodes = [18, 18]
        all_layer_sizes = [np.product(observation_space.shape)] + hidden_nodes + [action_space.n]
        weights = [np.random.uniform(-1, 1, size=[i + 1, o]) for (i, o) in
                   zip(all_layer_sizes[:-1], all_layer_sizes[1:])]

        def _get_action(genome: Genome, ob: np.ndarray) -> np.ndarray:
            ob_reshaped = ob.reshape([1, np.product(ob.shape)])
            h = ob_reshaped
            for i in range(len(genome.values) - 1):
                h = sigmoid(cat_ones(h).dot(genome.values[i]))
            action_logits = cat_ones(h).dot(genome.values[-1])
            return list(Direction)[np.argmax(action_logits)]

        return _get_action, Genome(weights)

    @classmethod
    def run_experiment(cls):
        np.random.seed(1)

        saved_agents_dir = get_or_make_data_dir('agents')
        info_path = get_empty_data_file('data.csv')

        r = cls.__new__(cls)
        r.__init__(num_agents=2000, num_champions=20, max_workers=1, info_file_path=info_path)
        generations = 100
        f_historical = deque(maxlen=10)

        for s in range(1, generations + 1):
            start_time = time()
            f = r.single_iteration()
            end_time = time()
            f_historical.append(sum(f) / len(f))
            print(f"Generation {s} \t"
                  f"Fitness: {f_historical[-1]} (moving avg. {sum(f_historical) / len(f_historical)})\t"
                  f"Best: {max(f)}\t"
                  f"in {end_time-start_time} s")

            # Remove the old agents and save the current ones.
            r.save_agents(directory=saved_agents_dir, overwrite=True)


class FitnessWrapper(ForwardingRewardWrapper):
    def __init__(self, env: Game):
        super(FitnessWrapper, self).__init__(env)
        self.env = env

    @property
    def info_fields(self):
        return self.env.info_fields

    def reward(self, reward):
        snake_length = len(self.env.snake_tail) + 1
        if snake_length < 10:
            return (self.env.timesteps ** 2) * (2 ** snake_length)
        else:
            return (self.env.timesteps ** 2) * (2 ** 10) * (snake_length - 9)


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
