from typing import Tuple, Callable

import gym
import numpy as np

from experiments import g_smaller_room
from genetic import Genome
from numpy_util import sigmoid, cat_ones
from snake import DistanceObservationGame, Direction


class ExperimentRunner(g_smaller_room.ExperimentRunner):
    @staticmethod
    def game_constructor() -> gym.Env:
        game = DistanceObservationGame(map_size=(20, 20), initial_snake_length=3)
        game = g_smaller_room.FitnessWrapper(game)
        return game

    @staticmethod
    def build_agent(observation_space: gym.Space, action_space: gym.Space) -> \
            Tuple[Callable[[Genome, np.ndarray], np.ndarray], Genome]:
        hidden_nodes = [18]
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


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
