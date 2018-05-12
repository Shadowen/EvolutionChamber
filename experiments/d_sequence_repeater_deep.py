from typing import Tuple, Callable

import gym
import numpy as np

from experiments import b_sequence_repeater
from genetic import Genome
from numpy_util import cat_ones, sigmoid


class ExperimentRunner(b_sequence_repeater.ExperimentRunner):
    @staticmethod
    def build_agent(observation_space: gym.Space, action_space: gym.Space) -> \
            Tuple[Callable[[Genome, np.ndarray], np.ndarray], Genome]:
        weights = [np.random.uniform(-1, 1, size=[np.product(observation_space.n) + 1, 10]),
                   np.random.uniform(-1, 1, size=[10 + 1, action_space.n])]

        def _get_action(genome: Genome, ob: np.ndarray) -> np.ndarray:
            ob_reshaped = ob.reshape([1, np.product(ob.shape)])
            h1 = sigmoid((cat_ones(ob_reshaped).dot(genome.values[0])))
            action_logits = sigmoid((cat_ones(h1).dot(genome.values[1])))
            return np.argmax(action_logits)

        return _get_action, Genome(weights)


if __name__ == '__main__':
    ExperimentRunner.run()
