from typing import Tuple, Callable

import gym
import numpy as np

from experiments import b_sequence_repeater
from genetic import Genome
from numpy_util import cat_ones, softmax


class ExperimentRunner(b_sequence_repeater.ExperimentRunner):
    @staticmethod
    def build_agent(observation_space: gym.Space, action_space: gym.Space) -> \
            Tuple[Callable[[Genome, np.ndarray], np.ndarray], Genome]:
        weights = [np.random.uniform(-1, 1, size=[np.product(observation_space.n) + 1, action_space.n]), ]

        def _get_action(genome: Genome, ob: np.ndarray) -> np.ndarray:
            ob_reshaped = ob.reshape([1, np.product(ob.shape)])
            action_logits = softmax((cat_ones(ob_reshaped).dot(genome.values[0])), temperature=1E-3)
            return np.random.choice(np.arange(action_space.n), p=action_logits[0])

        return _get_action, Genome(weights)


if __name__ == '__main__':
    ExperimentRunner.run()
