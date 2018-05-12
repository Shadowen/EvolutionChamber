from collections import deque
from time import time
from typing import Tuple, Callable

import gym
import numpy as np

import sequence_repeater
from genetic import Runner, Genome
from numpy_util import cat_ones


class ExperimentRunner(Runner):
    @staticmethod
    def game_constructor() -> gym.Env:
        return sequence_repeater.Game(max_num=4, max_timesteps=100)

    @staticmethod
    def build_agent(observation_space: gym.Space, action_space: gym.Space) -> \
            Tuple[Callable[[Genome, np.ndarray], np.ndarray], Genome]:
        weights = [np.random.uniform(-1, 1, size=[np.product(observation_space.n) + 1, action_space.n]), ]

        def _get_action(genome: Genome, ob: np.ndarray) -> np.ndarray:
            ob_reshaped = ob.reshape([1, np.product(ob.shape)])
            action_logits = cat_ones(ob_reshaped).dot(genome.values[0])
            return np.argmax(action_logits)

        return _get_action, Genome(weights)

    @staticmethod
    def run():
        from experiments.util import get_empty_data_file

        with open(get_empty_data_file('data.csv'), 'w') as f:
            r = ExperimentRunner(num_agents=200, num_champions=20, info_file=f, max_workers=16)
            steps = 100
            f_historical = deque(maxlen=5)
            for s in range(steps):
                start_time = time()
                f = r.single_iteration()
                end_time = time()
                f_historical.append(sum(f) / len(f))
                print(f"Generation {s} \t"
                      f"Fitness: {f_historical[-1]} (moving avg. {sum(f_historical) / len(f_historical)}) "
                      f"in {end_time-start_time} s")


if __name__ == '__main__':
    ExperimentRunner.run()
