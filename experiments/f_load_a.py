from collections import deque
from time import time

import numpy as np

from experiments import a_basic_experiment
from experiments.util import *


class ExperimentRunner(a_basic_experiment.ExperimentRunner):
    @classmethod
    def run_experiment(cls):
        np.random.seed(1)
        info_path = get_empty_data_file('data.csv')

        r = cls.__new__(cls)
        r.__init__(num_agents=200, num_champions=20, max_workers=1, info_file_path=info_path)
        r.load_agents(directory="/home/wesley/data/evolution_chamber/a_basic_experiment (save).py/agents/",
                      method='MATCHING')

        generations = 10
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


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
