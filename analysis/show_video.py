import numpy as np

from experiments.a_basic_experiment import ExperimentRunner
from experiments.util import *


class VideoRunner(ExperimentRunner):
    @classmethod
    def run_experiment(cls):
        np.random.seed(1)
        info_path = get_empty_data_file('data.csv')

        r = cls.__new__(cls)
        r.__init__(num_agents=2000, num_champions=20, max_workers=1, info_file_path=info_path)
        r.load_agents(directory="/home/wesley/data/evolution_chamber/a_basic_experiment (long).py/agents/",
                      method='SORTED')
        for a in r.agents:
            a.run_iteration(render=True)


if __name__ == '__main__':
    VideoRunner.run_experiment()
