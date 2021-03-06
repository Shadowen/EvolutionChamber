import numpy as np

from experiments.g_smaller_room import ExperimentRunner
from experiments.util import *
import snake

class VideoRunner(ExperimentRunner):
    @classmethod
    def run_experiment(cls):
        np.random.seed(1)
        # info_path = get_empty_data_file('data.csv')
        info_path = None

        r = cls.__new__(cls)
        r.__init__(agent_builder=cls.build_agent, num_agents=2000, num_champions=20, max_workers=1,
                   info_file_path=info_path)
        r.load_agents(directory="/home/wesley/data/evolution_chamber/g_smaller_room.py/agents/",
                      method='SORTED')
        for a in r.agents:
            f, info = a.run_iteration(render=True)
            print(f"{f}, {info}")


if __name__ == '__main__':
    VideoRunner.run_experiment()
