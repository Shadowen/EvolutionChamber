from collections import deque
from time import time

import sequence_repeater
from genetic import Runner


class ExperimentRunner(Runner):
    @staticmethod
    def build_agent():
        env = sequence_repeater.Game(max_num=4, max_timesteps=100)
        return sequence_repeater.Agent(env=env)

    @classmethod
    def run_experiment(cls):
        from experiments.util import get_empty_data_file

        r = cls.__new__(cls)
        r.__init__(agent_builder=ExperimentRunner.build_agent, num_agents=200, num_champions=2,
                   info_file_path=get_empty_data_file('data.csv'), max_workers=8)
        generations = 10
        f_historical = deque(maxlen=5)
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
