from collections import deque
from time import time

from genetic import Runner


class ExperimentRunner(Runner):

    @staticmethod
    def run():
        r = ExperimentRunner(num_agents=2000, max_workers=16)
        steps = 10000
        f_historical = deque(maxlen=100)
        for s in range(steps):
            start_time = time()
            f = r.single_iteration()
            end_time = time()
            f_historical.append(max(f))
            print(f"Generation {s} \t"
                  f"Fitness: {f_historical[-1]} (moving avg. {sum(f_historical) / len(f_historical)}) "
                  f"in {end_time-start_time} s")


if __name__ == '__main__':
    ExperimentRunner.run()
