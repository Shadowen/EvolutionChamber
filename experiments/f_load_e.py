from time import time

import numpy as np

import snake
from experiments import a_basic_experiment
from experiments.util import *
from snake import DistanceObservationGame


class ExperimentRunner(a_basic_experiment.ExperimentRunner):
    @staticmethod
    def build_agent():
        game = Game(map_size=(20, 20))
        return snake.Agent(env=game, hidden_nodes=[18, 18])

    @classmethod
    def run_experiment(cls):
        np.random.seed(1)
        saved_agents_dir = get_or_make_data_dir('agents')
        # TODO(wheung): Figure out how to log data.
        # info_path = get_empty_data_file('data.csv')
        info_path = None

        r = cls.__new__(cls)
        r.__init__(agent_builder=cls.build_agent, num_agents=2000, num_champions=20, max_workers=8,
                   info_file_path=info_path)
        r.load_agents(directory="/home/wesley/data/evolution_chamber/e_survival_experiment.py/agents/",
                      method='MATCHING')
        generations = 100

        for s in range(1, generations + 1):
            start_time = time()
            f, info = r.evaluate()
            end_time = time()
            avg_fitness = sum(f) / len(f)
            print(f"Generation {s} \t"
                  f"Fitness: {avg_fitness}\t"
                  f"Best: {max(f)}\t"
                  f"in {end_time-start_time} s")

            # Record info to log.
            r.record_info()
            # Remove the old agents and save the current ones.
            r.save_agents(directory=saved_agents_dir, overwrite=True)

            # Breed next generation.
            r.breed()


class Game(DistanceObservationGame):
    timestep_elbow = 30
    length_elbow = 5

    def reward(self, reward=0):
        snake_length = self.snake_length + 1 - 4
        if self.timesteps <= self.timestep_elbow:
            r1 = self.timesteps ** 2
        else:
            r1 = (self.timestep_elbow ** 2) + (self.timesteps - self.timestep_elbow)
        if snake_length <= self.length_elbow:
            r2 = 3 ** snake_length
        else:
            r2 = (3 ** self.length_elbow) * (snake_length - self.length_elbow)
        return r1 * r2


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
