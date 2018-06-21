from collections import deque
from time import time

import gym
import numpy as np

from experiments import g_smaller_room
from experiments.util import *
from gym_util.forwarding_wrappers import ForwardingRewardWrapper
from snake import Game, DistanceObservationGame


class ExperimentRunner(g_smaller_room.ExperimentRunner):

    @staticmethod
    def game_constructor() -> gym.Env:
        game = DistanceObservationGame(map_size=(20, 20), initial_snake_length=3)
        game = FitnessWrapper(game)
        return game

    @classmethod
    def run_experiment(cls):
        np.random.seed(1)
        saved_agents_dir = get_or_make_data_dir('agents')
        info_path = get_empty_data_file('data.csv')

        r = cls.__new__(cls)
        r.__init__(num_agents=2000, num_champions=20, max_workers=1, info_file_path=info_path)
        r.load_agents(directory="/home/wesley/data/evolution_chamber/g_smaller_room.py/agents/", method='MATCHING')

        generations = 20
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

            # Remove the old agents and save the current ones.
            r.save_agents(directory=saved_agents_dir, overwrite=True)


class FitnessWrapper(ForwardingRewardWrapper):
    def __init__(self, env: Game):
        super(FitnessWrapper, self).__init__(env)
        self.env = env

    @property
    def info_fields(self):
        return self.env.info_fields

    def reward(self, reward):
        snake_length = len(self.env.snake_tail) + 1
        if snake_length < 10:
            return (self.env.timesteps) * (2 ** snake_length)
        else:
            return (self.env.timesteps) * (2 ** 10) * (snake_length - 9)


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
