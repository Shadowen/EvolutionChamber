import copy
import threading
from collections import deque
from time import time
from typing import Tuple, Callable

import gym
import numpy as np

from experiments.util import *
from genetic import Agent
from genetic import Genome
from genetic import Runner
from gym_util.forwarding_wrappers import ForwardingRewardWrapper
from numpy_util import sigmoid, cat_ones
from snake import Game, DistanceObservationGame, Direction


class ExperimentRunner(Runner):
    @staticmethod
    def game_constructor() -> gym.Env:
        game = DistanceObservationGame(map_size=(80, 40), initial_snake_length=3)
        game = FitnessWrapper(game)
        return game

    @staticmethod
    def build_agent(observation_space: gym.Space, action_space: gym.Space) -> \
            Tuple[Callable[[Genome, np.ndarray], np.ndarray], Genome]:
        hidden_nodes = [18, 18]
        weights = [np.random.uniform(-1, 1, size=[np.product(observation_space.shape) + 1, hidden_nodes[0]]),
                   np.random.uniform(-1, 1, size=[hidden_nodes[0] + 1, hidden_nodes[1]]),
                   np.random.uniform(-1, 1, size=[hidden_nodes[1] + 1, action_space.n]),
                   ]

        def _get_action(genome: Genome, ob: np.ndarray) -> np.ndarray:
            ob_reshaped = ob.reshape([1, np.product(ob.shape)])
            h1 = sigmoid(cat_ones(ob_reshaped).dot(genome.values[0]))
            h2 = sigmoid(cat_ones(h1).dot(genome.values[1]))
            action_logits = cat_ones(h2).dot(genome.values[2])
            return list(Direction)[np.argmax(action_logits)]

        return _get_action, Genome(weights)

    @classmethod
    def run_experiment(cls):
        np.random.seed(1)

        saved_agents_dir = get_or_make_data_dir('agents')
        info_path = get_empty_data_file('data.csv')

        r = cls.__new__(cls)
        r.__init__(num_agents=2000, num_champions=20, max_workers=1, info_file_path=info_path)
        generations = 100
        f_historical = deque(maxlen=10)

        # human_display_agent: Agent = Agent(env=r.game_constructor(), build_agent=r.build_agent)
        # best_genome: Genome = human_display_agent.genome
        # best_genome_lock: threading.Lock = threading.Lock()
        #
        # def do_human_display():
        #     # Rendering...
        #     while human_display_agent.genome is not None:
        #         with best_genome_lock:
        #             human_display_agent.genome = best_genome
        #         human_display_agent.run_iteration(render=True)
        #
        # human_display_thread = threading.Thread(target=do_human_display)
        # human_display_thread.start()

        for s in range(1, generations + 1):
            start_time = time()
            f = r.single_iteration()
            end_time = time()
            f_historical.append(sum(f) / len(f))
            print(f"Generation {s} \t"
                  f"Fitness: {f_historical[-1]} (moving avg. {sum(f_historical) / len(f_historical)})\t"
                  f"Best: {max(f)}\t"
                  f"in {end_time-start_time} s")
            # with best_genome_lock:
            #     best_genome = copy.deepcopy(r.agents[np.argmax(r.fitnesses)].genome)

            # Remove the old agents and save the current ones.
            r.save_agents(directory=saved_agents_dir, overwrite=True)

        # Wait for threads to terminate.
        # human_display_thread.join()


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
            return (self.env.timesteps ** 2) * (2 ** snake_length)
        else:
            return (self.env.timesteps ** 2) * (2 ** 10) * (snake_length - 9)


if __name__ == '__main__':
    ExperimentRunner.run_experiment()
