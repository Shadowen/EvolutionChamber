from concurrent.futures import ThreadPoolExecutor
from typing import List

import tensorflow as tf

from genetic import Agent
from genetic import Genome
from genetic.distribution import Distribution
from gym_util import TimestepRewardWrapper
from snake import RGBObservationGame


class Runner:
    @staticmethod
    def game_constructor():
        game = RGBObservationGame(map_size=(10, 10))
        game = TimestepRewardWrapper(game, 0.1)
        return game

    @staticmethod
    def build_agent(input_placeholder, action_space):
        observation_reshaped = tf.reshape(input_placeholder, shape=[1, -1])
        h1 = tf.layers.dense(inputs=observation_reshaped, units=5, activation=tf.sigmoid)
        h2 = tf.layers.dense(inputs=h1, units=8, activation=tf.sigmoid)
        output = tf.layers.dense(inputs=h2, units=action_space.n, activation=tf.nn.softmax)
        return output

    def __init__(self, *, num_agents, max_workers):
        self.max_workers = max_workers
        self.sess = tf.Session()

        self.num_agents = num_agents
        self.agents = [
            Agent(env_constructor=self.game_constructor, build_agent=self.build_agent, sess=self.sess) for _ in
            range(num_agents)]

    def __enter__(self) -> 'Runner':
        self.sess.__enter__()
        self.sess.run(tf.global_variables_initializer())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.__exit__(exc_type, exc_value, traceback)

    def evaluate(self) -> List[float]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            generator = executor.map(lambda a: a.run_iteration(), self.agents)
        return list(generator)

    def single_iteration(self):
        fitnesses = self.evaluate()

        ## Crossover.
        current_genomes = [a.get_genome() for a in self.agents]
        new_genomes = []
        # Breed current generation weighted by fitnesses.
        d = Distribution(fitnesses, current_genomes)
        for i in range(self.num_agents):
            a, b = d.sample(n=2)
            new_genomes.append(Genome.crossover(a, b))

        ## Mutation.
        for a, g in zip(self.agents, new_genomes):
            a.set_genome(g.mutate(p=0.01))

        return fitnesses

    def run_experiment(self):
        raise NotImplementedError()


if __name__ == '__main__':
    def moving_average(data, N):
        import numpy as np
        return np.convolve(data, np.ones((N,)) / N, mode='valid')


    with Runner(num_agents=10) as r:
        f_historical = []
        for _ in range(1000):
            f = r.single_iteration()
            f_historical.append(sum(f))
        print(moving_average(f_historical, N=100))
