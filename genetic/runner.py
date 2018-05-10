from typing import List

import tensorflow as tf

from genetic import Agent
from genetic import Genome
from genetic.distribution import Distribution
from gym_util import TimestepRewardWrapper
from snake import RGBObservationGame


class Runner:
    def __init__(self, *, num_agents):
        def game_constructor():
            game = RGBObservationGame(map_size=(10, 10))
            game = TimestepRewardWrapper(game, 0.1)
            return game

        self.sess = tf.Session()

        self.num_agents = num_agents
        self.agents = [Agent(env_constructor=game_constructor, sess=self.sess) for _ in range(num_agents)]

    def __enter__(self) -> 'Runner':
        self.sess.__enter__()
        self.sess.run(tf.global_variables_initializer())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.sess.__exit__(exc_type, exc_value, traceback)

    def evaluate(self) -> List[float]:
        # TODO: Parallelize this.
        fitnesses = [float('-inf') for _ in self.agents]
        for i, a in enumerate(self.agents):
            fitnesses[i] = a.run_iteration()
        return fitnesses

    def do_crossover(self, fitnesses):
        current_genomes = [a.get_genome() for a in self.agents]
        new_genomes = []

        # Breed current generation weighted by fitnesses.
        d = Distribution(fitnesses, current_genomes)
        for i in range(self.num_agents):
            a, b = d.sample(n=2)
            new_genomes.append(Genome.crossover(a, b, p=0.5))  # TODO: Crossover probability

        # Apply new genomes.
        for a, g in zip(self.agents, new_genomes):
            a.genome = g

    def do_mutate(self, fitnesses):
        for a in self.agents:
            a.set_genome(a.get_genome().mutate(p=0.1))

    def single_iteration(self):
        fitnesses = self.evaluate()
        self.do_mutate(fitnesses)
        self.do_crossover(fitnesses)


if __name__ == '__main__':
    with Runner(num_agents=10) as r:
        r.single_iteration()
