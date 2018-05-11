from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List

import gym
import numpy as np

from genetic import Agent, Genome
from genetic.distribution import Distribution


class Runner:
    @staticmethod
    @abstractmethod
    def game_constructor() -> gym.Env:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def build_agent(observation_space: gym.Space, action_space: gym.Space):
        raise NotImplementedError()

    def __init__(self, *, num_agents, num_champions, max_workers):
        self.max_workers = max_workers

        self.num_agents = num_agents
        self.num_champions = num_champions
        self.agents = [
            Agent(env_constructor=self.game_constructor, build_agent=self.build_agent) for _ in
            range(num_agents)]

    def evaluate(self) -> List[float]:
        if self.max_workers == 1:
            generator = (a.run_iteration() for a in self.agents)
        else:
            # TODO: See if parallelization actually helps...
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                generator = executor.map(lambda a: a.run_iteration(), self.agents)
        return list(generator)

    def single_iteration(self):
        fitnesses = self.evaluate()

        ## Crossover and mutation.
        current_genomes = [a.genome for a in self.agents]
        # Filter out champions.
        argsorted_indices = np.argsort(fitnesses)
        champion_indices = argsorted_indices[-self.num_champions:]
        population_indices = argsorted_indices[:-self.num_champions]
        new_genomes = [current_genomes[i] for i in champion_indices]
        # Breed remaining population weighted by fitness.
        d = Distribution(fitnesses, current_genomes)
        for i in population_indices:
            a, b = d.sample(n=2)
            new_genomes.append(Genome.crossover(a, b).mutate(p=0.01))
        # Assign Genomes to Agents.
        for a, g in zip(self.agents, new_genomes):
            a.genome = g

        return fitnesses

    @staticmethod
    @abstractmethod
    def run_experiment(self):
        raise NotImplementedError()


if __name__ == '__main__':
    def moving_average(data, N):
        import numpy as np
        return np.convolve(data, np.ones((N,)) / N, mode='valid')


    r = Runner(num_agents=10, max_workers=2)
    f_historical = []
    for _ in range(100):
        f = r.single_iteration()
        f_historical.append(sum(f))
    print(moving_average(f_historical, N=100))
