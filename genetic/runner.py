import csv
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Dict, Any, List

import gym
import numpy as np

from genetic import Agent, Genome
from numpy_util import Distribution


class Runner:
    @staticmethod
    @abstractmethod
    def game_constructor() -> gym.Env:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def build_agent(observation_space: gym.Space, action_space: gym.Space):
        raise NotImplementedError()

    def __init__(self, *, num_agents: int, num_champions: int, info_file_path: str = None, max_workers: int = 1):

        self.num_agents = num_agents
        self.num_champions = num_champions
        self.agents = [
            Agent(env=self.game_constructor(), build_agent=self.build_agent) for _ in
            range(num_agents)]
        # self.agents[0].env = gym.wrappers.Monitor(self.agents[0].env, directory=experiments.util.get_or_make_data_dir(),
        #                                           video_callable=lambda e: True, force=True)
        self.envType = self.agents[0].env

        self.generation = 0

        self.info_file_writer = None
        if info_file_path is not None:
            self.info_file = open(info_file_path, 'w', buffering=1)
            self.info_file_writer = csv.DictWriter(self.info_file, ['generation'] + self.envType.info_fields)
            self.info_file_writer.writeheader()
        self.max_workers = max_workers

    def evaluate(self) -> Iterable[Any]:
        if self.max_workers == 1:
            fitnesses = []
            num_agents = len(self.agents)
            for i, a in enumerate(self.agents):
                fitnesses.append(a.run_iteration())
                print(f"\rEvaluating... {i}/{num_agents} ", end="")
            print("\r", end="")
            return zip(*fitnesses)
        else:
            # TODO: See if parallelization actually helps...
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                generator = executor.map(lambda a: a.run_iteration(), self.agents)
            return zip(*generator)

    def do_selection(self, fitnesses: Iterable[float]) -> None:
        """Modify the genomes of the agents to create the next generation."""
        # Crossover and mutation.
        current_genomes = [a.genome for a in self.agents]
        # Filter out champions.
        argsorted_indices = np.argsort(fitnesses)
        champion_indices = argsorted_indices[-self.num_champions:]
        population_indices = argsorted_indices[:-self.num_champions]
        new_genomes = [current_genomes[i] for i in champion_indices]
        # Breed remaining population weighted by fitness.
        d = Distribution(fitnesses, current_genomes)
        for _ in population_indices:
            a, b = d.sample(n=2)
            new_genomes.append(Genome.crossover(a, b).mutate(p=0.01))
        # Assign Genomes to Agents.
        for a, g in zip(self.agents, new_genomes):
            a.genome = g

    def record_info(self, generation: int, info: Dict) -> None:
        """Record the given info dict to disk."""
        if self.info_file_writer is not None:
            d = {'generation': generation}
            for a, k in enumerate(self.envType.info_fields):
                d[k] = [i[a] for i in info]
            self.info_file_writer.writerow(d)

    def single_iteration(self) -> List[float]:
        """
        Run one iteration of GA and return the fitnesses of the population.
        :returns a list of the fitnesses of the agents currently in this population.
        """
        self.generation += 1

        fitnesses, info = self.evaluate()
        self.do_selection(fitnesses)
        self.record_info(self.generation, info)
        return fitnesses

    @classmethod
    @abstractmethod
    def run_experiment(cls) -> None:
        raise NotImplementedError()
