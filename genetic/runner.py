from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Callable

import gym
import numpy as np

from genetic import Agent
from genetic import Genome
from genetic.distribution import Distribution
from numpy_util import sigmoid, softmax, cat_ones
from snake import Direction
from snake import Game, DistanceObservationGame


class Runner:
    @staticmethod
    def game_constructor() -> gym.Env:
        game = DistanceObservationGame(map_size=(30, 30))
        game = FitnessWrapper(game)
        return game

    @staticmethod
    def build_agent(observation_space: gym.Space, action_space: gym.Space) -> \
            Tuple[Callable[[Genome, np.ndarray], np.ndarray], Genome]:
        hidden_nodes = 18  # TODO: Tune this.
        weights = [np.random.uniform(-1, 1, size=[np.product(observation_space.shape) + 1, hidden_nodes]),
                   np.random.uniform(-1, 1, size=[hidden_nodes + 1, action_space.n]),
                   ]

        def _get_action(genome: Genome, ob: np.ndarray) -> np.ndarray:
            ob_reshaped = ob.reshape([1, np.product(ob.shape)])
            h1 = sigmoid(cat_ones(ob_reshaped).dot(genome.values[0]))
            action_logits = softmax(cat_ones(h1).dot(genome.values[1]))
            return np.random.choice(list(Direction), p=action_logits[0])

        return _get_action, Genome(weights)

    def __init__(self, *, num_agents, num_champions, max_workers):
        self.max_workers = max_workers

        self.num_agents = num_agents
        self.num_champions = num_champions
        self.agents = [
            Agent(env_constructor=self.game_constructor, build_agent=self.build_agent) for _ in
            range(num_agents)]

    def evaluate(self) -> List[float]:
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
        population_indices =argsorted_indices[:-self.num_champions]
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

    def run_experiment(self):
        raise NotImplementedError()


class FitnessWrapper(gym.RewardWrapper):
    def __init__(self, env: Game):
        super(FitnessWrapper, self).__init__(env)
        self.env = env

    def reward(self, reward):
        snake_length = len(self.env.snake_tail)
        if snake_length < 10:
            return self.env.num_steps ** 2 * 2 ** snake_length
        else:
            return self.env.num_steps ** 2 * 2 ** 10 * (snake_length - 9)


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
