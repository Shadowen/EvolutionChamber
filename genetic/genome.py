from typing import List

import numpy as np


class Genome:
    def __init__(self, values: List[np.ndarray]):
        self.values = values

    def mutate(self, p: float) -> 'Genome':
        """
        Mutates a Genome in place.
        :param p: probability of point mutation.
        :returns: self for chaining.
        """

        for i in range(len(self.values)):
            mask = np.random.choice([1, 0], size=self.values[i].shape, p=[p, 1 - p])
            mutations = np.random.random(
                size=self.values[i].shape)  # TODO: Figure out a systematic way of inserting mutations.
            self.values[i] = mask * self.values[i] + (1 - mask) * mutations
        return self

    @classmethod
    def crossover(cls, a: 'Genome', b: 'Genome', p) -> 'Genome':
        """
        Crosses over two Genomes to create a third, new Genome.
        :param a: first Genome.
        :param b: second Genome.
        :param p: probability of crossover occuring.
        :return: new resultant Genome.
        """
        assert len(a.values) == len(b.values)
        assert all(np.all(x.shape == y.shape) for x, y in zip(a.values, b.values))

        # TODO: Do real crossover where you get continuous chunks crossed rather than random masks.
        c = []
        for x, y in zip(a.values, b.values):
            m = np.random.choice([1, 0], size=x.shape, p=[p, 1 - p])
            c.append(x * m + y * (1 - m))

        return Genome(c)
