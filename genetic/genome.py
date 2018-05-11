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
            mutations = np.clip(np.random.standard_normal(size=self.values[i].shape) / 5, -1, 1)
            self.values[i] = mask * self.values[i] + (1 - mask) * mutations
        return self

    @classmethod
    def crossover(cls, a: 'Genome', b: 'Genome') -> 'Genome':
        """
        Crosses over two Genomes to create a third, new Genome.
        :param a: first Genome.
        :param b: second Genome.
        :param p: probability of crossover occurring.
        :return: new resultant Genome.
        """
        assert len(a.values) == len(b.values)
        assert all(np.all(x.shape == y.shape) for x, y in zip(a.values, b.values))

        c = []
        for x, y in zip(a.values, b.values):
            randR = np.random.randint(x.shape[0])
            randC = np.random.randint(x.shape[1])

            z = np.empty_like(x)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    if i < randR or (i == randR and j <= randC):
                        z[i, j] = x[i, j]
                    else:
                        z[i, j] = y[i, j]
                c.append(z)

        return Genome(c)
